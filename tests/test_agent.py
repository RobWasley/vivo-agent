import json
import time

from app import config, tools
from app.agent import Agent, ReasoningDelta, ToolRound


def _sse(*events):
    return [f"data: {json.dumps(ev)}" for ev in events] + ["data: [DONE]"]


def _text_delta(text):
    return {"choices": [{"delta": {"content": text}}]}


def _tool_calls_delta(calls):
    tcs = []
    for i, (cid, name, args) in enumerate(calls):
        tcs.append(
            {"index": i, "id": cid, "type": "function",
             "function": {"name": name, "arguments": args}}
        )
    return {"choices": [{"delta": {"tool_calls": tcs}}]}


def _reasoning_delta(text, key="reasoning"):
    return {"choices": [{"delta": {key: text}}]}


class FakeStreamResult:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class FakeClient:
    """Stands in for httpx.Client: scripted streams, captured payloads."""

    def __init__(self, streams=(), post_body=None):
        self.streams = list(streams)
        self.post_body = post_body
        self.payloads = []

    def stream(self, method, url, json=None):
        self.payloads.append(json)
        return FakeStreamResult(self.streams.pop(0))

    def post(self, url, json=None):
        self.payloads.append(json)
        return FakeResponse(self.post_body)


def test_no_thinking_and_tool_roundtrip():
    with Agent(
        config.LLM_BASE_URL, config.LLM_MODEL, config.PERSONA, tools=tools.TOOLS
    ) as agent:
        chunks = []
        rounds = 0
        for item in agent.reply(
            "What time is it? Use the get_time tool to check, then answer.",
            tools.execute,
        ):
            if isinstance(item, str):
                chunks.append(item)
            else:
                assert isinstance(item, ToolRound)
                rounds += 1
    text = "".join(chunks).strip()
    assert rounds >= 1, "expected at least one tool round-trip"
    assert text, "final answer was empty"
    low = text.lower()
    assert "<think" not in low and "think>" not in low, f"thinking leaked: {text!r}"


def test_thinking_mode_streams_reasoning_live():
    """Live: with thinking on, reasoning streams before the spoken answer and
    never leaks into it."""
    with Agent(
        config.LLM_BASE_URL, config.LLM_MODEL, config.PERSONA,
        tools=tools.TOOLS, thinking=True,
    ) as agent:
        items = list(agent.reply(
            "What is 17 times 24? Answer with the number only.", tools.execute
        ))
    reasoning = "".join(i.text for i in items if isinstance(i, ReasoningDelta))
    text = "".join(i for i in items if isinstance(i, str)).strip()
    assert reasoning, "thinking mode streamed no reasoning deltas"
    assert text, "final answer was empty"
    low = text.lower()
    assert "<think" not in low and "think>" not in low, f"thinking leaked: {text!r}"


def test_reply_sends_history():
    client = FakeClient(streams=[_sse(_text_delta("hi"))])
    agent = Agent("http://x/v1", "m", "P", client=client)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    out = list(agent.reply("now?", lambda n, a: "r", history=history))
    assert out == ["hi"]
    messages = client.payloads[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "earlier question"}
    assert messages[2] == {"role": "assistant", "content": "earlier answer"}
    assert messages[-1] == {"role": "user", "content": "now?"}


def test_reply_without_history_unchanged():
    client = FakeClient(streams=[_sse(_text_delta("hi"))])
    agent = Agent("http://x/v1", "m", "P", client=client)
    list(agent.reply("hello", lambda n, a: "r"))
    messages = client.payloads[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]


def test_user_profile_appended_to_system_prompt():
    client = FakeClient(streams=[_sse(_text_delta("hi"))])
    agent = Agent(
        "http://x/v1", "m", "P", system_prompt="S",
        user_profile="User profile: Your user's name is Rob.", client=client,
    )
    list(agent.reply("hello", lambda n, a: "r"))
    system = client.payloads[0]["messages"][0]["content"]
    assert system == "P\nS\nUser profile: Your user's name is Rob."


def test_system_prompt_without_user_profile_unchanged():
    client = FakeClient(streams=[_sse(_text_delta("hi"))])
    agent = Agent("http://x/v1", "m", "P", system_prompt="S", client=client)
    list(agent.reply("hello", lambda n, a: "r"))
    assert client.payloads[0]["messages"][0]["content"] == "P\nS"


def test_user_profile_from_config(monkeypatch):
    from app import pipeline

    monkeypatch.setattr(config, "USER_NAME", "Alex")
    monkeypatch.setattr(config, "USER_LOCATION", "Bristol, UK")
    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    monkeypatch.setattr(config, "USER_UNITS", "imperial")
    profile = pipeline.user_profile()
    for frag in ("Alex", "Bristol, UK", "Europe/London", "imperial"):
        assert frag in profile, profile


def test_user_profile_empty_when_unset(monkeypatch):
    from app import pipeline

    monkeypatch.setattr(config, "USER_NAME", "")
    monkeypatch.setattr(config, "USER_LOCATION", "")
    monkeypatch.setattr(config, "USER_TIMEZONE", "")
    monkeypatch.setattr(config, "USER_UNITS", "metric")
    assert pipeline.user_profile() == ""


def test_tool_calls_run_in_parallel_and_errors_are_feedback():
    client = FakeClient(
        streams=[
            _sse(_tool_calls_delta([("c1", "slow", "{}"), ("c2", "slow", "{}"),
                                    ("c3", "boom", "{}")])),
            _sse(_text_delta("done")),
        ]
    )
    agent = Agent("http://x/v1", "m", "P", client=client)

    def execute(name, args):
        if name == "boom":
            raise RuntimeError("kapow")
        time.sleep(0.3)
        return "ok"

    t0 = time.monotonic()
    items = list(agent.reply("go", execute))
    elapsed = time.monotonic() - t0
    assert [i for i in items if isinstance(i, str)] == ["done"]
    assert sum(1 for i in items if isinstance(i, ToolRound)) == 1
    # two 0.3 s sleeps overlapped (sequential would be >= 0.6 s)
    assert elapsed < 0.55, f"tool round took {elapsed:.2f}s, not parallel"
    second = client.payloads[1]["messages"]
    assistant = [m for m in second if m["role"] == "assistant"][-1]
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["c1", "c2", "c3"]
    tool_msgs = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2", "c3"]
    assert tool_msgs[0]["content"] == "ok"
    assert tool_msgs[2]["content"].startswith("error: RuntimeError: kapow")


def test_max_tool_rounds_stops_loop():
    tool_round = _sse(_tool_calls_delta([("c1", "slow", "{}")]))
    client = FakeClient(streams=[tool_round] * 10)
    agent = Agent("http://x/v1", "m", "P", client=client, max_tool_rounds=2)
    rounds = sum(1 for item in agent.reply("go", lambda n, a: "ok")
                 if isinstance(item, ToolRound))
    assert rounds == 2
    assert len(client.payloads) == 3  # initial + 2 re-calls


def test_multi_round_tools_loop(tmp_path, monkeypatch):
    """Two consecutive tool rounds through the real dispatcher (offline)."""
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    client = FakeClient(
        streams=[
            _sse(_tool_calls_delta([("c1", "exec", '{"command": "echo step-one"}')])),
            _sse(_tool_calls_delta([("c2", "exec", '{"command": "echo step-two"}')])),
            _sse(_text_delta("all done")),
        ]
    )
    agent = Agent("http://x/v1", "m", "P", client=client)
    items = list(agent.reply("do it", tools.execute))
    assert [i for i in items if isinstance(i, str)] == ["all done"]
    assert sum(1 for i in items if isinstance(i, ToolRound)) == 2
    first_round = [m for m in client.payloads[1]["messages"] if m["role"] == "tool"]
    assert any("step-one" in m["content"] for m in first_round)
    second_round = [m for m in client.payloads[2]["messages"] if m["role"] == "tool"]
    assert any("step-two" in m["content"] for m in second_round)


def test_summarize_is_non_streaming_without_tools():
    client = FakeClient(
        post_body={"choices": [{"message": {"content": "the summary"}}]}
    )
    agent = Agent("http://x/v1", "m", "P", client=client)
    assert agent.summarize([{"role": "user", "content": "hi"}]) == "the summary"
    payload = client.payloads[0]
    assert payload["stream"] is False
    assert "tools" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


def test_thinking_toggle_in_payload():
    on = FakeClient(streams=[_sse(_text_delta("hi"))])
    list(Agent("http://x/v1", "m", "P", client=on, thinking=True).reply(
        "hello", lambda n, a: "r"
    ))
    assert on.payloads[0]["chat_template_kwargs"] == {"enable_thinking": True}

    off = FakeClient(streams=[_sse(_text_delta("hi"))])
    list(Agent("http://x/v1", "m", "P", client=off).reply("hello", lambda n, a: "r"))
    assert off.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_reasoning_deltas_surfaced_separately_from_text():
    client = FakeClient(
        streams=[
            _sse(
                _reasoning_delta("hmm "),
                _reasoning_delta("let me think."),
                _text_delta("the answer"),
            )
        ]
    )
    agent = Agent("http://x/v1", "m", "P", client=client, thinking=True)
    items = list(agent.reply("why?", lambda n, a: "r"))
    reasoning = [i.text for i in items if isinstance(i, ReasoningDelta)]
    assert reasoning == ["hmm ", "let me think."]
    # spoken text stays clean: no reasoning mixed into the str deltas
    assert [i for i in items if isinstance(i, str)] == ["the answer"]
    assert not any("hmm" in i for i in items if isinstance(i, str))


def test_reasoning_content_key_also_surfaced():
    client = FakeClient(
        streams=[
            _sse(
                _reasoning_delta("openai-style", key="reasoning_content"),
                _text_delta("ok"),
            )
        ]
    )
    agent = Agent("http://x/v1", "m", "P", client=client, thinking=True)
    items = list(agent.reply("why?", lambda n, a: "r"))
    assert [i.text for i in items if isinstance(i, ReasoningDelta)] == [
        "openai-style"
    ]
    assert [i for i in items if isinstance(i, str)] == ["ok"]


def test_reasoning_across_tool_rounds_stays_separate():
    """Thinking before a tool round and before the final answer must both be
    surfaced as ReasoningDelta, with ToolRound markers intact."""
    client = FakeClient(
        streams=[
            _sse(
                _reasoning_delta("need the time."),
                _tool_calls_delta([("c1", "slow", "{}")]),
            ),
            _sse(
                _reasoning_delta("now to answer."),
                _text_delta("done"),
            ),
        ]
    )
    agent = Agent("http://x/v1", "m", "P", client=client, thinking=True)
    items = list(agent.reply("go", lambda n, a: "ok"))
    assert [i.text for i in items if isinstance(i, ReasoningDelta)] == [
        "need the time.",
        "now to answer.",
    ]
    assert sum(1 for i in items if isinstance(i, ToolRound)) == 1
    assert [i for i in items if isinstance(i, str)] == ["done"]
