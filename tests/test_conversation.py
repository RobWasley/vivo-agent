import json
import threading
import time

from app.conversation import Conversation


def test_add_turn_and_messages(tmp_path):
    c = Conversation(
        data_path=str(tmp_path / "conv.json"),
        compact_after_chars=100,
        keep_recent_turns=2,
    )
    c.add_turn("hello", "hi there")
    c.add_turn("what did I say?", "you said hello")
    c.add_turn("  ", "   ")  # blank turn is dropped
    msgs = c.messages()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "hello"
    assert msgs[3]["content"] == "you said hello"


def test_persistence_roundtrip(tmp_path):
    p = str(tmp_path / "conv.json")
    c = Conversation(data_path=p)
    c.add_turn("a", "b")
    c.add_turn("c", "d")
    c2 = Conversation(data_path=p)
    assert c2.turns == [("a", "b"), ("c", "d")]
    vault_doc = c2._vault.read("sessions", "conv.md")
    assert vault_doc is not None
    turns_raw = vault_doc.frontmatter.get("turns")
    if isinstance(turns_raw, str):
        turns_list = json.loads(turns_raw)
    else:
        turns_list = turns_raw
    assert turns_list == [["a", "b"], ["c", "d"]]


def test_corrupt_file_starts_fresh(tmp_path):
    p = tmp_path / "conv.json"
    p.write_text("{not json", encoding="utf-8")
    c = Conversation(data_path=str(p))
    assert c.turns == [] and c.summary is None
    c.add_turn("x", "y")  # still usable


def test_needs_compaction_thresholds():
    c = Conversation(compact_after_chars=100, keep_recent_turns=2)
    for i in range(2):
        c.add_turn(f"u{i}", "y" * 50)  # 104 chars but only 2 turns
    assert c.size_chars() > 100
    assert not c.needs_compaction()  # turns <= keep_recent
    c.add_turn("u2", "y" * 50)
    assert c.needs_compaction()


def test_compaction():
    calls = []

    def summarize(messages):
        calls.append(messages)
        return "user greeted; asked the time"

    c = Conversation(compact_after_chars=10, keep_recent_turns=2)
    for i in range(5):
        c.add_turn(f"u{i} " + "x" * 10, f"a{i} " + "y" * 10)
    assert c.compact(summarize) is True
    assert c.summary == "user greeted; asked the time"
    assert [u[:2] for u, _ in c.turns] == ["u3", "u4"]
    # no "Previous summary" on the first pass
    assert not any("Previous summary:" in m["content"] for m in calls[0])

    # second round folds the previous summary in
    for i in range(5, 8):
        c.add_turn(f"u{i} " + "x" * 10, f"a{i} " + "y" * 10)
    assert c.compact(summarize) is True
    assert calls[1][0]["content"].startswith("Previous summary: user greeted")

    # messages() carries the summary first
    msgs = c.messages()
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("Earlier in this conversation")


def test_compact_skipped_when_history_grows():
    c = Conversation(compact_after_chars=10, keep_recent_turns=2)
    for i in range(5):
        c.add_turn(f"u{i}" + "x" * 10, f"a{i}" + "y" * 10)

    def slow_summarize(messages):
        c.add_turn("late", "user")  # history grows during the LLM call
        return "S"

    assert c.compact(slow_summarize) is False
    assert len(c.turns) == 6
    assert c.summary is None
    # not stuck: a later pass works
    assert c.compact(lambda m: "S") is True


def test_compact_noop_when_small():
    c = Conversation(compact_after_chars=10**6, keep_recent_turns=2)
    c.add_turn("u", "a")
    assert c.compact(lambda m: "S") is False


def test_maybe_compact_runs_in_background():
    c = Conversation(compact_after_chars=10, keep_recent_turns=2)
    for i in range(5):
        c.add_turn(f"u{i}" + "x" * 10, f"a{i}" + "y" * 10)
    c.maybe_compact(lambda m: "bg summary")
    deadline = time.time() + 5
    while c.summary is None and time.time() < deadline:
        time.sleep(0.05)
    assert c.summary == "bg summary"
    assert len(c.turns) == 2


# -- token threshold, callback and guardrails (T026/T027) ---------------------


def test_token_threshold_triggers_compaction():
    c = Conversation(compact_after_chars=10**6, compact_after_tokens=20)
    for i in range(10):
        c.add_turn(f"user {i} " + "word " * 10, f"assistant {i} " + "word " * 10)
    assert c.size_chars() < 10**6
    assert c.size_tokens() > 20
    assert c.needs_compaction()
    assert c.compact(lambda m: "S") is True
    assert c.summary == "S"


def test_token_threshold_disabled_by_default():
    c = Conversation(compact_after_chars=10**6)
    for i in range(10):
        c.add_turn(f"user {i} " + "word " * 10, f"assistant {i} " + "word " * 10)
    assert not c.needs_compaction()


def test_compact_reports_through_on_compact():
    events = []
    c = Conversation(
        compact_after_chars=10, keep_recent_turns=2,
        on_compact=lambda s, b, a: events.append((s, b, a)),
    )
    for i in range(5):
        c.add_turn(f"u{i}" + "x" * 10, f"a{i}" + "y" * 10)
    assert c.compact(lambda m: "S") is True
    assert events == [("S", 5, 2)]


def test_summary_is_hard_capped():
    c = Conversation(compact_after_chars=10, keep_recent_turns=2)
    for i in range(3):
        c.add_turn(f"u{i}" + "x" * 10, f"a{i}" + "y" * 10)
    c.compact(lambda m: "s" * 2000)
    assert c.summary == "s" * 500


def test_checkpoint_resets_periodically(monkeypatch):
    import app.conversation as conv

    monkeypatch.setattr(conv, "RESET_EVERY", 2)
    folded = []

    def summarize(messages):
        folded.append(any("Previous summary:" in m["content"] for m in messages))
        return f"S{len(folded)}"

    c = Conversation(compact_after_chars=10, keep_recent_turns=1)
    for _ in range(3):
        c.add_turn("u" + "x" * 10, "y" * 10)
    c.compact(summarize)
    c.add_turn("u" + "x" * 10, "y" * 10)
    c.add_turn("u" + "x" * 10, "y" * 10)
    c.compact(summarize)  # 2nd: the reset pass, no folding
    c.add_turn("u" + "x" * 10, "y" * 10)
    c.add_turn("u" + "x" * 10, "y" * 10)
    c.compact(summarize)  # 3rd: folds the previous summary in
    assert folded == [False, False, True]


def test_session_store_forwards_compaction(tmp_path):
    from app.conversation import SessionStore

    events = []
    store = SessionStore(
        data_dir=str(tmp_path),
        compact_after_chars=10,
        keep_recent_turns=2,
        on_compact=lambda sid, s, b, a: events.append((sid, s, b, a)),
    )
    sid = store.active_id
    conv = store.conversation_for(sid)
    for i in range(5):
        conv.add_turn(f"u{i}" + "x" * 10, f"a{i}" + "y" * 10)
    assert conv.compact(lambda m: "S") is True
    assert events == [(sid, "S", 5, 2)]


def test_set_limits_applies_token_threshold_to_loaded_sessions(tmp_path):
    from app.conversation import SessionStore

    store = SessionStore(
        data_dir=str(tmp_path), compact_after_chars=10**6, keep_recent_turns=2
    )
    conv = store.conversation_for(store.active_id)
    for i in range(10):
        conv.add_turn(f"user {i} " + "word " * 10, f"assistant {i} " + "word " * 10)
    assert not conv.needs_compaction()
    store.set_limits(10**6, 2, compact_after_tokens=20)
    assert conv.needs_compaction()


# -- finalised tool-task state + 'continue' resolution -------------------------


def test_tool_state_roundtrip():
    c = Conversation()
    assert c.tool_state is None
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    c.set_tool_state(msgs, 1)
    assert c.tool_state == {"messages": list(msgs), "continuations": 1}
    c.clear_tool_state()
    assert c.tool_state is None


def test_tool_state_not_persisted(tmp_path):
    p = str(tmp_path / "conv.json")
    c = Conversation(data_path=p)
    c.set_tool_state([{"role": "user", "content": "u"}], 2)
    assert Conversation(data_path=p).tool_state is None


def test_is_continue_variants():
    from app.conversation import is_continue

    for text in (
        "continue", "Continue", " CONTINUE ", "continue.", "Continue?", "continue!"
    ):
        assert is_continue(text), text
    for text in ("", "continue please", "keep going", "continue the song", "continues"):
        assert not is_continue(text), text


def test_resolve_continue_within_budget(monkeypatch):
    from app import config
    from app.conversation import resolve_continue

    monkeypatch.setattr(config, "MAX_CONTINUATIONS", 2)
    c = Conversation()
    msgs = [{"role": "user", "content": "task"}]
    c.set_tool_state(msgs, 0)
    resume, refusal = resolve_continue(c, "continue")
    assert resume == msgs and refusal is None
    assert c.tool_state["continuations"] == 1
    resume, refusal = resolve_continue(c, "Continue.")
    assert resume == msgs and refusal is None
    assert c.tool_state["continuations"] == 2
    # budget exhausted: fixed refusal, state cleared
    resume, refusal = resolve_continue(c, "continue")
    assert resume is None and refusal is not None
    assert c.tool_state is None


def test_resolve_continue_without_pending_state():
    from app.conversation import resolve_continue

    c = Conversation()
    assert resolve_continue(c, "continue") == (None, None)
    assert c.tool_state is None


def test_resolve_continue_new_utterance_clears_stale_state():
    from app.conversation import resolve_continue

    c = Conversation()
    c.set_tool_state([{"role": "user", "content": "task"}], 0)
    assert resolve_continue(c, "what's the weather?") == (None, None)
    assert c.tool_state is None
