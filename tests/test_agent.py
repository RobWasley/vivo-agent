from app import config, tools
from app.agent import Agent, ToolRound


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
