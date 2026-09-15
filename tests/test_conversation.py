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
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["turns"] == [["a", "b"], ["c", "d"]]


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
