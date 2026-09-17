"""T020: clause-level chunking for early TTS start (offline).

Sentence endings always win. A still-open sentence is split at clause
punctuation (comma/semicolon/colon) once the buffer exceeds clause_max_chars,
so TTS can start on the first clause before the full sentence is generated.
The max_chars hard split stays the backstop for unpunctuated text, and
clause_max_chars=0 (the default) restores the classic behaviour.
"""

from __future__ import annotations

import asyncio
import time

from app import config
from app.pipeline import SentenceChunker, VoiceSession
from test_tts_queue import FakeAgent, FakeTTS, FakeWS, make_engines, start, wait_event


def run_chunker(text: str, max_chars: int = 90, clause: int = 40, step: int = 1) -> list[str]:
    c = SentenceChunker(max_chars=max_chars, clause_max_chars=clause)
    out: list[str] = []
    for i in range(0, len(text), step):
        out.extend(c.add(text[i : i + step]))
    out.extend(c.flush())
    return out


# ---------------- SentenceChunker (unit) ----------------

def test_sentence_endings_still_win():
    assert run_chunker("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_clause_emitted_before_sentence_complete():
    c = SentenceChunker(max_chars=90, clause_max_chars=40)
    out = c.add("The weather in Reykjavik is mild today, with light winds expected.")
    # the sentence is not finished (no ". " yet) but the first clause went out
    assert out == ["The weather in Reykjavik is mild today,"]
    out.extend(c.flush())
    assert out == [
        "The weather in Reykjavik is mild today,",
        "with light winds expected.",
    ]


def test_no_split_below_threshold():
    assert run_chunker("Short, and sweet.") == ["Short, and sweet."]


def test_multiple_clauses_split_in_order():
    text = "First part of the answer, second part of the answer, third part of the answer."
    assert run_chunker(text, clause=30) == [
        "First part of the answer,",
        "second part of the answer,",
        "third part of the answer.",
    ]


def test_semicolon_is_a_boundary():
    text = "Alpha beta gamma delta epsilon; zeta eta theta iota kappa."
    assert run_chunker(text, clause=30) == [
        "Alpha beta gamma delta epsilon;",
        "zeta eta theta iota kappa.",
    ]


def test_colon_is_a_boundary():
    text = "This is a long leading part before the colon: then the rest of it follows."
    assert run_chunker(text, clause=40) == [
        "This is a long leading part before the colon:",
        "then the rest of it follows.",
    ]


def test_tiny_boundary_waits_for_more_text():
    c = SentenceChunker(max_chars=90, clause_max_chars=40)
    assert c.add("Yes, a much longer continuation follows here today for sure") == []
    assert c.add(" and more.") == []
    assert c.flush() == ["Yes, a much longer continuation follows here today for sure and more."]


def test_hard_split_backstop_unpunctuated():
    out = run_chunker("x" * 120)
    assert "".join(out) == "x" * 120
    assert [len(s) for s in out] == [90, 30]


def test_disabled_when_zero():
    text = "A fairly long first clause here, with a second part coming after it"
    c = SentenceChunker(max_chars=90, clause_max_chars=0)
    assert c.add(text) == []
    assert c.flush() == [text]


def test_default_is_legacy_behaviour():
    text = "A fairly long first clause here, with a second part coming after it"
    c = SentenceChunker(max_chars=90)
    assert c.add(text) == []
    assert c.flush() == [text]


# ---------------- pipeline level ----------------

class TimingTTS(FakeTTS):
    def __init__(self, delay: float = 0.02) -> None:
        super().__init__(delay=delay)
        self.first_call_at: float | None = None
        self.t0 = 0.0

    def synthesize(self, text: str):
        if self.first_call_at is None:
            self.first_call_at = time.monotonic() - self.t0
        return super().synthesize(text)


def test_first_clause_starts_tts_before_sentence_complete(monkeypatch):
    """The first TTS synthesis is the sentence's first clause, enqueued while
    the LLM is still generating the rest (T020)."""
    monkeypatch.setattr(config, "SENTENCE_MAX_CHARS", 90)
    monkeypatch.setattr(config, "CLAUSE_MAX_CHARS", 40)
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 30.0)  # no filler noise
    text = "The weather in Reykjavik is mild today, with light winds expected all week long."
    tts = TimingTTS(delay=0.02)
    agent = FakeAgent(list(text), token_delay=0.01)
    ws = FakeWS()

    async def go() -> None:
        session = VoiceSession(ws, make_engines(agent, tts))
        tts.t0 = time.monotonic()
        start(session)
        await wait_event(ws, "reply_done")

    asyncio.run(go())
    assert tts.calls[0] == "The weather in Reykjavik is mild today,"
    assert len(tts.calls) == 2
    assert tts.first_call_at is not None
    producer_end = len(text) * 0.01
    assert tts.first_call_at < producer_end * 0.75, (
        f"first synthesis at {tts.first_call_at:.2f}s, producer finished at {producer_end:.2f}s"
    )
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == text
