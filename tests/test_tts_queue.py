"""T014: bounded TTS queue, ordering, cooperative barge-in (offline).

Exercises the real VoiceSession / _handle_utterance / _tts_worker machinery
with fake engines and an in-process fake WebSocket: sentences must be
synthesized and sent strictly in enqueue order, the producer may run at most
TTS_QUEUE_SIZE ahead of the consumer, and a barge-in must drop all stale work
so no stale audio can leak into the next generation.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np

from app import config
from app.pipeline import VoiceSession


class FakeWS:
    """Records every frame the pipeline sends (thread-safe)."""

    def __init__(self) -> None:
        self._frames: list[tuple[str, object]] = []
        self._lock = threading.Lock()

    async def send_bytes(self, b: bytes) -> None:
        with self._lock:
            self._frames.append(("audio", len(b)))

    async def send_text(self, t: str) -> None:
        with self._lock:
            self._frames.append(("text", json.loads(t)))

    def audio_frames(self) -> int:
        with self._lock:
            return sum(1 for k, _ in self._frames if k == "audio")

    def texts(self, kind: str) -> list[dict]:
        with self._lock:
            return [d for k, d in self._frames if k == "text" and d.get("type") == kind]


class FakeTTS:
    def __init__(self, delay: float = 0.02, secs: float = 0.05) -> None:
        self.delay = delay
        self.n = int(secs * 24000)
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            self.calls.append(text)
        time.sleep(self.delay)
        return np.zeros(self.n, dtype=np.float32)


class FakeAgent:
    def __init__(self, deltas: list[str], token_delay: float = 0.0) -> None:
        self.deltas = deltas
        self.token_delay = token_delay

    def reply(self, text, execute, history=None):
        for d in self.deltas:
            if self.token_delay:
                time.sleep(self.token_delay)
            yield d

    def summarize(self, messages) -> str:
        return ""

    def close(self) -> None:
        pass


def make_engines(agent: FakeAgent, tts: FakeTTS):
    return SimpleNamespace(
        stt=SimpleNamespace(transcribe=lambda samples: "hello there"),
        tts=tts,
        agent=agent,
        conversation=SimpleNamespace(
            messages=lambda: [],
            add_turn=lambda user, answer: None,
            maybe_compact=lambda summarize: None,
        ),
    )


def start(session: VoiceSession) -> None:
    assert session.start_utterance(np.zeros(320, dtype=np.float32))


async def wait_event(ws: FakeWS, kind: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ws.texts(kind):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {kind!r}")


async def wait_audio(ws: FakeWS, n: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ws.audio_frames() >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {n} audio frames (got {ws.audio_frames()})")


def test_sentences_synthesized_and_sent_in_order():
    deltas = ["One. ", "Two. ", "Three. ", "Four"]
    tts = FakeTTS(delay=0.02)
    ws = FakeWS()

    async def go() -> None:
        session = VoiceSession(ws, make_engines(FakeAgent(deltas), tts))
        start(session)
        await wait_event(ws, "reply_done")

    asyncio.run(go())
    assert tts.calls == ["One.", "Two.", "Three.", "Four"]
    assert ws.audio_frames() == 4
    assert [e["text"] for e in ws.texts("transcript")] == ["hello there"]
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == "One. Two. Three. Four"
    assert len(ws.texts("reply_done")) == 1


def test_producer_overlaps_consumer():
    """The LLM keeps generating while TTS synthesizes (no serial wait)."""
    deltas = [f"S{i}. " for i in range(1, 7)]
    tts = FakeTTS(delay=0.08)
    agent = FakeAgent(deltas, token_delay=0.05)
    ws = FakeWS()

    async def go() -> float:
        session = VoiceSession(ws, make_engines(agent, tts))
        t0 = time.monotonic()
        start(session)
        await wait_event(ws, "reply_done", timeout=30)
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    serial = 6 * 0.05 + 6 * 0.08  # if LLM and TTS ran strictly one after the other
    assert elapsed < serial * 0.9, f"no overlap: elapsed={elapsed:.3f}s serial={serial:.3f}s"
    assert tts.calls == [d.strip() for d in deltas]
    assert ws.audio_frames() == 6


def test_queue_never_exceeds_bound():
    deltas = [f"S{i}. " for i in range(1, 21)]
    tts = FakeTTS(delay=0.03)
    ws = FakeWS()
    max_seen = 0

    async def go() -> None:
        nonlocal max_seen
        session = VoiceSession(ws, make_engines(FakeAgent(deltas), tts))
        start(session)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if session.tts_queue is not None:
                max_seen = max(max_seen, session.tts_queue.qsize())
            if ws.texts("reply_done"):
                return
            await asyncio.sleep(0.001)
        raise AssertionError("timed out waiting for reply_done")

    asyncio.run(go())
    assert max_seen <= config.TTS_QUEUE_SIZE, f"queue exceeded bound: {max_seen}"
    assert tts.calls == [d.strip() for d in deltas]
    assert ws.audio_frames() == 20


def test_barge_in_drops_stale_audio_and_recovers():
    deltas = [f"S{i}. " for i in range(1, 11)]
    tts = FakeTTS(delay=0.06)
    ws = FakeWS()

    async def go() -> int:
        session = VoiceSession(ws, make_engines(FakeAgent(deltas), tts))
        start(session)
        await wait_audio(ws, 1)
        before = ws.audio_frames()
        session.release()
        await asyncio.sleep(0.5)  # the old generation must fully wind down
        leaked = ws.audio_frames() - before
        assert leaked == 0, f"{leaked} stale audio frames leaked after barge-in"
        # recovery: a fresh utterance still works end to end
        start(session)
        await wait_event(ws, "reply_done", timeout=30)
        return ws.audio_frames() - before

    fresh = asyncio.run(go())
    assert fresh == 10, f"expected 10 fresh frames, got {fresh}"
    # the cancelled reply never sent reply_done; only the fresh one did
    assert len(ws.texts("reply_done")) == 1


def test_barge_after_generation_complete_does_not_deadlock():
    """Barge-in after the producer finished (sentinel already queued) must not
    deadlock the pair: release() drains the queue and must preserve the
    _TTS_DONE sentinel the worker needs to terminate (the producer is parked
    in worker.join() until it does)."""
    deltas = ["S1. ", "S2"]
    tts = FakeTTS(delay=0.3)
    ws = FakeWS()

    async def go() -> None:
        session = VoiceSession(ws, make_engines(FakeAgent(deltas), tts))
        n0 = len(threading.enumerate())
        start(session)
        await wait_audio(ws, 1)
        await asyncio.sleep(0.1)  # producer is now parked in worker.join()
        assert len(threading.enumerate()) >= n0 + 2
        session.release()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and len(threading.enumerate()) > n0:
            await asyncio.sleep(0.05)
        assert len(threading.enumerate()) == n0, \
            "producer/worker threads leaked (worker lost its sentinel)"
        assert ws.audio_frames() == 1  # s2 was synthesized but never sent
        assert len(ws.texts("reply_done")) == 0  # cancelled: no reply_done

    asyncio.run(go())
