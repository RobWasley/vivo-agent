"""T015: spoken filler phrases while the model is thinking (offline).

Exercises the real VoiceSession / _handle_utterance / ThinkingFiller /
_tts_worker machinery with fake engines and an in-process fake WebSocket:
while no speakable text has arrived for THINK_FILLER_FIRST_AFTER seconds
(prefill, reasoning, or a tool round) the pipeline must speak a short filler
phrase, repeat every THINK_FILLER_INTERVAL seconds while the thinking goes
on, never speak or store the reasoning itself, and stop the filler at once
on barge-in.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from app import config
from app.agent import ReasoningDelta
from app.conversation import SessionStore
from app.pipeline import Bench, FILLER_PHRASES, ThinkingFiller, VoiceSession
from app.wake import WakeState


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
    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            self.calls.append(text)
        time.sleep(self.delay)
        return np.zeros(int(0.05 * 48000), dtype=np.float32)


class ThinkingAgent:
    """Fake agent: streams reasoning deltas for think_secs, then the answer."""

    def __init__(self, think_secs: float, answer: list[str], tick: float = 0.02) -> None:
        self.think_secs = think_secs
        self.answer = answer
        self.tick = tick

    def reply(self, text, execute, history=None):
        t_end = time.monotonic() + self.think_secs
        while time.monotonic() < t_end:
            time.sleep(self.tick)
            yield ReasoningDelta("r")
        for d in self.answer:
            yield d

    def summarize(self, messages) -> str:
        return ""

    def close(self) -> None:
        pass


def make_engines(agent: ThinkingAgent, tts: FakeTTS):
    sessions = SessionStore()  # in-memory: no data_dir
    conv = sessions.conversation_for(sessions.active_id)
    return (
        SimpleNamespace(
            stt=SimpleNamespace(transcribe=lambda samples: "hello there"),
            tts=tts,
            agent=agent,
            memory=SimpleNamespace(core_summary=lambda: "No important memory yet."),
            skills=SimpleNamespace(index=lambda: ""),
            sessions=sessions,
            wake=WakeState(),  # disabled (no phrase): legacy always-answer behavior
        ),
        conv,
    )


def start(session: VoiceSession) -> None:
    assert session.start_utterance(np.zeros(320, dtype=np.float32))


def filler_calls(tts: FakeTTS) -> list[str]:
    return [c for c in tts.calls if c in FILLER_PHRASES]


def answer_calls(tts: FakeTTS) -> list[str]:
    return [c for c in tts.calls if c not in FILLER_PHRASES]


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
    raise AssertionError(
        f"timed out waiting for {n} audio frames (got {ws.audio_frames()})"
    )


def _drain(q: "queue.Queue") -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_no_filler_when_answer_starts_quickly(monkeypatch):
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 0.5)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 5.0)
    tts = FakeTTS()
    agent = ThinkingAgent(think_secs=0.1, answer=["Quick answer. "])
    ws = FakeWS()
    engines, conv = make_engines(agent, tts)

    async def go() -> None:
        session = VoiceSession(ws, engines)
        start(session)
        await wait_event(ws, "reply_done")

    asyncio.run(go())
    assert filler_calls(tts) == []
    assert answer_calls(tts) == ["Quick answer."]
    assert ws.audio_frames() == 1
    # reasoning is never spoken, never sent, never stored; the whole answer
    # is one message (a single start frame, no filler)
    frames = ws.texts("agent_text")
    assert [e["delta"] for e in frames] == ["Quick answer. "]
    assert frames[0]["start"] is True
    assert not any(e.get("filler") for e in frames)
    assert conv.turns == [("hello there", "Quick answer.")]


def test_filler_spoken_while_model_thinks(monkeypatch):
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 0.3)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 5.0)
    tts = FakeTTS()
    agent = ThinkingAgent(think_secs=0.8, answer=["The answer is forty. "])
    ws = FakeWS()
    engines, conv = make_engines(agent, tts)

    async def go() -> None:
        session = VoiceSession(ws, engines)
        start(session)
        await wait_event(ws, "reply_done")

    asyncio.run(go())
    # exactly one filler: due at 0.3 s, the next only at ~5 s
    assert len(filler_calls(tts)) == 1
    assert filler_calls(tts)[0] in FILLER_PHRASES
    # the filler is synthesized before the first answer sentence
    assert tts.calls[0] in FILLER_PHRASES
    assert answer_calls(tts) == ["The answer is forty."]
    assert ws.audio_frames() == 2
    # reasoning never reaches the client or the conversation memory; the
    # filler and the answer are separate messages (their own start frames)
    frames = ws.texts("agent_text")
    filler_frames = [e for e in frames if e.get("filler")]
    answer_frames = [e for e in frames if not e.get("filler")]
    assert len(filler_frames) == 1
    assert filler_frames[0]["delta"] in FILLER_PHRASES
    assert filler_frames[0]["start"] is True
    assert answer_frames[0]["start"] is True
    assert "".join(e["delta"] for e in answer_frames) == "The answer is forty. "
    assert conv.turns == [("hello there", "The answer is forty.")]


def test_filler_repeats_while_thinking_goes_on(monkeypatch):
    """A long reasoning stretch keeps getting fresh filler audio (T015)."""
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 0.3)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 0.6)
    tts = FakeTTS()
    agent = ThinkingAgent(think_secs=2.5, answer=["Finally, done. "])
    ws = FakeWS()
    engines, _ = make_engines(agent, tts)

    async def go() -> None:
        session = VoiceSession(ws, engines)
        start(session)
        await wait_event(ws, "reply_done", timeout=30)

    asyncio.run(go())
    n = len(filler_calls(tts))
    assert n >= 2, f"expected repeated fillers over 2.5 s of thinking, got {n}"
    assert answer_calls(tts) == ["Finally, done."]
    assert ws.audio_frames() == n + 1


def test_filler_stops_on_barge_in(monkeypatch):
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 0.3)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 0.3)
    tts = FakeTTS()
    agent = ThinkingAgent(think_secs=3.0, answer=["never heard. "])
    ws = FakeWS()
    engines, _ = make_engines(agent, tts)

    async def go() -> int:
        session = VoiceSession(ws, engines)
        n0 = len(threading.enumerate())
        start(session)
        await wait_audio(ws, 1)  # the first filler
        before = ws.audio_frames()
        session.release()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and len(threading.enumerate()) > n0:
            await asyncio.sleep(0.05)
        assert len(threading.enumerate()) == n0, \
            "filler/producer/worker threads leaked after barge-in"
        return ws.audio_frames() - before

    leaked = asyncio.run(go())
    assert leaked == 0, f"{leaked} stale filler frames leaked after barge-in"
    # only fillers were ever synthesized; the answer never arrived
    assert len(filler_calls(tts)) >= 1
    assert answer_calls(tts) == []
    assert len(ws.texts("reply_done")) == 0  # cancelled: no reply_done


def test_scheduler_suppressed_by_steady_text(monkeypatch):
    """Continuous speakable deltas keep the silence below FIRST_AFTER."""
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 0.25)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 5.0)
    session = SimpleNamespace(alive=lambda gen: True, send=lambda payload: None)
    q: "queue.Queue" = queue.Queue(maxsize=8)
    f = ThinkingFiller(session, 1, q, Bench(time.monotonic()))
    f.start()
    end = time.monotonic() + 0.9
    while time.monotonic() < end:
        f.note_text()
        time.sleep(0.05)
    f.stop()
    assert _drain(q) == []


def test_scheduler_repeats_during_pure_silence(monkeypatch):
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 0.2)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 0.4)
    session = SimpleNamespace(alive=lambda gen: True, send=lambda payload: None)
    q: "queue.Queue" = queue.Queue(maxsize=8)
    f = ThinkingFiller(session, 1, q, Bench(time.monotonic()))
    f.start()
    time.sleep(1.2)
    f.stop()
    items = _drain(q)
    assert len(items) >= 2
    assert all(item[2] == "filler" and item[1] in FILLER_PHRASES for item in items)


def test_filler_skips_when_queue_is_full(monkeypatch):
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 0.05)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 0.1)
    session = SimpleNamespace(alive=lambda gen: True, send=lambda payload: None)
    q: "queue.Queue" = queue.Queue(maxsize=1)
    q.put((0, "busy", "tts"))
    f = ThinkingFiller(session, 1, q, Bench(time.monotonic()))
    f.start()
    time.sleep(0.25)
    f.stop()
    assert f.count == 0
    assert q.qsize() == 1
    assert _drain(q)[0][1] == "busy"


def test_wake_ack_and_reply_are_separate_messages(monkeypatch):
    """The wake ack and the LLM reply stream as separate messages: each
    begins with a start frame (a new transcript line + caption reset)."""
    monkeypatch.setattr(config, "THINK_FILLER_FIRST_AFTER", 5.0)
    monkeypatch.setattr(config, "THINK_FILLER_INTERVAL", 5.0)
    tts = FakeTTS()
    agent = ThinkingAgent(think_secs=0.05, answer=["Here is the answer. "])
    sessions = SessionStore()
    conv = sessions.conversation_for(sessions.active_id)
    engines = SimpleNamespace(
        stt=SimpleNamespace(transcribe=lambda samples: "hey vivo what time is it"),
        tts=tts,
        agent=agent,
        memory=SimpleNamespace(core_summary=lambda: "No important memory yet."),
        skills=SimpleNamespace(index=lambda: ""),
        sessions=sessions,
        wake=WakeState("hey vivo", ("goodbye",), 30.0),
    )
    ws = FakeWS()

    async def go() -> None:
        session = VoiceSession(ws, engines)
        start(session)
        await wait_event(ws, "reply_done")

    asyncio.run(go())
    frames = ws.texts("agent_text")
    assert [e["delta"] for e in frames] == [config.WAKE_ACK, "Here is the answer. "]
    assert all(e["start"] is True for e in frames)
    assert not any(e.get("filler") for e in frames)
    # conversation memory keeps the combined answer as one turn
    assert conv.turns == [("hey vivo what time is it", config.WAKE_ACK + "Here is the answer.")]
    assert engines.wake.active
