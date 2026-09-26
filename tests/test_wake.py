"""T024: wake-phrase session gating (offline).

WakeState unit tests (pattern matching, state transitions, fake clock,
hot-apply) plus the pipeline gate: asleep utterances are ignored silently,
a wake phrase-only utterance spawns the spoken ack without an LLM round
trip, the text after the phrase is processed normally, an end phrase ends
the session, the idle timeout speaks the goodnight (never while a reply is
in progress; suppressed when the audio has a long gap, e.g. the mic was off
or the tab closed), a manual wake pressed mid-STT never processes the
pre-wake utterance, a manual wake pressed mid-utterance drops the
in-progress pre-wake speech (only audio after the button is processed),
dropped utterances leave no INFO stt log line, and an empty phrase keeps
the legacy always-answer behavior.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np

from app import config, pipeline
from app.conversation import SessionStore
from app.pipeline import VoiceSession
from app.wake import WakeState
from test_tts_queue import FakeAgent, FakeTTS, FakeWS, start, wait_event
from test_vad import SPEECH, silence


# ---------------- unit: pattern matching ----------------

def test_pattern_matches_punctuation_and_case():
    w = WakeState("hey vivo")
    for text in ["Hey, vivo!", "HEY VIVO", "yo hey vivo bro", "say: hey vivo. now"]:
        assert w.match_wake(text), text


def test_pattern_requires_word_boundaries():
    w = WakeState("hey vivo")
    for text in ["hello vivo", "hey vivov", "hey viv", "vivo hey"]:
        assert w.match_wake(text) is None, text


def test_empty_phrase_disables_feature():
    w = WakeState("")
    assert not w.enabled
    assert w.match_wake("hey vivo") is None


def test_end_phrase_matching():
    w = WakeState("hey vivo", ("that's all", "goodbye"))
    assert w.match_end("okay, that's all for today") == "that's all"
    assert w.match_end("Goodbye!") == "goodbye"
    assert w.match_end("never mind") is None


def test_update_hot_applies_phrase_and_end_phrases():
    w = WakeState("hey vivo")
    assert w.match_wake("hey vivo")
    w.update("hi vivo", ("done",), 60.0)
    assert w.match_wake("hey vivo") is None
    assert w.match_wake("hi vivo")
    assert w.match_end("all done") == "done"
    w.update("", (), 30.0)
    assert not w.enabled


# ---------------- unit: state transitions (fake clock) ----------------

def test_wake_and_sleep_transitions():
    t = [0.0]
    w = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])
    assert not w.active
    assert w.wake() is True
    assert w.wake() is False  # idempotent
    assert w.sleep() is True
    assert not w.sleep()


def test_claim_sleep_respects_timeout():
    t = [0.0]
    w = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])
    w.wake()
    t[0] = 29.0
    assert not w.claim_sleep()
    t[0] = 31.0
    assert w.claim_sleep()
    assert not w.active
    assert not w.claim_sleep()  # already asleep


def test_touch_resets_idle_clock():
    t = [0.0]
    w = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])
    w.wake()
    t[0] = 20.0
    w.touch()
    t[0] = 49.0  # 29s since the touch, 49s since the wake
    assert not w.claim_sleep()
    t[0] = 51.0
    assert w.claim_sleep()


# ---------------- pipeline: the gate ----------------

class RecordingAgent(FakeAgent):
    def __init__(self, deltas: list[str]) -> None:
        super().__init__(deltas)
        self.prompts: list[str] = []

    def reply(self, text, execute, history=None, resume_messages=None):
        self.prompts.append(text)
        return super().reply(text, execute, history)


def wake_engines(text: str, agent, tts: FakeTTS, wake: WakeState):
    return SimpleNamespace(
        stt=SimpleNamespace(transcribe=lambda samples: text),
        tts=tts,
        agent=agent,
        memory=SimpleNamespace(core_summary=lambda: "No important memory yet."),
        skills=SimpleNamespace(index=lambda: ""),
        sessions=SessionStore(),  # in-memory: no data_dir
        wake=wake,
    )


def test_asleep_ignores_utterance_silently(caplog):
    tts = FakeTTS()
    ws = FakeWS()
    eng = wake_engines("hello there", FakeAgent(["Ok."]), tts, WakeState("hey vivo"))

    async def go() -> None:
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)  # observe broadcasts, as serve_session would
        try:
            start(session)
            await asyncio.sleep(0.3)  # the handler thread is done long after this
        finally:
            pipeline._ACTIVE.discard(session)

    with caplog.at_level(logging.DEBUG, logger="vivo.pipeline"):
        asyncio.run(go())
    # dropped: no transcript, no agent text, nothing synthesized, state unchanged
    # (a bare reply_done may still arrive — same as the empty-transcript path)
    assert ws.texts("transcript") == []
    assert ws.texts("agent_text") == []
    assert tts.calls == []
    assert ws.audio_frames() == 0
    assert eng.wake.active is False
    # and the transcript never reaches the system log (no INFO stt line)
    assert not any(
        r.name == "vivo.pipeline" and r.getMessage().startswith("stt ")
        for r in caplog.records
    )


def test_manual_wake_during_stt_drops_pre_wake_utterance():
    """A wake button pressed while STT is still running must not pull the
    pre-wake utterance into the session: the gate reads the state at capture
    time, so the phrase-less utterance is dropped as usual."""
    tts = FakeTTS()
    ws = FakeWS()
    agent = RecordingAgent(["Nope."])
    wake = WakeState("hey vivo")
    entered = threading.Event()
    release = threading.Event()

    class SlowSTT:
        def transcribe(self, samples):
            entered.set()
            release.wait(timeout=5)
            return "turn on the kitchen light"

    eng = SimpleNamespace(
        stt=SlowSTT(),
        tts=tts,
        agent=agent,
        memory=SimpleNamespace(core_summary=lambda: "No important memory yet."),
        skills=SimpleNamespace(index=lambda: ""),
        sessions=SessionStore(),
        wake=wake,
    )

    async def go() -> None:
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            start(session)
            assert entered.wait(timeout=5)
            assert wake.wake() is True  # the user hits the wake button
            release.set()
            await wait_event(ws, "reply_done")  # the dropped path still ends
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert wake.active is True  # the manual wake still took effect
    assert agent.prompts == []
    assert ws.texts("transcript") == []
    assert ws.texts("agent_text") == []
    assert tts.calls == []


def test_manual_wake_drops_in_progress_utterance():
    """The real mic path: a wake button pressed *mid-utterance* must not
    process the speech that predates it. The in-progress VAD utterance is
    dropped (no 'end' event), and only audio after the button becomes the
    new utterance."""
    tts = FakeTTS()
    ws = FakeWS()
    agent = RecordingAgent(["Nope."])
    wake = WakeState("hey vivo")
    lengths: list[int] = []

    class SizingSTT:
        def transcribe(self, samples: np.ndarray) -> str:
            lengths.append(len(samples))
            return "what is 2 plus 2"

    SR = 16000

    def feed(session: VoiceSession, sig: np.ndarray) -> None:
        pcm16 = (np.clip(sig, -1.0, 1.0) * 32767).astype(np.int16)
        for i in range(0, len(pcm16), 1024):
            pipeline._on_audio(session, pcm16[i : i + 1024].tobytes())

    eng = SimpleNamespace(
        stt=SizingSTT(),
        tts=tts,
        agent=agent,
        memory=SimpleNamespace(core_summary=lambda: "No important memory yet."),
        skills=SimpleNamespace(index=lambda: ""),
        sessions=SessionStore(),
        wake=wake,
    )

    async def go() -> None:
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            # mid-utterance: the user is talking while vivo is asleep, and
            # no endpoint has fired yet
            feed(session, SPEECH[:SR])
            assert session.vad._in_speech
            # ...and hits the wake button
            assert pipeline.manual_wake(eng) is True
            # the rest of the sentence, then silence past the endpoint
            # (min_silence + reopen come from vivo.toml)
            endpoint_s = (config.VAD_MIN_SILENCE_MS + config.VAD_REOPEN_MS + 500) / 1000.0
            feed(session, np.concatenate([SPEECH[SR:], silence(endpoint_s)]))
            await wait_event(ws, "reply_done")
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert wake.active is True
    assert [e["active"] for e in ws.texts("wake")] == [True]
    # STT saw exactly one utterance — the post-wake part only (~1.6-2.2 s),
    # not the full ~2.8 s fixture
    assert len(lengths) == 1
    assert 1.4 * SR < lengths[0] < 2.4 * SR, f"STT length {lengths[0] / SR:.2f}s"
    # and it was processed: awake, so straight to the agent
    assert agent.prompts == ["what is 2 plus 2"]
    assert [e["text"] for e in ws.texts("transcript")] == ["what is 2 plus 2"]


def test_on_audio_drops_stale_session_silently_after_gap():
    """The real mic-on path: a long audio gap before this frame means the
    session went quiet unwatched, so the stale goodnight is suppressed."""
    tts = FakeTTS()
    ws = FakeWS()
    t = [0.0]
    wake = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])

    async def go() -> None:
        eng = wake_engines("irrelevant", RecordingAgent(["Nope."]), tts, wake)
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            wake.wake()
            t[0] = 31.0
            session.last_audio_at = time.monotonic() - 600.0  # mic was off
            pipeline._on_audio(session, np.zeros(320, dtype=np.int16).tobytes())
            assert wake.active is False
            await asyncio.sleep(0.2)
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert [e["active"] for e in ws.texts("wake")] == [False]
    assert ws.texts("agent_text") == []
    assert tts.calls == []
    assert ws.audio_frames() == 0


def test_wake_phrase_only_spawns_ack():
    tts = FakeTTS()
    ws = FakeWS()
    agent = RecordingAgent(["Nope."])
    eng = wake_engines("hey vivo", agent, tts, WakeState("hey vivo"))

    async def go() -> None:
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            start(session)
            await wait_event(ws, "reply_done")
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert [e["text"] for e in ws.texts("transcript")] == ["hey vivo"]
    assert eng.wake.active is True
    assert [e["active"] for e in ws.texts("wake")] == [True]
    # deterministic ack, no LLM round trip
    assert agent.prompts == []
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == config.WAKE_ACK
    assert tts.calls == [config.WAKE_ACK]
    assert ws.audio_frames() == 1


def test_text_after_phrase_is_processed_normally():
    tts = FakeTTS()
    ws = FakeWS()
    agent = RecordingAgent(["Four."])
    eng = wake_engines("hey vivo what is 2 plus 2", agent, tts, WakeState("hey vivo"))

    async def go() -> None:
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            start(session)
            await wait_event(ws, "reply_done")
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    # transcript keeps the full text; the LLM only sees the remainder,
    # with the ack spoken first
    assert [e["text"] for e in ws.texts("transcript")] == ["hey vivo what is 2 plus 2"]
    assert agent.prompts == ["what is 2 plus 2"]
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == config.WAKE_ACK + "Four."
    assert tts.calls == [config.WAKE_ACK, "Four."]
    assert eng.wake.active is True


def test_end_phrase_ends_session_with_goodnight():
    tts = FakeTTS()
    ws = FakeWS()
    agent = RecordingAgent(["Nope."])
    wake = WakeState("hey vivo", ("that's all",))
    eng = wake_engines("that's all", agent, tts, wake)

    async def go() -> None:
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            wake.wake()
            start(session)
            await wait_event(ws, "reply_done")
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert wake.active is False
    assert [e["active"] for e in ws.texts("wake")] == [False]
    assert agent.prompts == []
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == config.WAKE_GOODNIGHT
    # the fixed reply goes through the sentence chunker: one call per sentence
    assert " ".join(tts.calls) == config.WAKE_GOODNIGHT


def test_idle_timeout_sleeps_and_speaks_goodnight():
    tts = FakeTTS()
    ws = FakeWS()
    t = [0.0]
    wake = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])

    async def go() -> None:
        eng = wake_engines("irrelevant", RecordingAgent(["Nope."]), tts, wake)
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            wake.wake()
            t[0] = 31.0  # idle past the timeout
            pipeline._wake_tick(session, gap=0.1)  # audio flowing: user listening
            assert wake.active is False
            await wait_event(ws, "reply_done")
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert [e["active"] for e in ws.texts("wake")] == [False]
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == config.WAKE_GOODNIGHT
    assert tts.calls == [config.WAKE_GOODNIGHT]
    assert ws.audio_frames() == 1


def test_idle_timeout_after_audio_gap_sleeps_silently():
    """The mic was off (long audio gap) when the timeout elapsed: nobody was
    listening, so the session sleeps without the goodnight — but the state
    change is still broadcast."""
    tts = FakeTTS()
    ws = FakeWS()
    t = [0.0]
    wake = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])

    async def go() -> None:
        eng = wake_engines("irrelevant", RecordingAgent(["Nope."]), tts, wake)
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            wake.wake()
            t[0] = 31.0
            pipeline._wake_tick(session, gap=60.0)
            assert wake.active is False
            await asyncio.sleep(0.2)  # no goodnight worker should be running
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert [e["active"] for e in ws.texts("wake")] == [False]
    assert ws.texts("agent_text") == []
    assert tts.calls == []
    assert ws.audio_frames() == 0


def test_idle_timeout_on_new_connection_sleeps_silently():
    """A fresh connection (no audio history) detecting the stale session must
    not speak the previous session's goodnight."""
    tts = FakeTTS()
    ws = FakeWS()
    t = [0.0]
    wake = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])

    async def go() -> None:
        eng = wake_engines("irrelevant", RecordingAgent(["Nope."]), tts, wake)
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)
        try:
            wake.wake()
            t[0] = 31.0
            pipeline._wake_tick(session, gap=None)
            assert wake.active is False
            await asyncio.sleep(0.2)
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert [e["active"] for e in ws.texts("wake")] == [False]
    assert ws.texts("agent_text") == []
    assert tts.calls == []
    assert ws.audio_frames() == 0


def test_idle_timeout_suppressed_while_reply_active():
    tts = FakeTTS()
    ws = FakeWS()
    t = [0.0]
    wake = WakeState("hey vivo", ("bye",), 30.0, now=lambda: t[0])

    async def go() -> None:
        eng = wake_engines("irrelevant", FakeAgent(["Nope."]), tts, wake)
        session = VoiceSession(ws, eng)
        pipeline._ACTIVE.add(session)  # goodnight has a target
        try:
            wake.wake()
            t[0] = 31.0
            session.reply_active = True  # a reply is in progress
            pipeline._wake_tick(session, gap=0.1)
            assert wake.active is True  # not ended mid-reply
            session.reply_active = False
            pipeline._wake_tick(session, gap=0.1)
            assert wake.active is False
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert tts.calls == [config.WAKE_GOODNIGHT]  # spoken exactly once


def test_disabled_wake_keeps_legacy_always_answer():
    tts = FakeTTS()
    ws = FakeWS()
    agent = RecordingAgent(["Hi!"])

    async def go() -> None:
        eng = wake_engines("hello there", agent, tts, WakeState())
        session = VoiceSession(ws, eng)
        start(session)
        await wait_event(ws, "reply_done")

    asyncio.run(go())
    assert agent.prompts == ["hello there"]
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == "Hi!"
    assert ws.texts("wake") == []
