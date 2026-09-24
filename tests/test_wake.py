"""T024: wake-phrase session gating (offline).

WakeState unit tests (pattern matching, state transitions, fake clock,
hot-apply) plus the pipeline gate: asleep utterances are ignored silently,
a wake phrase-only utterance spawns the spoken ack without an LLM round
trip, the text after the phrase is processed normally, an end phrase ends
the session, the idle timeout speaks the goodnight (never while a reply is
in progress), and an empty phrase keeps the legacy always-answer behavior.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from app import config, pipeline
from app.conversation import SessionStore
from app.pipeline import VoiceSession
from app.wake import WakeState
from test_tts_queue import FakeAgent, FakeTTS, FakeWS, start, wait_event


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

    def reply(self, text, execute, history=None):
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


def test_asleep_ignores_utterance_silently():
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

    asyncio.run(go())
    # dropped: no transcript, no agent text, nothing synthesized, state unchanged
    # (a bare reply_done may still arrive — same as the empty-transcript path)
    assert ws.texts("transcript") == []
    assert ws.texts("agent_text") == []
    assert tts.calls == []
    assert ws.audio_frames() == 0
    assert eng.wake.active is False


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
            pipeline._wake_tick(session)
            assert wake.active is False
            await wait_event(ws, "reply_done")
        finally:
            pipeline._ACTIVE.discard(session)

    asyncio.run(go())
    assert [e["active"] for e in ws.texts("wake")] == [False]
    assert "".join(e["delta"] for e in ws.texts("agent_text")) == config.WAKE_GOODNIGHT
    assert tts.calls == [config.WAKE_GOODNIGHT]
    assert ws.audio_frames() == 1


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
            pipeline._wake_tick(session)
            assert wake.active is True  # not ended mid-reply
            session.reply_active = False
            pipeline._wake_tick(session)
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
