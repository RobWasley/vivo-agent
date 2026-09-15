"""Voice pipeline over a WebSocket connection (T007).

Flow per utterance: mic PCM (16 kHz) -> Silero VAD -> endpointing ->
faster-whisper STT -> agent (llama.cpp, streaming, tools) -> sentence
chunker -> kokoro TTS -> PCM chunks (24 kHz) back to the client.

Protocol (JSON text frames unless noted):
  client -> server
    binary            int16 mono 16 kHz PCM (any frame size)
    {"type":"barge_in"}   cancel the active reply (stop speaking, abort LLM)
    {"type":"flush"}      force-close an in-progress utterance
    {"type":"ping"}       -> {"type":"pong"}
  server -> client
    {"type":"start"}            VAD: speech started
    {"type":"end"}              VAD: utterance captured, processing begins
    {"type":"transcript","text"}
    {"type":"agent_text","delta"}   LLM text, streamed as generated
    {"type":"tool","name","result"} a tool was executed
    binary                    int16 mono 24 kHz TTS PCM (one chunk per sentence)
    {"type":"barge_ack"}
    {"type":"reply_done"}       reply's audio is finished (or was cancelled)
    {"type":"error","message"}

While a reply is active the mic input is ignored (echo guard); interrupt by
sending barge_in, which frees the mic and invalidates the active reply at once
(the interrupted worker finishes its current step but can no longer send). The
STT/TTS/agent engines are expensive and shared across connections; each
connection owns its VAD state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time

import numpy as np

from app import config, tools
from app.agent import Agent, ToolRound
from app.conversation import Conversation
from app.stt import STT
from app.tts import TTS
from app.vad import VAD

log = logging.getLogger("vivo.pipeline")


class SentenceChunker:
    """Splits a stream of text deltas into speakable sentences."""

    SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chars: int = 180):
        self.max_chars = max_chars
        self._buf = ""

    def add(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            m = self.SENT_RE.search(self._buf)
            if not m:
                break
            out.append(self._buf[: m.end()].strip())
            self._buf = self._buf[m.end():]
        while len(self._buf) > self.max_chars:
            cut = self._buf.rfind(", ", 0, self.max_chars)
            if cut < self.max_chars // 2:
                cut = self._buf.rfind(" ", 0, self.max_chars)
            if cut < self.max_chars // 2:
                cut = self.max_chars
            out.append(self._buf[:cut].strip())
            self._buf = self._buf[cut:].lstrip()
        return [s for s in out if s]

    def flush(self) -> list[str]:
        if self._buf.strip():
            out = [self._buf.strip()]
            self._buf = ""
            return out
        return []


class Engines:
    """Shared, expensive engine singletons (one per process)."""

    def __init__(self) -> None:
        self.stt = STT(download_root=config.MODEL_DIR)
        self.tts = TTS(model_dir=config.MODEL_DIR)
        self.agent = Agent(
            config.LLM_BASE_URL, config.LLM_MODEL, config.PERSONA, tools=tools.TOOLS
        )
        self.conversation = Conversation(
            data_path=os.path.join(config.DATA_DIR, "conversation.json"),
            compact_after_chars=config.COMPACT_AFTER_CHARS,
            keep_recent_turns=config.KEEP_RECENT_TURNS,
        )

    def close(self) -> None:
        self.agent.close()


class VoiceSession:
    """Per-connection pipeline state."""

    def __init__(self, ws, engines: Engines) -> None:
        self.ws = ws
        self.engines = engines
        self.loop = asyncio.get_running_loop()
        self.vad = VAD()
        self.lock = threading.Lock()
        self.reply_active = False
        self.generation = 0
        self.closed = False

    def alive(self, gen: int) -> bool:
        """True while the reply of generation `gen` is still current."""
        with self.lock:
            return not self.closed and self.generation == gen

    def release(self) -> None:
        """Barge-in: free the mic and invalidate the active reply at once.

        The interrupted worker keeps running until its current (uninterruptible)
        TTS/LLM step finishes, but `alive(gen)` then fails so it can never send
        stale audio; fresh mic input is fed to the VAD immediately.
        """
        with self.lock:
            self.reply_active = False
            self.generation += 1

    def send(self, payload) -> None:
        """Thread-safe send; a vanished client just marks the session closed."""
        if self.closed:
            return
        if isinstance(payload, (bytes, bytearray)):
            coro = self.ws.send_bytes(bytes(payload))
        else:
            coro = self.ws.send_text(json.dumps(payload))
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        fut.add_done_callback(self._on_send_done)

    def _on_send_done(self, fut) -> None:
        try:
            fut.result()
        except Exception:
            self.closed = True

    def start_utterance(self, samples: np.ndarray) -> bool:
        """Begin processing an utterance unless one is already active."""
        with self.lock:
            if self.reply_active:
                return False
            self.reply_active = True
            self.generation += 1
            gen = self.generation
        self.send({"type": "end"})
        threading.Thread(
            target=_handle_utterance, args=(samples, self, gen), daemon=True
        ).start()
        return True


def _handle_utterance(samples: np.ndarray, session: VoiceSession, gen: int) -> None:
    engines = session.engines
    t_start = time.monotonic()
    tim: dict = {"stt": None, "first_text": None, "first_audio": None}
    text = ""
    spoken: list[str] = []
    try:
        t0 = time.monotonic()
        text = engines.stt.transcribe(samples)
        tim["stt"] = time.monotonic() - t0
        log.info("stt %.2fs: %r", tim["stt"], text[:100])
        if not session.alive(gen):
            return
        session.send({"type": "transcript", "text": text})
        if not text.strip():
            return

        chunker = SentenceChunker()
        history = engines.conversation.messages()
        spoken: list[str] = []

        def speak(sentence: str) -> None:
            if not session.alive(gen):
                return
            t0 = time.monotonic()
            audio = engines.tts.synthesize(sentence)
            log.info("tts %.2fs for %d chars", time.monotonic() - t0, len(sentence))
            if not session.alive(gen) or audio.size == 0:
                return
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            if tim["first_audio"] is None:
                tim["first_audio"] = time.monotonic() - t_start
            session.send(pcm16.tobytes())

        def execute(name: str, args: dict) -> str:
            result = tools.execute(name, args)
            log.info("tool %s -> %s", name, str(result)[:120])
            if session.alive(gen):
                session.send({"type": "tool", "name": name, "result": str(result)[:300]})
            return result

        for item in engines.agent.reply(text, execute, history=history):
            if not session.alive(gen):
                break
            if isinstance(item, ToolRound):
                continue
            if tim["first_text"] is None:
                tim["first_text"] = time.monotonic() - t_start
            spoken.append(item)
            session.send({"type": "agent_text", "delta": item})
            for sentence in chunker.add(item):
                speak(sentence)
        for sentence in chunker.flush():
            speak(sentence)
    except Exception as e:  # noqa: BLE001 - surface pipeline errors to the client
        log.exception("pipeline error")
        if session.alive(gen):
            session.send({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        if tim["stt"] is not None:
            log.info(
                "utterance %.2fs (stt %.2fs, first text %s, first audio %s)",
                time.monotonic() - t_start,
                tim["stt"],
                f"{tim['first_text']:.2f}s" if tim["first_text"] is not None else "n/a",
                f"{tim['first_audio']:.2f}s" if tim["first_audio"] is not None else "n/a",
            )
        with session.lock:
            is_current = session.generation == gen
            if is_current:
                session.reply_active = False
                session.generation += 1
        answer = "".join(spoken).strip()
        if answer:
            engines.conversation.add_turn(text, answer)
        engines.conversation.maybe_compact(engines.agent.summarize)
        if is_current and not session.closed:
            session.send({"type": "reply_done"})


def _on_audio(session: VoiceSession, data: bytes) -> None:
    if len(data) < 2 or len(data) % 2:
        return
    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if session.reply_active:
        return  # echo guard: ignore the mic while the toy is speaking
    for ev in session.vad.process(pcm):
        if ev.type == "start":
            session.send({"type": "start"})
        elif ev.samples is not None and ev.samples.size >= 320:
            session.start_utterance(ev.samples)


async def serve_session(ws, engines: Engines) -> None:
    session = VoiceSession(ws, engines)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                _on_audio(session, data)
                continue
            text = msg.get("text")
            if text is None:
                continue
            try:
                cmd = json.loads(text)
            except json.JSONDecodeError:
                continue
            kind = cmd.get("type")
            if kind == "barge_in":
                session.release()
                session.send({"type": "barge_ack"})
            elif kind == "flush":
                for ev in session.vad.flush():
                    if ev.type == "start":
                        session.send({"type": "start"})
                    elif ev.samples is not None and ev.samples.size >= 320:
                        session.start_utterance(ev.samples)
            elif kind == "ping":
                session.send({"type": "pong"})
    finally:
        session.closed = True
        log.info("session closed")
