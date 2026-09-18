"""Voice pipeline over a WebSocket connection (T007; T013/T014).

Flow per utterance: mic PCM (16 kHz) -> Silero VAD -> endpointing ->
faster-whisper STT -> agent (llama.cpp, streaming, tools) -> sentence
chunker -> bounded TTS queue -> LuxTTS voice-clone worker -> PCM chunks
(48 kHz) back to the client.

LLM and TTS run decoupled (T014): the utterance thread consumes the LLM
stream and enqueues finished sentences on a bounded per-reply queue
(config.TTS_QUEUE_SIZE); a single worker thread synthesizes and sends them
strictly in enqueue order. The producer only blocks while the queue is full,
so the LLM keeps generating while TTS speaks. Barge-in bumps the generation
and drains the queue; every TTS send is gated on generation checks (before
synth, after synth, before send) so stale audio can never leak into a fresh
reply, and the LLM stream is closed as soon as the consumer notices.

Thinking (T015): the agent runs the LLM in thinking mode — reasoning tokens
stream before (and between rounds) the spoken answer and surface as
ReasoningDelta; they are never spoken or stored. While the model is thinking
(prefill gap, reasoning stretch, or tool round) and no speakable text has
arrived for config.THINK_FILLER_FIRST_AFTER seconds, a per-reply ThinkingFiller
thread enqueues a short filler phrase ("Let me think about that.", "Working on
it.", …) onto the same TTS queue, and repeats every config.THINK_FILLER_INTERVAL
seconds while the silence goes on, so a long reasoning stretch keeps producing
fresh filler speech. Filler enqueue is non-blocking (a full queue, i.e. a TTS
backlog, skips the tick) and generation-gated, so a barged-in reply gets no
more filler.

Bench logging (T013): each utterance records phase events relative to speech
end (t0) and logs one line, `bench gen=<n> {json}` (see Bench.report), for
offline correlation (bench.py) and before/after comparison.

Protocol (JSON text frames unless noted):
  client -> server
    binary            int16 mono 16 kHz PCM (any frame size)
    {"type":"barge_in"}   cancel the active reply (stop speaking, abort LLM)
    {"type":"flush"}      force-close an in-progress utterance
    {"type":"session"}           start a new session, bind this connection to it
    {"type":"session","id":str}  bind this connection to an existing session
    {"type":"ping"}       -> {"type":"pong"}
  server -> client
    {"type":"config","barge_in":{level_threshold,sustain_ms,cooldown_ms}}
                                  sent on connect and after settings are saved
                                  (T019); UI auto-barge timings (vivo.toml [barge_in])
    {"type":"session","id":str[,"created":true]}
                                  sent on connect (the session this connection
                                  is bound to) and after a session command
    {"type":"start"}            VAD: speech started
    {"type":"end"}              VAD: utterance captured, processing begins
    {"type":"transcript","text"}
    {"type":"agent_text","delta"}   LLM text, streamed as generated
    {"type":"tool","name","result"} a tool was executed
    binary                    int16 mono TTS PCM (one chunk per sentence,
                                  sample rate in config.audio.tts_sample_rate)
    {"type":"barge_ack"}
    {"type":"reply_done"}       reply's audio is finished
    {"type":"error","message"}

While a reply is active the mic input is ignored (echo guard); interrupt by
sending barge_in, which frees the mic and invalidates the active reply at once
(the interrupted worker finishes its current (uninterruptible) TTS step but can
no longer send). The STT/TTS/agent engines are expensive and shared across
connections; each connection owns its VAD state.

Sessions (T021): conversation memory is split into named sessions managed by
a SessionStore (app/conversation.py). A connection may connect with
?session=<id> to pin a specific session; without it (or with an unknown id)
it binds to the active session. Session commands rebind the connection AND
make that session active for new connections; the new conversation applies
from the next utterance. Sessions are also managed via /api/sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue as queue_mod
import random
import re
import threading
import time

import numpy as np

from app import config, shell, tools
from app.agent import Agent, ReasoningDelta, ToolRound
from app.conversation import SessionStore
from app.memory import MemoryStore, DreamScheduler
from app.reminders import ReminderScheduler, ReminderStore, get_default_scheduler, reminder_message
from app.stt import STT
from app.tts import TTS
from app.vad import VAD

log = logging.getLogger("vivo.pipeline")

_TTS_DONE = None  # queue sentinel: no more sentences will be enqueued

# Short spoken fillers for thinking gaps (T015). Kept to one brief clause so a
# single synthesis stays short and the phrase reads naturally aloud.
# Customisable via vivo.toml [filler] phrases / THINK_FILLER_PHRASES.
FILLER_PHRASES = config.FILLER_PHRASES


def user_profile() -> str:
    """The [user] settings (T023) as a system-prompt block: who vivo is
    talking to, where they are, their time zone and units. The tools also
    read the same settings directly (weather default location/units,
    get_time timezone). Returns "" when nothing is set."""
    name = config.USER_NAME.strip()
    location = config.USER_LOCATION.strip()
    tz = config.USER_TIMEZONE.strip()
    units = config.USER_UNITS.strip().lower()
    parts = []
    if name:
        parts.append(f"Your user's name is {name}.")
    if location:
        parts.append(f"They are in {location}" + (f", time zone {tz}" if tz else "") + ".")
    elif tz:
        parts.append(f"Their time zone is {tz}.")
    if units == "imperial":
        parts.append("They use imperial units (°F, mph).")
    else:
        parts.append("They use metric units (°C, km/h).")
    parts.append(
        "Unless they ask about another place or time zone, use their location "
        "for weather questions and their time zone for time questions."
    )
    if not (name or location or tz or units != "metric"):
        return ""
    return "User profile: " + " ".join(parts)


def memory_context(memory: MemoryStore | None = None) -> str:
    """A compact summary of the local memory file for the model prompt."""
    if memory is None:
        memory = MemoryStore(path=os.path.join(config.DATA_DIR, "memory.md"))
    summary = memory.dream()
    if summary == "No important memory yet.":
        return ""
    return "Local memory: " + summary.replace("\n", " ")


class SentenceChunker:
    """Splits a stream of text deltas into speakable sentences.

    Sentence endings (. ! ?) always win. A still-open sentence is additionally
    split at clause punctuation (comma, semicolon, colon) once the buffer
    exceeds clause_max_chars, so TTS can start on the first clause instead of
    waiting for the full sentence (T020); max_chars stays the hard backstop
    for text without any punctuation.
    """

    SENT_RE = re.compile(r"(?<=[.!?])\s+")
    CLAUSE_RE = re.compile(r"[,;:] ")

    def __init__(self, max_chars: int = 180, clause_max_chars: int = 0):
        self.max_chars = max_chars
        self.clause_max_chars = clause_max_chars
        self._buf = ""

    def _clause_splits(self) -> list[str]:
        out: list[str] = []
        if not self.clause_max_chars:
            return out
        while len(self._buf) > self.clause_max_chars:
            cut = -1
            for m in self.CLAUSE_RE.finditer(self._buf):
                cut = m.end()  # last clause boundary; streaming deltas are
                # small, so it sits close to the threshold
            # A boundary too close to the start would emit a tiny fragment:
            # wait for more text (or the max_chars hard split).
            if cut < self.clause_max_chars // 2:
                break
            out.append(self._buf[:cut].strip())
            self._buf = self._buf[cut:].lstrip()
        return out

    def add(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            m = self.SENT_RE.search(self._buf)
            if not m:
                break
            out.append(self._buf[: m.end()].strip())
            self._buf = self._buf[m.end():]
        out.extend(self._clause_splits())
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


class Bench:
    """Per-utterance phase-event recorder for latency benchmarks (T013).

    All times are seconds (monotonic) relative to t0 — the moment the finished
    utterance reached the pipeline. Thread-safe: the LLM producer, the TTS
    worker and the bookkeeping code all record into the same instance.
    """

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.t0_wall = time.time()
        self._lock = threading.Lock()
        self.events: dict[str, float] = {"utterance_end": 0.0}
        self.extras: dict[str, float] = {}
        self.cancelled = False

    def mark(self, name: str) -> float:
        with self._lock:
            t = round(time.monotonic() - self.t0, 3)
            self.events[name] = t
        return t

    def record(self, name: str, value: float) -> None:
        with self._lock:
            self.events[name] = round(max(0.0, value), 3)

    def accumulate(self, key: str, secs: float) -> None:
        with self._lock:
            self.extras[key] = round(self.extras.get(key, 0.0) + secs, 3)

    def get(self, name: str) -> float | None:
        with self._lock:
            return self.events.get(name)

    def first_audio(self) -> float | None:
        """First *reply* audio sent (filler audio is excluded, T015)."""
        with self._lock:
            vals = [v for k, v in self.events.items()
                    if k.startswith("tts_") and k.endswith("_sent")]
        return min(vals) if vals else None

    def report(self, gen: int, **extra) -> None:
        """Log the per-utterance summary line consumed by bench.py."""
        with self._lock:
            ev = dict(self.events)
            fa = min((v for k, v in ev.items()
                      if k.startswith("tts_") and k.endswith("_sent")), default=None)
            if fa is not None:
                self.extras["first_audio"] = fa
            if "llm_first_token" in ev and "generation_complete" in ev:
                self.extras["llm_gen_total"] = round(
                    ev["generation_complete"] - ev["llm_first_token"], 3
                )
            payload = {
                "gen": gen,
                "t0_wall": round(self.t0_wall, 3),
                "cancelled": self.cancelled,
                "events": ev,
                **self.extras,
            }
        for k, v in extra.items():
            if v is not None:
                payload[k] = v
        log.info("bench gen=%d %s", gen, json.dumps(payload))


class ThinkingFiller:
    """Speaks short filler phrases while the model is thinking (T015).

    A daemon thread watches the reply's silence: once no speakable text has
    arrived for config.THINK_FILLER_FIRST_AFTER seconds (prefill, reasoning,
    or a tool round) it enqueues a random filler phrase on the reply's TTS
    queue, and again every config.THINK_FILLER_INTERVAL seconds while the
    silence goes on — a long reasoning stretch keeps getting fresh filler
    audio. `note_text()` is called for every speakable LLM delta. Enqueueing
    never blocks (a full queue means TTS already has backlog, so skip the
    tick) and is generation-gated, so barge-in stops the filler at once.
    """

    TICK = 0.25

    def __init__(self, session: "VoiceSession", gen: int, q: "queue_mod.Queue", bench: "Bench") -> None:
        self.session = session
        self.gen = gen
        self.q = q
        self.bench = bench
        self._lock = threading.Lock()
        self.last_text_at = time.monotonic()
        self._next_fill_at = self.last_text_at  # first filler is gated on FIRST_AFTER alone
        self._last_idx = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def note_text(self) -> None:
        """A speakable delta arrived: the silence (and the filler) ends."""
        with self._lock:
            self.last_text_at = time.monotonic()

    @property
    def count(self) -> int:
        return self._count

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self.TICK):
                return
            if not self.session.alive(self.gen):
                return  # barged in or closed: no more filler for this reply
            with self._lock:
                now = time.monotonic()
                silent = now - self.last_text_at
                if silent < config.THINK_FILLER_FIRST_AFTER:
                    continue
                if now < self._next_fill_at:
                    continue
                idx = random.randrange(len(FILLER_PHRASES))
                if idx == self._last_idx:
                    idx = (idx + 1) % len(FILLER_PHRASES)
                self._last_idx = idx
                self._next_fill_at = now + config.THINK_FILLER_INTERVAL
                phrase = FILLER_PHRASES[idx]
            try:
                self.q.put((self._count + 1, phrase, "filler"), timeout=0.2)
            except queue_mod.Full:
                # The TTS worker is already busy; skip the tick instead of
                # hammering the queue while the backlog persists.
                with self._lock:
                    self._next_fill_at = time.monotonic() + config.THINK_FILLER_INTERVAL
                continue
            with self._lock:
                self._count += 1
                n = self._count
            self.bench.mark(f"filler_{n}_queued")
            log.info("filler %d queued (%.1fs into thinking): %s", n, silent, phrase)


class Engines:
    """Shared, expensive engine singletons (one per process)."""

    def __init__(self) -> None:
        self.memory = MemoryStore(path=os.path.join(config.DATA_DIR, "memory.md"))
        self.reminders = ReminderStore(path=os.path.join(config.DATA_DIR, "reminders.json"))
        self.reminder_scheduler = get_default_scheduler(on_due=self._handle_due_reminder)
        self.reminders = self.reminder_scheduler.store
        self.dream_scheduler = DreamScheduler(
            self.memory,
            interval_seconds=config.DREAM_INTERVAL_S,
            on_state=self._handle_dream_state,
        )
        self.reminder_scheduler.start()
        self.dream_scheduler.start()
        self.stt = STT(
            model_size=config.STT_MODEL,
            compute_type=config.STT_COMPUTE_TYPE,
            download_root=config.MODEL_DIR,
            cpu_threads=config.STT_CPU_THREADS,
            language=config.STT_LANGUAGE,
            beam_size=config.STT_BEAM_SIZE,
        )
        self.tts = TTS(
            model_dir=config.MODEL_DIR,
            voice=config.TTS_VOICE,
            speed=config.TTS_SPEED,
            sentence_pause=config.TTS_SENTENCE_PAUSE,
            data_dir=config.DATA_DIR,
            cpu_threads=config.TTS_CPU_THREADS,
        )
        self.agent = Agent(
            config.LLM_BASE_URL, config.LLM_MODEL, config.PERSONA, tools=tools.TOOLS,
            thinking=config.LLM_THINKING, max_tokens=config.LLM_MAX_TOKENS,
            max_tool_rounds=config.MAX_TOOL_ROUNDS,
            system_prompt=config.SYSTEM_PROMPT,
            user_profile=user_profile() + ("\n" + memory_context(self.memory) if memory_context(self.memory) else ""),
        )
        self.sessions = SessionStore(
            data_dir=config.DATA_DIR,
            compact_after_chars=config.COMPACT_AFTER_CHARS,
            keep_recent_turns=config.KEEP_RECENT_TURNS,
        )

    def _handle_due_reminder(self, reminder: dict) -> None:
        text = str(reminder.get("text", "Reminder")).strip()
        spoken = reminder_message(text)
        self.memory.observe(f"Reminder fired: {text}")
        payload = {"type": "reminder", "text": spoken}
        for session in list(_ACTIVE):
            if session.closed:
                continue
            session.send(payload)
            try:
                audio = self.tts.synthesize(spoken)
            except Exception:  # pragma: no cover - degrade gracefully for reminders
                log.exception("reminder TTS synthesis failed for %r", spoken)
                session.send({"type": "reply_done"})
                continue
            if audio.size == 0:
                session.send({"type": "reply_done"})
                continue
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            session.send(pcm16.tobytes())
            session.send({"type": "reply_done"})
        log.info("reminder fired: %s", spoken)

    def _handle_dream_state(self, active: bool) -> None:
        broadcast_dream_state(active)

    def close(self) -> None:
        self.reminder_scheduler.stop()
        self.dream_scheduler.stop()
        self.agent.close()


def apply_config(engines: Engines) -> None:
    """Hot-apply the current config to the live engines (T019).

    Called after a settings save, once config.refresh() has re-read the
    file. Values read per call (filler timings, exec/web limits) pick up
    automatically; this patches the instances that captured theirs at
    startup. STT model/compute/threads need a restart (the model loads
    once); VAD and chunker settings apply to new connections/utterances.
    """
    global FILLER_PHRASES
    a = engines.agent
    a.base_url = config.LLM_BASE_URL.rstrip("/")
    a.model = config.LLM_MODEL
    a.persona = config.PERSONA
    a.system_prompt = config.SYSTEM_PROMPT
    a.user_profile = user_profile()
    a.thinking = config.LLM_THINKING
    a.max_tokens = config.LLM_MAX_TOKENS
    a.max_tool_rounds = config.MAX_TOOL_ROUNDS
    t = engines.tts
    t.voice = config.TTS_VOICE
    t.speed = config.TTS_SPEED
    t.sentence_pause = config.TTS_SENTENCE_PAUSE
    s = engines.stt
    s.language = config.STT_LANGUAGE
    s.beam_size = config.STT_BEAM_SIZE
    engines.sessions.set_limits(
        config.COMPACT_AFTER_CHARS, config.KEEP_RECENT_TURNS
    )
    if hasattr(engines, "dream_scheduler") and config.DREAM_INTERVAL_S > 0:
        engines.dream_scheduler.interval_seconds = config.DREAM_INTERVAL_S
    shell.MAX_TIMEOUT = config.EXEC_MAX_TIMEOUT  # shell.py snapshots it at import
    FILLER_PHRASES = config.FILLER_PHRASES  # ThinkingFiller reads the module global


class VoiceSession:
    """Per-connection pipeline state."""

    def __init__(self, ws, engines: Engines, session_id: str | None = None) -> None:
        self.ws = ws
        self.engines = engines
        self.loop = asyncio.get_running_loop()
        store = engines.sessions
        if session_id is not None and session_id not in store.ids():
            log.warning("unknown session %r: binding to active %s",
                        session_id, store.active_id)
            session_id = None
        self.session_id = session_id if session_id is not None else store.active_id
        self.conversation = store.conversation_for(self.session_id)
        self.vad = VAD(
            threshold=config.VAD_THRESHOLD,
            min_speech_ms=config.VAD_MIN_SPEECH_MS,
            min_silence_ms=config.VAD_MIN_SILENCE_MS,
            reopen_ms=config.VAD_REOPEN_MS,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
            max_speech_s=config.VAD_MAX_SPEECH_S,
        )
        self.lock = threading.Lock()
        self.reply_active = False
        self.generation = 0
        self.closed = False
        self.tts_queue: queue_mod.Queue | None = None  # current reply's TTS queue
        self.barge_mono: dict[int, float] = {}         # gen -> monotonic barge-in time

    def alive(self, gen: int) -> bool:
        """True while the reply of generation `gen` is still current."""
        with self.lock:
            return not self.closed and self.generation == gen

    def switch_session(self, session_id: str) -> bool:
        """Rebind this connection to another session (T021).

        Affects the next utterance only: an in-flight reply keeps the
        conversation it started with. False if the session is unknown.
        """
        store = self.engines.sessions
        if session_id not in store.ids():
            return False
        self.session_id = session_id
        self.conversation = store.conversation_for(session_id)
        return True

    def release(self) -> None:
        """Barge-in: free the mic, invalidate the active reply, drain its queue.

        Draining unblocks a producer waiting on a full queue; the worker then
        sees failed generation checks and discards the rest of the stale work.
        The interrupted worker keeps running until its current (uninterruptible)
        TTS step finishes, but `alive(gen)` then fails so it can never send
        stale audio; fresh mic input is fed to the VAD immediately.
        """
        with self.lock:
            self.reply_active = False
            self.generation += 1
            self.barge_mono[self.generation - 1] = time.monotonic()
            q, self.tts_queue = self.tts_queue, None
        if q is not None:
            while True:
                try:
                    item = q.get_nowait()
                except queue_mod.Empty:
                    break
                if item is _TTS_DONE:
                    # keep the sentinel: the worker needs it to terminate,
                    # and the producer blocks in worker.join() until it does
                    q.put_nowait(_TTS_DONE)
                    break

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
            self.tts_queue = queue_mod.Queue(maxsize=config.TTS_QUEUE_SIZE)
            q = self.tts_queue
        self.send({"type": "end"})
        threading.Thread(
            target=_handle_utterance, args=(samples, self, gen, q), daemon=True
        ).start()
        return True


def _tts_worker(
    q: "queue_mod.Queue", session: VoiceSession, gen: int, bench: Bench
) -> None:
    """Single TTS consumer (T014): synthesize + send strictly in enqueue order.

    Every send is gated on generation checks — before synth (skip stale work),
    after synth (drop stale results), before send (no stale audio) — so a
    barged-in reply can never leak audio into the next generation.
    """
    tts = session.engines.tts
    while True:
        item = q.get()
        if item is _TTS_DONE:
            return
        n, sentence, kind = item
        if not session.alive(gen):
            continue  # stale: drop without synthesizing
        ev = f"{kind}_{n}"
        bench.mark(f"{ev}_start")
        t0 = time.monotonic()
        audio = tts.synthesize(sentence)
        dt = time.monotonic() - t0
        bench.mark(f"{ev}_end")
        bench.accumulate("tts_total", dt)
        log.info("tts %.2fs for %d chars (%s%d)", dt, len(sentence), kind[0], n)
        if not session.alive(gen) or audio.size == 0:
            continue  # stale result: never send
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        if not session.alive(gen):
            continue  # stale: pre-send check
        bench.mark(f"{ev}_sent")
        session.send(pcm16.tobytes())


def _handle_utterance(
    samples: np.ndarray, session: VoiceSession, gen: int, q: "queue_mod.Queue"
) -> None:
    engines = session.engines
    # Pin this utterance's conversation now: a mid-reply session switch must
    # not rewrite the history this reply reads from or writes to (T021).
    conv = session.conversation
    t_start = time.monotonic()
    bench = Bench(t_start)
    text = ""
    spoken: list[str] = []
    worker: threading.Thread | None = None
    filler: ThinkingFiller | None = None
    n_sentences = 0
    in_thinking = False
    think_started: float | None = None
    thinking_secs = 0.0
    try:
        text = engines.stt.transcribe(samples)
        bench.mark("stt_end")
        log.info("stt %.2fs: %r", bench.events["stt_end"], text[:100])
        if not session.alive(gen):
            return
        session.send({"type": "transcript", "text": text})
        if not text.strip():
            return

        worker = threading.Thread(
            target=_tts_worker, args=(q, session, gen, bench), daemon=True
        )
        worker.start()

        chunker = SentenceChunker(
            max_chars=config.SENTENCE_MAX_CHARS,
            clause_max_chars=config.CLAUSE_MAX_CHARS,
        )
        history = conv.messages()
        filler = ThinkingFiller(session, gen, q, bench)
        filler.start()

        def enqueue(sentence: str) -> None:
            nonlocal n_sentences
            n_sentences += 1
            bench.mark(f"sentence_{n_sentences}_ready")
            t0 = time.monotonic()
            q.put((n_sentences, sentence, "tts"))  # blocks only while full
            bench.accumulate("llm_blocked_on_tts", time.monotonic() - t0)

        def execute(name: str, args: dict) -> str:
            result = tools.execute(name, args)
            log.info("tool %s -> %s", name, str(result)[:120])
            if session.alive(gen):
                session.send({"type": "tool", "name": name, "result": str(result)})
            return result

        for item in engines.agent.reply(text, execute, history=history):
            if not session.alive(gen):
                break
            if isinstance(item, ReasoningDelta):
                if not in_thinking:
                    in_thinking = True
                    think_started = time.monotonic()
                    if "thinking_start" not in bench.events:
                        bench.mark("thinking_start")
                continue  # thinking: never spoken, never stored
            if in_thinking:
                in_thinking = False
                bench.mark("thinking_end")
                thinking_secs += time.monotonic() - think_started
            if isinstance(item, ToolRound):
                continue
            if "llm_first_token" not in bench.events:
                bench.mark("llm_first_token")
            spoken.append(item)
            filler.note_text()
            session.send({"type": "agent_text", "delta": item})
            for sentence in chunker.add(item):
                enqueue(sentence)
        for sentence in chunker.flush():
            enqueue(sentence)
        bench.mark("generation_complete")
    except Exception as e:  # noqa: BLE001 - surface pipeline errors to the client
        log.exception("pipeline error")
        if session.alive(gen):
            session.send({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        if filler is not None:
            filler.stop()
        if worker is not None:
            q.put(_TTS_DONE)
            worker.join()
        barge_at = session.barge_mono.pop(gen, None)
        if barge_at is not None:
            bench.cancelled = True
            bench.record("barge_in", barge_at - t_start)
            bench.mark("stale_stop")
        with session.lock:
            is_current = session.generation == gen
            if is_current:
                session.reply_active = False
                session.generation += 1
                session.tts_queue = None
        answer = "".join(spoken).strip()
        if answer:
            conv.add_turn(text, answer)
        conv.maybe_compact(engines.agent.summarize)
        if is_current and not session.closed:
            session.send({"type": "reply_done"})
            bench.mark("reply_done")

        def _fmt(name: str) -> str:
            v = bench.get(name)
            return f"{v:.2f}s" if v is not None else "n/a"

        fa = bench.first_audio()
        n_fillers = filler.count if filler is not None else 0
        log.info(
            "utterance %.2fs (stt %s, thinking %s, first text %s, first audio %s, fillers %d)",
            time.monotonic() - t_start,
            _fmt("stt_end"),
            f"{thinking_secs:.2f}s" if thinking_secs > 0 else "n/a",
            _fmt("llm_first_token"),
            f"{fa:.2f}s" if fa is not None else "n/a",
            n_fillers,
        )
        bench.report(
            gen,
            sentences=n_sentences,
            fillers=n_fillers,
            thinking=round(thinking_secs, 3) if thinking_secs > 0 else None,
            queue_max=config.TTS_QUEUE_SIZE,
        )


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


_ACTIVE: set[VoiceSession] = set()


def barge_config_msg() -> dict:
    """The `config` handshake frame (UI barge + audio settings)."""
    return {
        "type": "config",
        "barge_in": {
            "level_threshold": config.BARGE_LEVEL_THRESHOLD,
            "sustain_ms": config.BARGE_SUSTAIN_MS,
            "cooldown_ms": config.BARGE_COOLDOWN_MS,
        },
        "audio": {
            "tts_sample_rate": config.TTS_SAMPLE_RATE,
        },
    }


def broadcast_config() -> None:
    """Re-send the `config` handshake to every open session (T019)."""
    msg = barge_config_msg()
    for s in list(_ACTIVE):
        if not s.closed:
            s.send(msg)


def broadcast_dream_state(active: bool) -> None:
    """Report whether the background dream pass is running."""
    msg = {"type": "dream", "active": bool(active)}
    for s in list(_ACTIVE):
        if not s.closed:
            s.send(msg)


async def serve_session(ws, engines: Engines) -> None:
    # ?session=<id> pins this connection to a named session; default is the
    # active one (unknown ids fall back in VoiceSession.__init__).
    pinned = ws.query_params.get("session") if hasattr(ws, "query_params") else None
    session = VoiceSession(ws, engines, session_id=pinned or None)
    _ACTIVE.add(session)
    # Session handshake, then UI tuning that used to be hardcoded in
    # static/app.js (vivo.toml [barge_in]).
    session.send({"type": "session", "id": session.session_id})
    session.send(barge_config_msg())
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
            elif kind == "session":
                target = cmd.get("id")
                if not target:
                    new_id = engines.sessions.create()
                    session.session_id = new_id
                    session.conversation = engines.sessions.conversation_for(new_id)
                    session.send({"type": "session", "id": new_id, "created": True})
                    log.info("new session %s", new_id)
                elif session.switch_session(target):
                    engines.sessions.set_active(target)
                    session.send({"type": "session", "id": target})
                else:
                    session.send({"type": "error", "message": f"unknown session: {target}"})
            elif kind == "flush":
                for ev in session.vad.flush():
                    if ev.type == "start":
                        session.send({"type": "start"})
                    elif ev.samples is not None and ev.samples.size >= 320:
                        session.start_utterance(ev.samples)
            elif kind == "ping":
                session.send({"type": "pong"})
    finally:
        _ACTIVE.discard(session)
        session.closed = True
        log.info("session closed")
