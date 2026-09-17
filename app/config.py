"""Central configuration (T018; T019 UI write-back).

Precedence, highest first: environment variable > vivo.toml > built-in
default. Every vivo.toml key keeps its historical env-var name (e.g.
THINK_FILLER_FIRST_AFTER); an *empty* env value counts as unset, so
`VAR: ${VAR:-}` passthroughs in docker-compose.yml are no-ops unless the
host .env defines them. Deployment paths (MODEL_DIR, DATA_DIR, WORK_DIR)
are env-only and live in docker-compose.yml.

The web settings pane (T019) saves by rewriting vivo.toml (`write_config`)
and re-reading it in place (`refresh`); the file is regenerated with the
schema's comments (app/config_schema.dump_toml). The write is a single
small write — not rename-based, because the file is a bind mount and the
rename would cross filesystems; the loader tolerates an unreadable file
(defaults + warning) so a torn write can never kill startup.
"""
import os
import tomllib
from pathlib import Path

from app.config_schema import dump_toml

CONFIG_PATH = os.environ.get("CONFIG_PATH", "vivo.toml")


def _load_file() -> dict:
    path = Path(CONFIG_PATH)
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"vivo: ignoring unreadable {CONFIG_PATH}: {e}")
        return {}


_file = _load_file()


def _env(name: str):
    v = os.environ.get(name)
    return None if v in (None, "") else v


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _get(section: str, key: str, env, default, cast=str):
    """env var > vivo.toml [section].key > default."""
    if env is not None:
        v = _env(env)
        if v is not None:
            return cast(v)
    v = _file.get(section, {}).get(key)
    if v is not None:
        return v if cast is str else cast(v)
    return cast(default)


def _list(section: str, key: str, env, default):
    """env var (comma-separated) > vivo.toml array > default."""
    if env is not None:
        v = _env(env)
        if v is not None:
            return tuple(s.strip() for s in v.split(",") if s.strip())
    v = _file.get(section, {}).get(key)
    if v is not None:
        return tuple(str(x) for x in v)
    return tuple(default)


def _emit() -> None:
    """Recompute every public name from env + file into the module globals."""
    g = globals()

    # --- LLM ---------------------------------------------------------------
    g["LLM_BASE_URL"] = _get("llm", "base_url", "LLM_BASE_URL", "http://host.docker.internal:8080/v1")
    g["LLM_MODEL"] = _get("llm", "model", "LLM_MODEL", "qwen3.8-27b")
    # Thinking (T015): reasoning tokens stream before the spoken answer; the
    # pipeline speaks filler phrases during the silence ([filler] below).
    g["LLM_THINKING"] = _get("llm", "thinking", "LLM_THINKING", True, _bool)
    # Thinking tokens count against max_tokens; 300 was only sized for the answer.
    g["LLM_MAX_TOKENS"] = _get("llm", "max_tokens", "LLM_MAX_TOKENS", 600, int)

    # --- Persona & prompt ----------------------------------------------------
    g["PERSONA"] = _get(
        "persona", "blurb", "PERSONA",
        "You are vivo, a helpful voice assistant. Keep replies to one or two short spoken sentences.",
    )
    g["SYSTEM_PROMPT"] = _get(
        "persona", "system_prompt", None,
        "You are a hands-free voice assistant; your replies are spoken aloud. Keep "
        "replies to one or two short spoken sentences and summarise tool results in "
        "plain words; never read raw output, code, or lists aloud. You have a "
        "sandboxed shell (exec) and file tools (read_file, write_file, list_dir) in "
        "the workspace directory, current weather, web search, and web page reading. "
        "Prefer quick commands. Before a tool call that may take a while (search, "
        "fetch, long command), first say in a few words what you are doing. If a "
        "tool fails, try once differently, then say what went wrong. Earlier "
        "conversation context may be included; use it naturally and do not repeat it back.",
    )

    # --- User profile (T023) ---------------------------------------------------
    # Injected into the system prompt so the model knows who it is talking to,
    # and used directly by the tools: location is the weather default, timezone
    # drives get_time, units drive the weather answer. Empty = not set.
    g["USER_NAME"] = _get("user", "name", "USER_NAME", "")
    g["USER_LOCATION"] = _get("user", "location", "USER_LOCATION", "")
    g["USER_TIMEZONE"] = _get("user", "timezone", "USER_TIMEZONE", "")
    g["USER_UNITS"] = _get("user", "units", "USER_UNITS", "metric")

    # --- Thinking fillers (T015) ---------------------------------------------
    g["THINK_FILLER_FIRST_AFTER"] = _get("filler", "first_after_s", "THINK_FILLER_FIRST_AFTER", 2.0, float)
    g["THINK_FILLER_INTERVAL"] = _get("filler", "interval_s", "THINK_FILLER_INTERVAL", 8.0, float)
    g["FILLER_PHRASES"] = _list(
        "filler", "phrases", "THINK_FILLER_PHRASES",
        (
            "Let me think about that.",
            "Working on it.",
            "One moment.",
            "Hmm, give me a second.",
            "Still thinking.",
            "Let me work that out.",
        ),
    )

    # --- Voice (TTS + chunking) ------------------------------------------------
    # Voice names are reference clips in <DATA_DIR>/voices (T022); "default" is
    # seeded from the repo on first run (app/models.ensure_models).
    g["TTS_VOICE"] = _get("voice", "tts_voice", "TTS_VOICE", "default")
    g["TTS_SPEED"] = _get("voice", "tts_speed", "TTS_SPEED", 1.0, float)
    g["TTS_SENTENCE_PAUSE"] = _get("voice", "sentence_pause_s", "TTS_SENTENCE_PAUSE", 0.2, float)
    # SentenceChunker hard-split threshold (chars) when no punctuation boundary is
    # found: bounds first-audio latency on punctuation-poor LLM output (T014).
    g["SENTENCE_MAX_CHARS"] = _get("voice", "sentence_max_chars", "SENTENCE_MAX_CHARS", 90, int)
    # Clause-level split (T020): once the buffered sentence exceeds this length and a
    # clause boundary (comma/semicolon/colon) exists, the clause is emitted early so
    # TTS can start before the full sentence is generated (0 = disabled).
    g["CLAUSE_MAX_CHARS"] = _get("voice", "clause_max_chars", "CLAUSE_MAX_CHARS", 40, int)
    # LLM/TTS decoupling (T014): bounded per-reply queue of sentences between the
    # LLM producer and the single TTS consumer.
    g["TTS_QUEUE_SIZE"] = _get("voice", "tts_queue_size", "TTS_QUEUE_SIZE", 2, int)

    g["AUDIO_SAMPLE_RATE"] = 16000
    g["TTS_SAMPLE_RATE"] = 48000

    # --- STT -------------------------------------------------------------------
    g["STT_MODEL"] = _get("stt", "model", "STT_MODEL", "small")
    g["STT_COMPUTE_TYPE"] = _get("stt", "compute_type", "STT_COMPUTE_TYPE", "int8")
    g["STT_CPU_THREADS"] = _get("stt", "cpu_threads", "STT_CPU_THREADS", 8, int)
    g["STT_LANGUAGE"] = _get("stt", "language", "STT_LANGUAGE", "en")
    g["STT_BEAM_SIZE"] = _get("stt", "beam_size", "STT_BEAM_SIZE", 1, int)

    # --- VAD endpointing (breath-tolerant, two-stage) ---------------------------
    g["VAD_THRESHOLD"] = _get("vad", "threshold", "VAD_THRESHOLD", 0.5, float)
    g["VAD_MIN_SPEECH_MS"] = _get("vad", "min_speech_ms", "VAD_MIN_SPEECH_MS", 200, int)
    # An utterance goes *pending* after VAD_MIN_SILENCE_MS of silence; speech
    # resuming within the additional VAD_REOPEN_MS (a breath pause) continues the
    # same utterance. Total silence before an utterance is finalized:
    # min_silence + reopen (1000 ms default — the cost of not cutting off
    # mid-sentence at every turn).
    g["VAD_MIN_SILENCE_MS"] = _get("vad", "min_silence_ms", "VAD_MIN_SILENCE_MS", 400, int)
    g["VAD_REOPEN_MS"] = _get("vad", "reopen_ms", "VAD_REOPEN_MS", 600, int)
    g["VAD_SPEECH_PAD_MS"] = _get("vad", "speech_pad_ms", "VAD_SPEECH_PAD_MS", 200, int)
    g["VAD_MAX_SPEECH_S"] = _get("vad", "max_speech_s", "VAD_MAX_SPEECH_S", 30.0, float)

    # --- Barge-in (UI timings, sent to the client over WS) ----------------------
    g["BARGE_LEVEL_THRESHOLD"] = _get("barge_in", "level_threshold", "BARGE_LEVEL_THRESHOLD", 0.25, float)
    g["BARGE_SUSTAIN_MS"] = _get("barge_in", "sustain_ms", "BARGE_SUSTAIN_MS", 250, int)
    g["BARGE_COOLDOWN_MS"] = _get("barge_in", "cooldown_ms", "BARGE_COOLDOWN_MS", 700, int)

    # --- Conversation memory ----------------------------------------------------
    # Compact history once it grows past this many characters (roughly /4 tokens),
    # keeping the most recent N turns verbatim.
    g["COMPACT_AFTER_CHARS"] = _get("memory", "compact_after_chars", "COMPACT_AFTER_CHARS", 12000, int)
    g["KEEP_RECENT_TURNS"] = _get("memory", "keep_recent_turns", "KEEP_RECENT_TURNS", 4, int)

    # --- Agent tools -------------------------------------------------------------
    g["MAX_TOOL_ROUNDS"] = _get("agent", "max_tool_rounds", "MAX_TOOL_ROUNDS", 8, int)
    # Shell exec limits (voice latency: keep commands short by default).
    g["EXEC_TIMEOUT"] = _get("agent", "exec_timeout_s", "EXEC_TIMEOUT", 60, int)
    g["EXEC_MAX_TIMEOUT"] = _get("agent", "exec_max_timeout_s", "EXEC_MAX_TIMEOUT", 120, int)
    g["EXEC_MAX_OUTPUT"] = _get("agent", "exec_max_output_chars", "EXEC_MAX_OUTPUT", 8000, int)
    # Web tools.
    g["SEARCH_MAX_RESULTS"] = _get("agent", "search_max_results", "SEARCH_MAX_RESULTS", 5, int)
    g["FETCH_MAX_CHARS"] = _get("agent", "fetch_max_chars", "FETCH_MAX_CHARS", 6000, int)

    # --- Deployment (env-only, docker-compose.yml) -------------------------------
    g["MODEL_DIR"] = os.environ.get("MODEL_DIR", "models")
    g["DATA_DIR"] = os.environ.get("DATA_DIR", "data")
    # LuxTTS (onnxruntime) inference threads; not a user-facing setting (T022).
    g["TTS_CPU_THREADS"] = int(os.environ.get("TTS_CPU_THREADS") or 8)
    # Agent workspace: mounted host dir that is the exec working directory and
    # the sandbox root for the file tools (read_file, write_file, list_dir).
    g["WORK_DIR"] = os.environ.get("WORK_DIR", g["DATA_DIR"])


_emit()


def refresh() -> None:
    """Re-read vivo.toml (+ env) and update all public names in place (T019)."""
    global _file
    _file = _load_file()
    _emit()


def effective_dict() -> dict:
    """Current effective values shaped like vivo.toml (section -> key -> value)."""
    g = globals()
    return {
        "llm": {
            "base_url": g["LLM_BASE_URL"],
            "model": g["LLM_MODEL"],
            "thinking": g["LLM_THINKING"],
            "max_tokens": g["LLM_MAX_TOKENS"],
        },
        "persona": {
            "blurb": g["PERSONA"],
            "system_prompt": g["SYSTEM_PROMPT"],
        },
        "user": {
            "name": g["USER_NAME"],
            "location": g["USER_LOCATION"],
            "timezone": g["USER_TIMEZONE"],
            "units": g["USER_UNITS"],
        },
        "filler": {
            "first_after_s": g["THINK_FILLER_FIRST_AFTER"],
            "interval_s": g["THINK_FILLER_INTERVAL"],
            "phrases": list(g["FILLER_PHRASES"]),
        },
        "voice": {
            "tts_voice": g["TTS_VOICE"],
            "tts_speed": g["TTS_SPEED"],
            "sentence_pause_s": g["TTS_SENTENCE_PAUSE"],
            "sentence_max_chars": g["SENTENCE_MAX_CHARS"],
            "clause_max_chars": g["CLAUSE_MAX_CHARS"],
            "tts_queue_size": g["TTS_QUEUE_SIZE"],
        },
        "stt": {
            "model": g["STT_MODEL"],
            "compute_type": g["STT_COMPUTE_TYPE"],
            "cpu_threads": g["STT_CPU_THREADS"],
            "language": g["STT_LANGUAGE"],
            "beam_size": g["STT_BEAM_SIZE"],
        },
        "vad": {
            "threshold": g["VAD_THRESHOLD"],
            "min_speech_ms": g["VAD_MIN_SPEECH_MS"],
            "min_silence_ms": g["VAD_MIN_SILENCE_MS"],
            "reopen_ms": g["VAD_REOPEN_MS"],
            "speech_pad_ms": g["VAD_SPEECH_PAD_MS"],
            "max_speech_s": g["VAD_MAX_SPEECH_S"],
        },
        "barge_in": {
            "level_threshold": g["BARGE_LEVEL_THRESHOLD"],
            "sustain_ms": g["BARGE_SUSTAIN_MS"],
            "cooldown_ms": g["BARGE_COOLDOWN_MS"],
        },
        "memory": {
            "compact_after_chars": g["COMPACT_AFTER_CHARS"],
            "keep_recent_turns": g["KEEP_RECENT_TURNS"],
        },
        "agent": {
            "max_tool_rounds": g["MAX_TOOL_ROUNDS"],
            "exec_timeout_s": g["EXEC_TIMEOUT"],
            "exec_max_timeout_s": g["EXEC_MAX_TIMEOUT"],
            "exec_max_output_chars": g["EXEC_MAX_OUTPUT"],
            "search_max_results": g["SEARCH_MAX_RESULTS"],
            "fetch_max_chars": g["FETCH_MAX_CHARS"],
        },
    }


def write_config(data: dict) -> None:
    """Rewrite vivo.toml from `data`, regenerating the schema comments (T019)."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(dump_toml(data))
