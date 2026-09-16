"""Central configuration (T016).

Precedence, highest first: environment variable > vivo.toml > built-in
default. Every vivo.toml key keeps its historical env-var name (e.g.
THINK_FILLER_FIRST_AFTER); an *empty* env value counts as unset, so
`VAR: ${VAR:-}` passthroughs in docker-compose.yml are no-ops unless the
host .env defines them. Deployment paths (MODEL_DIR, DATA_DIR, WORK_DIR)
are env-only and live in docker-compose.yml.
"""
import os
import tomllib
from pathlib import Path

CONFIG_PATH = os.environ.get("CONFIG_PATH", "vivo.toml")


def _load_file() -> dict:
    path = Path(CONFIG_PATH)
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


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


# --- LLM ---------------------------------------------------------------

LLM_BASE_URL = _get("llm", "base_url", "LLM_BASE_URL", "http://host.docker.internal:8080/v1")
LLM_MODEL = _get("llm", "model", "LLM_MODEL", "qwen3.8-27b")
# Thinking (T015): reasoning tokens stream before the spoken answer; the
# pipeline speaks filler phrases during the silence ([filler] below).
LLM_THINKING = _get("llm", "thinking", "LLM_THINKING", True, _bool)
# Thinking tokens count against max_tokens; 300 was only sized for the answer.
LLM_MAX_TOKENS = _get("llm", "max_tokens", "LLM_MAX_TOKENS", 600, int)

# --- Persona & prompt ----------------------------------------------------

PERSONA = _get(
    "persona", "blurb", "PERSONA",
    "You are vivo, a helpful voice assistant. Keep replies to one or two short spoken sentences.",
)
SYSTEM_PROMPT = _get(
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

# --- Thinking fillers (T015) ---------------------------------------------

THINK_FILLER_FIRST_AFTER = _get("filler", "first_after_s", "THINK_FILLER_FIRST_AFTER", 2.0, float)
THINK_FILLER_INTERVAL = _get("filler", "interval_s", "THINK_FILLER_INTERVAL", 8.0, float)
FILLER_PHRASES = _list(
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

TTS_VOICE = _get("voice", "tts_voice", "TTS_VOICE", "af_heart")
TTS_SPEED = _get("voice", "tts_speed", "TTS_SPEED", 1.0, float)
TTS_SENTENCE_PAUSE = _get("voice", "sentence_pause_s", "TTS_SENTENCE_PAUSE", 0.2, float)
# SentenceChunker hard-split threshold (chars) when no punctuation boundary is
# found: bounds first-audio latency on punctuation-poor LLM output (T014).
SENTENCE_MAX_CHARS = _get("voice", "sentence_max_chars", "SENTENCE_MAX_CHARS", 90, int)
# LLM/TTS decoupling (T014): bounded per-reply queue of sentences between the
# LLM producer and the single TTS consumer.
TTS_QUEUE_SIZE = _get("voice", "tts_queue_size", "TTS_QUEUE_SIZE", 2, int)

AUDIO_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

# --- STT -------------------------------------------------------------------

STT_MODEL = _get("stt", "model", "STT_MODEL", "small")
STT_COMPUTE_TYPE = _get("stt", "compute_type", "STT_COMPUTE_TYPE", "int8")
STT_CPU_THREADS = _get("stt", "cpu_threads", "STT_CPU_THREADS", 8, int)
STT_LANGUAGE = _get("stt", "language", "STT_LANGUAGE", "en")
STT_BEAM_SIZE = _get("stt", "beam_size", "STT_BEAM_SIZE", 1, int)

# --- VAD endpointing (breath-tolerant, two-stage) ---------------------------

VAD_THRESHOLD = _get("vad", "threshold", "VAD_THRESHOLD", 0.5, float)
VAD_MIN_SPEECH_MS = _get("vad", "min_speech_ms", "VAD_MIN_SPEECH_MS", 200, int)
# An utterance goes *pending* after VAD_MIN_SILENCE_MS of silence; speech
# resuming within the additional VAD_REOPEN_MS (a breath pause) continues the
# same utterance. Total silence before an utterance is finalized:
# min_silence + reopen (1000 ms default — the cost of not cutting off
# mid-sentence at every turn).
VAD_MIN_SILENCE_MS = _get("vad", "min_silence_ms", "VAD_MIN_SILENCE_MS", 400, int)
VAD_REOPEN_MS = _get("vad", "reopen_ms", "VAD_REOPEN_MS", 600, int)
VAD_SPEECH_PAD_MS = _get("vad", "speech_pad_ms", "VAD_SPEECH_PAD_MS", 200, int)
VAD_MAX_SPEECH_S = _get("vad", "max_speech_s", "VAD_MAX_SPEECH_S", 30.0, float)

# --- Barge-in (UI timings, sent to the client over WS) ----------------------

BARGE_LEVEL_THRESHOLD = _get("barge_in", "level_threshold", "BARGE_LEVEL_THRESHOLD", 0.25, float)
BARGE_SUSTAIN_MS = _get("barge_in", "sustain_ms", "BARGE_SUSTAIN_MS", 250, int)
BARGE_COOLDOWN_MS = _get("barge_in", "cooldown_ms", "BARGE_COOLDOWN_MS", 700, int)

# --- Conversation memory ----------------------------------------------------

# Compact history once it grows past this many characters (roughly /4 tokens),
# keeping the most recent N turns verbatim.
COMPACT_AFTER_CHARS = _get("memory", "compact_after_chars", "COMPACT_AFTER_CHARS", 12000, int)
KEEP_RECENT_TURNS = _get("memory", "keep_recent_turns", "KEEP_RECENT_TURNS", 4, int)

# --- Agent tools -------------------------------------------------------------

MAX_TOOL_ROUNDS = _get("agent", "max_tool_rounds", "MAX_TOOL_ROUNDS", 8, int)
# Shell exec limits (voice latency: keep commands short by default).
EXEC_TIMEOUT = _get("agent", "exec_timeout_s", "EXEC_TIMEOUT", 60, int)
EXEC_MAX_TIMEOUT = _get("agent", "exec_max_timeout_s", "EXEC_MAX_TIMEOUT", 120, int)
EXEC_MAX_OUTPUT = _get("agent", "exec_max_output_chars", "EXEC_MAX_OUTPUT", 8000, int)
# Web tools.
SEARCH_MAX_RESULTS = _get("agent", "search_max_results", "SEARCH_MAX_RESULTS", 5, int)
FETCH_MAX_CHARS = _get("agent", "fetch_max_chars", "FETCH_MAX_CHARS", 6000, int)

# --- Deployment (env-only, docker-compose.yml) -------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DATA_DIR = os.environ.get("DATA_DIR", "data")
# Agent workspace: mounted host dir that is the exec working directory and
# the sandbox root for the file tools (read_file, write_file, list_dir).
WORK_DIR = os.environ.get("WORK_DIR", DATA_DIR)
