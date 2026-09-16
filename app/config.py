import os

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://host.docker.internal:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")
PERSONA = os.environ.get(
    "PERSONA",
    "You are vivo, a helpful voice assistant. Keep replies to one or two short spoken sentences.",
)
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DATA_DIR = os.environ.get("DATA_DIR", "data")

# Thinking (T015): run the LLM in thinking mode (reasoning tokens stream
# before the spoken answer). While no speakable text has been generated for
# THINK_FILLER_FIRST_AFTER seconds the pipeline speaks a short filler phrase,
# and repeats every THINK_FILLER_INTERVAL seconds while the silence goes on.
LLM_THINKING = os.environ.get("LLM_THINKING", "1") == "1"
# Thinking tokens count against max_tokens; 300 was only sized for the answer.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "600"))
THINK_FILLER_FIRST_AFTER = float(os.environ.get("THINK_FILLER_FIRST_AFTER", "2.0"))
THINK_FILLER_INTERVAL = float(os.environ.get("THINK_FILLER_INTERVAL", "8.0"))

AUDIO_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

# Conversation memory: compact history once it grows past this many
# characters (roughly /4 tokens), keeping the most recent N turns verbatim.
COMPACT_AFTER_CHARS = int(os.environ.get("COMPACT_AFTER_CHARS", "12000"))
KEEP_RECENT_TURNS = int(os.environ.get("KEEP_RECENT_TURNS", "4"))

# Agent workspace: mounted host dir that is the exec working directory and
# the sandbox root for the file tools (read_file, write_file, list_dir).
WORK_DIR = os.environ.get("WORK_DIR", DATA_DIR)

# Shell exec limits (voice latency: keep commands short by default).
EXEC_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "60"))
EXEC_MAX_TIMEOUT = 120
EXEC_MAX_OUTPUT = int(os.environ.get("EXEC_MAX_OUTPUT", "8000"))

# Web tools.
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))
FETCH_MAX_CHARS = int(os.environ.get("FETCH_MAX_CHARS", "6000"))

# LLM/TTS decoupling (T014): bounded per-reply queue of sentences between the
# LLM producer and the single TTS consumer — how many may await synthesis while
# the LLM keeps generating.
TTS_QUEUE_SIZE = int(os.environ.get("TTS_QUEUE_SIZE", "2"))
# SentenceChunker hard-split threshold (chars) when no punctuation boundary is
# found: bounds first-audio latency on punctuation-poor LLM output (T014).
SENTENCE_MAX_CHARS = int(os.environ.get("SENTENCE_MAX_CHARS", "90"))

# VAD endpointing (breath-tolerant, two-stage): an utterance goes *pending*
# after VAD_MIN_SILENCE_MS of silence; speech resuming within the additional
# VAD_REOPEN_MS (a breath pause) continues the same utterance. Total silence
# before an utterance is finalized: min_silence + reopen (1000 ms default —
# the cost of not cutting off mid-sentence at every turn).
VAD_MIN_SILENCE_MS = int(os.environ.get("VAD_MIN_SILENCE_MS", "400"))
VAD_REOPEN_MS = int(os.environ.get("VAD_REOPEN_MS", "600"))
