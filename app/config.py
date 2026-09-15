import os

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://host.docker.internal:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")
PERSONA = os.environ.get(
    "PERSONA",
    "You are vivo, a helpful voice assistant. Keep replies to one or two short spoken sentences.",
)
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DATA_DIR = os.environ.get("DATA_DIR", "data")

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
