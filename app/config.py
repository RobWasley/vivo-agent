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
