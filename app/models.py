"""Model/asset bootstrap. Models live on a mounted volume (or the HF cache)
and are fetched once, on first run.

The LuxTTS CPU loader (zipvoice) resolves both of its models through the
Hugging Face cache — set HF_HOME to persist them (compose: /models/hf) — so
"download" here means pre-fetching into that cache. The faster-whisper model
still downloads lazily on first transcription (download_root=model_dir).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("vivo.models")

# Hugging Face repos the TTS engine needs (see app/tts.py).
HF_REPOS = ("YatharthS/LuxTTS", "openai/whisper-tiny")

# Voice reference clip shipped with the repo; seeded into <data>/voices on
# first run so a fresh deployment has a voice out of the box (T022).
DEFAULT_VOICE = "default"
DEFAULT_VOICE_SRC = Path(__file__).resolve().parent.parent / "voices" / "default.wav"


def ensure_models(model_dir: str, data_dir: str | None = None) -> None:
    """Pre-fetch every model/asset (idempotent)."""
    from huggingface_hub import snapshot_download

    for repo in HF_REPOS:
        log.info("ensuring HF model: %s", repo)
        snapshot_download(repo)
    if data_dir is not None and DEFAULT_VOICE_SRC.is_file():
        dst = Path(data_dir) / "voices" / f"{DEFAULT_VOICE}.wav"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEFAULT_VOICE_SRC, dst)
            log.info("seeded default voice -> %s", dst)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
