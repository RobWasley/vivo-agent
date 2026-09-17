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

# The out-of-the-box voice name (T022); every .wav shipped in the repo voices/
# dir is seeded into <data>/voices on first run so a fresh deployment has
# voices available immediately.
DEFAULT_VOICE = "default"
VOICES_SRC = Path(__file__).resolve().parent.parent / "voices"


def ensure_models(model_dir: str, data_dir: str | None = None) -> None:
    """Pre-fetch every model/asset (idempotent)."""
    from huggingface_hub import snapshot_download

    for repo in HF_REPOS:
        log.info("ensuring HF model: %s", repo)
        snapshot_download(repo)
    if data_dir is not None:
        seed_voices(data_dir)
    Path(model_dir).mkdir(parents=True, exist_ok=True)


def seed_voices(data_dir: str) -> None:
    """Copy every reference clip shipped in voices/ into <data>/voices.

    Existing files are never overwritten, so user-uploaded or renamed clips
    survive upgrades and re-runs."""
    if not VOICES_SRC.is_dir():
        return
    dst_dir = Path(data_dir) / "voices"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(VOICES_SRC.glob("*.wav")):
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
            log.info("seeded voice %s -> %s", src.name, dst)
