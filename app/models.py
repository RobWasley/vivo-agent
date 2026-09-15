"""Model download/management. Models live on a mounted volume and are
downloaded once, on first run."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

log = logging.getLogger("vivo.models")

_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"

KOKORO_FILES = {
    "kokoro-v1.0.int8.onnx": f"{_RELEASE}/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": f"{_RELEASE}/voices-v1.0.bin",
}


def ensure_models(model_dir: str) -> None:
    """Download any missing model files into model_dir (idempotent).

    Kokoro files are fetched here; the faster-whisper model downloads
    lazily on first transcription (download_root=model_dir).
    """
    kokoro_dir = Path(model_dir) / "kokoro"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    for name, url in KOKORO_FILES.items():
        dest = kokoro_dir / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            log.info("model present: %s", dest)
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        log.info("downloading %s (%s)...", name, url)
        with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(dest)
        log.info("downloaded %s", dest)
