"""faster-whisper STT wrapper (small int8, CPU, in-process)."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000


class STT:
    def __init__(
        self,
        model_size: str = "small",
        compute_type: str = "int8",
        download_root: str = "models",
        cpu_threads: int = 8,
        language: str = "en",
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.download_root = download_root
        self.cpu_threads = cpu_threads
        self.language = language
        self._model = None
        self.load_time: Optional[float] = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            t0 = time.monotonic()
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                download_root=self.download_root,
                cpu_threads=self.cpu_threads,
            )
            self.load_time = time.monotonic() - t0
        return self._model

    def transcribe(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe mono float32 PCM -> text. Input must be 16 kHz."""
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        if pcm.size == 0:
            return ""
        model = self._load()
        segments, _info = model.transcribe(
            pcm,
            language=self.language,
            beam_size=1,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
