"""kokoro-onnx TTS wrapper (82M, int8 onnx, CPU, 24 kHz)."""
from __future__ import annotations

import os
import time
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE = 24000

MODEL_NAME = "kokoro-v1.0.int8.onnx"
VOICES_NAME = "voices-v1.0.bin"


class TTS:
    def __init__(
        self,
        model_dir: str = "models",
        voice: str = "af_heart",
        speed: float = 1.0,
        sentence_pause: float = 0.2,
    ):
        self.model_dir = model_dir
        self.voice = voice
        self.speed = speed
        self.sentence_pause = sentence_pause
        self._engine = None
        self.load_time: Optional[float] = None

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, "kokoro", MODEL_NAME)

    @property
    def voices_path(self) -> str:
        return os.path.join(self.model_dir, "kokoro", VOICES_NAME)

    def _load(self):
        if self._engine is None:
            from kokoro_onnx import Kokoro

            t0 = time.monotonic()
            self._engine = Kokoro(self.model_path, self.voices_path)
            self.load_time = time.monotonic() - t0
        return self._engine

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize one piece of text -> float32 mono PCM @ 24 kHz."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        engine = self._load()
        audio, sr = engine.create(
            text=text,
            voice=self.voice,
            speed=self.speed,
            sentence_pause=self.sentence_pause,
        )
        audio = np.asarray(audio, dtype=np.float32)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"unexpected TTS sample rate {sr}")
        return audio

    def stream(self, chunks: Iterator[str]) -> Iterator[np.ndarray]:
        """Synthesize a stream of text chunks (e.g. sentences) one by one."""
        for chunk in chunks:
            audio = self.synthesize(chunk)
            if audio.size:
                yield audio
