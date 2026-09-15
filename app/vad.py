"""Streaming Silero v6 VAD with endpointing (SPEC D006).

Runs the silero_vad_v6.onnx model bundled with faster-whisper via onnxruntime.
The model consumes 512-sample (32 ms) windows at 16 kHz, each prefixed with the
last 64 samples of the previous window (context), and carries hidden/cell state
between calls. Feed audio in any frame size; call process() per audio chunk.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

SR = 16000
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64


def default_model_path() -> str:
    import faster_whisper

    return str(Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx")


@dataclass
class VadEvent:
    """type is 'start' or 'end'. 'end' carries the utterance audio (16 kHz float32)."""

    type: str
    start: float  # seconds, absolute within this VAD's stream
    end: float | None
    samples: np.ndarray | None


class VAD:
    """Streaming voice activity detector with speech endpointing."""

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        min_speech_ms: int = 200,
        min_silence_ms: int = 400,
        speech_pad_ms: int = 200,
        max_speech_s: float = 30.0,
    ):
        path = model_path or default_model_path()
        self.threshold = threshold
        self.neg_threshold = threshold - 0.15
        self.min_speech_samples = int(min_speech_ms * SR / 1000)
        self.min_silence_samples = int(min_silence_ms * SR / 1000)
        self.pad_samples = int(speech_pad_ms * SR / 1000)
        self.max_speech_samples = int(max_speech_s * SR)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)

        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._prev_ctx = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._tail = np.zeros(0, dtype=np.float32)
        self._total = 0
        self._in_speech = False
        self._speech_start = 0
        self._last_voiced = 0
        self._ring: deque[tuple[int, np.ndarray]] = deque(
            maxlen=int(self.max_speech_samples / WINDOW_SAMPLES) + 16
        )

    @staticmethod
    def _to_float(pcm: np.ndarray) -> np.ndarray:
        if pcm.dtype == np.float32:
            return pcm
        if pcm.dtype in (np.int16, np.int32):
            max_val = float(np.iinfo(pcm.dtype).max)
            return (pcm / max_val).astype(np.float32)
        raise TypeError(f"unsupported PCM dtype: {pcm.dtype}")

    def process(self, pcm: np.ndarray) -> list[VadEvent]:
        """Feed 16 kHz audio (int16/float32) and return any completed events."""
        data = self._to_float(pcm)
        buf = np.concatenate([self._tail, data]) if self._tail.size else data
        buf_start = self._total
        events: list[VadEvent] = []
        n_full = len(buf) - len(buf) % WINDOW_SAMPLES
        for i in range(0, n_full, WINDOW_SAMPLES):
            events.extend(self._on_window(buf, i, buf_start))
        self._total = buf_start + n_full
        self._tail = buf[n_full:]
        return events

    def flush(self) -> list[VadEvent]:
        """Close any open utterance; call when the audio stream stops."""
        events: list[VadEvent] = []
        if self._in_speech:
            if self._last_voiced - self._speech_start >= self.min_speech_samples:
                events.append(self._finish(self._last_voiced))
            self._in_speech = False
        return events

    def _on_window(self, buf: np.ndarray, i: int, buf_start: int) -> list[VadEvent]:
        window = buf[i : i + WINDOW_SAMPLES]
        win_start = buf_start + i
        events: list[VadEvent] = []

        row = np.concatenate([self._prev_ctx, window]).astype(np.float32)[None, :]
        prob, self._h, self._c = self.session.run(None, {"input": row, "h": self._h, "c": self._c})
        self._prev_ctx = window[-CONTEXT_SAMPLES:].astype(np.float32)
        self._ring.append((win_start, window.astype(np.float32)))
        p = float(np.asarray(prob).ravel()[0])
        end = win_start + WINDOW_SAMPLES

        if p >= self.threshold:
            self._last_voiced = end
            if not self._in_speech:
                self._in_speech = True
                start = max(0, win_start - self.pad_samples)
                self._speech_start = start
                events.append(VadEvent("start", start / SR, None, None))
        elif p < self.neg_threshold and self._in_speech:
            silence = end - self._last_voiced
            if silence >= self.min_silence_samples:
                if self._last_voiced - self._speech_start >= self.min_speech_samples:
                    events.append(self._finish(self._last_voiced))
                else:
                    logger.debug(
                        "dropped %d ms sub-min_speech utterance",
                        (self._last_voiced - self._speech_start) // 16,
                    )
                self._in_speech = False

        if self._in_speech and end - self._speech_start >= self.max_speech_samples:
            events.append(self._finish(self._last_voiced))
            self._speech_start = self._last_voiced

        return events

    def _segment(self, lo_abs: int, hi_abs: int) -> np.ndarray:
        out: list[np.ndarray] = []
        for abs_start, w in self._ring:
            if abs_start + WINDOW_SAMPLES <= lo_abs:
                continue
            if abs_start >= hi_abs:
                break
            a = max(0, lo_abs - abs_start)
            b = min(WINDOW_SAMPLES, hi_abs - abs_start)
            out.append(w[a:b])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def _finish(self, end_abs: int) -> VadEvent:
        seg = self._segment(self._speech_start, end_abs)
        return VadEvent("end", self._speech_start / SR, end_abs / SR, seg)
