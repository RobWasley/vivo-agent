import os
import wave

import numpy as np

from app import config
from app.stt import SAMPLE_RATE, STT

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "stt_sample.wav")


def _read_wav_16k(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getnframes() > 0
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        x = np.linspace(0, len(pcm) - 1, len(pcm))
        xi = np.linspace(0, len(pcm) - 1, int(len(pcm) * SAMPLE_RATE / sr))
        pcm = np.interp(xi, x, pcm)
    return pcm.astype(np.float32)


def test_transcribe_fixture():
    pcm = _read_wav_16k(FIXTURE)
    stt = STT(download_root=config.MODEL_DIR)
    text = stt.transcribe(pcm)
    expected = {"quick", "brown", "fox", "jumps", "over", "lazy", "dog"}
    hit = expected & set(text.lower().replace(",", " ").split())
    assert len(hit) >= 5, f"transcript was: {text!r}"
    assert stt.load_time is not None
