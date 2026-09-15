import numpy as np

from app import config
from app.models import ensure_models
from app.tts import SAMPLE_RATE, TTS


def test_synthesize_hello():
    ensure_models(config.MODEL_DIR)
    tts = TTS(model_dir=config.MODEL_DIR)
    audio = tts.synthesize("Hello world")
    assert audio.dtype == np.float32
    assert len(audio) > SAMPLE_RATE * 0.4, f"too short: {len(audio)} samples"
    assert float(np.abs(audio).max()) > 0.01, "audio is silent"
    assert tts.load_time is not None


def test_empty_text():
    ensure_models(config.MODEL_DIR)
    tts = TTS(model_dir=config.MODEL_DIR)
    audio = tts.synthesize("   ")
    assert audio.size == 0


def test_stream_chunks():
    ensure_models(config.MODEL_DIR)
    tts = TTS(model_dir=config.MODEL_DIR)
    parts = list(tts.stream(["One sentence.", "Another one."]))
    assert len(parts) == 2
    assert all(p.dtype == np.float32 and p.size for p in parts)
