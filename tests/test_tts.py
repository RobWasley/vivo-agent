import io

import numpy as np
import soundfile as sf

from app import config
from app.models import ensure_models
from app.tts import SAMPLE_RATE, TTS


def _tts(data_dir=None):
    ensure_models(config.MODEL_DIR, config.DATA_DIR)
    return TTS(model_dir=config.MODEL_DIR, data_dir=data_dir or config.DATA_DIR)


def _wav_bytes(secs, sr=24000, hz=440.0):
    t = np.arange(int(secs * sr)) / sr
    bio = io.BytesIO()
    sf.write(bio, (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr, format="WAV")
    return bio.getvalue()


def test_synthesize_hello():
    tts = _tts()
    audio = tts.synthesize("Hello world")
    assert audio.dtype == np.float32
    assert len(audio) > SAMPLE_RATE * 0.4, f"too short: {len(audio)} samples"
    assert float(np.abs(audio).max()) > 0.01, "audio is silent"
    assert tts.load_time is not None


def test_empty_text():
    tts = _tts()
    audio = tts.synthesize("   ")
    assert audio.size == 0


def test_stream_chunks():
    tts = _tts()
    parts = list(tts.stream(["One sentence.", "Another one."]))
    assert len(parts) == 2
    assert all(p.dtype == np.float32 and p.size for p in parts)


def test_default_voice_seeded():
    tts = _tts()
    assert "default" in tts.voices()
    assert tts.voice_path("default").exists()
    assert tts.voice_path("default").suffix == ".wav"


def test_save_voice_slug_and_range():
    tts = _tts(data_dir="/tmp/vivo-tts-test")
    slug, dur = tts.save_voice(_wav_bytes(4.0), "My Test Voice!")
    assert slug == "my-test-voice"
    assert 3.9 < dur < 4.1
    assert slug in tts.voices()

    # clips shorter than 3 s are rejected
    try:
        tts.save_voice(_wav_bytes(1.0), "short")
        raise AssertionError("short clip accepted")
    except ValueError:
        pass
    assert "short" not in tts.voices()

    # unknown formats are rejected (undecodable by libsndfile)
    try:
        tts.save_voice(b"not a wav", "garbage")
        raise AssertionError("garbage accepted")
    except ValueError:
        pass


def test_delete_voice():
    tts = _tts(data_dir="/tmp/vivo-tts-test2")
    slug, _ = tts.save_voice(_wav_bytes(3.2), "doomed")
    tts.delete_voice(slug)
    assert slug not in tts.voices()
    try:
        tts.delete_voice(slug)
        raise AssertionError("double delete accepted")
    except FileNotFoundError:
        pass
