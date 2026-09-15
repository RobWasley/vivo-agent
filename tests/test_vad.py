"""T002 verification: VAD triggers, endpointing splits, silence produces nothing."""

import wave
from pathlib import Path

import numpy as np

from app.vad import VAD

SR = 16000
FIXTURE = Path(__file__).parent / "fixtures" / "stt_sample.wav"


def load_speech() -> np.ndarray:
    """The TTS fixture ('The quick brown fox...') resampled to 16 kHz float32."""
    with wave.open(str(FIXTURE)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n16 = int(len(pcm) * SR / sr)
    x = np.linspace(0, len(pcm) - 1, len(pcm))
    xi = np.linspace(0, len(pcm) - 1, n16)
    return np.interp(xi, x, pcm).astype(np.float32)


SPEECH = load_speech()  # ~2.84 s of clean speech


def silence(s: float) -> np.ndarray:
    return np.zeros(int(s * SR), dtype=np.float32)


def run(vad: VAD, sig: np.ndarray, frame: int = 1024) -> list:
    events = []
    for i in range(0, len(sig), frame):
        events.extend(vad.process(sig[i : i + frame]))
    return events


def test_endpointing_single_speech():
    vad = VAD(min_silence_ms=400, speech_pad_ms=200, min_speech_ms=200)
    sig = np.concatenate([silence(0.5), SPEECH, silence(1.0)])
    events = run(vad, sig)
    starts = [e for e in events if e.type == "start"]
    ends = [e for e in events if e.type == "end"]
    assert len(starts) == 1, f"got {len(starts)} starts: {events}"
    assert len(ends) == 1, f"got {len(ends)} ends: {events}"
    # Speech occupies 0.5s..~3.0s of the signal (0.06s..2.6s inside the fixture).
    assert starts[0].start < 0.6
    assert 2.8 < ends[0].end < 3.6
    seg = ends[0].samples
    assert seg is not None and seg.dtype == np.float32
    assert 2.0 * SR < len(seg) < 4.0 * SR, f"segment length {len(seg) / SR:.2f}s"


def test_two_segments_split_on_silence():
    vad = VAD(min_silence_ms=400, min_speech_ms=200)
    gap = silence(0.6)
    sig = np.concatenate([SPEECH, gap, SPEECH, silence(1.0)])
    events = run(vad, sig)
    ends = [e for e in events if e.type == "end"]
    assert len(ends) == 2, f"got {len(ends)} segments: {ends}"
    gap_start = len(SPEECH)
    # First segment ends before the gap; second starts inside/after the gap.
    assert ends[0].end * SR < gap_start + 0.4 * SR
    assert ends[1].start * SR > gap_start - 0.3 * SR
    assert ends[1].end > ends[0].end


def test_silence_only_no_events():
    vad = VAD(min_silence_ms=400, min_speech_ms=200)
    events = run(vad, silence(3.0))
    assert events == []


def test_short_segment_dropped():
    # 0.2 s of speech is below min_speech_ms=500 -> no end event.
    vad = VAD(min_speech_ms=500, min_silence_ms=400)
    sig = np.concatenate([SPEECH[: int(0.2 * SR)], silence(1.0)])
    ends = [e for e in run(vad, sig) if e.type == "end"]
    assert len(ends) == 0, f"expected no end events, got {ends}"


def test_partial_frames_buffered():
    vad = VAD(min_silence_ms=400, speech_pad_ms=200, min_speech_ms=200)
    sig = np.concatenate([silence(0.5), SPEECH, silence(1.0)])
    frame = 512 * 3 + 7  # awkward frame size, not a multiple of the 512 window
    events = run(vad, sig, frame=frame)
    ends = [e for e in events if e.type == "end"]
    assert len(ends) == 1, f"got {len(ends)} segments: {ends}"
    assert 2.8 < ends[0].end < 3.6
