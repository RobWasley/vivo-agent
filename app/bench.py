"""In-process STT/TTS micro-benchmarks for the settings pane.

Unlike the host-side bench.py (which drives the full WS pipeline from
outside the container to measure end-to-end latency), these run inline
against fresh engine instances and return timing/quality data straight to
the UI, along with a simple suggested config. They're meant to be short,
manual, on-demand checks — not a replacement for bench.py.
"""
from __future__ import annotations

import logging
import time
import wave
from pathlib import Path

import numpy as np

from app.stt import STT
from app.tts import TTS

log = logging.getLogger("vivo.bench")

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
STT_FIXTURE = FIXTURE_DIR / "stt_sample.wav"

TTS_SAMPLE_TEXTS = (
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Good morning. Here is a quick summary of today's weather and your calendar.",
)

# Ordered smallest to largest; used to rank suggestions by accuracy.
STT_MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
STT_RTF_MARGIN = 0.7  # comfortably faster than real time


def _load_stt_fixture() -> np.ndarray:
    with wave.open(str(STT_FIXTURE)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        n16 = int(len(pcm) * 16000 / sr)
        x = np.linspace(0, len(pcm) - 1, len(pcm))
        xi = np.linspace(0, n16 - 1, n16)
        pcm = np.interp(xi, x, pcm)
    return pcm.astype(np.float32)


def bench_stt(
    models: list[str], compute_type: str, download_root: str, cpu_threads: int, language: str
) -> list[dict]:
    """Time model load + transcribe for each requested size on a fixed clip.

    Larger sizes may need to download first (can take a while); the caller
    should expect this call to block for a long time when new sizes are
    requested. Accuracy isn't scored (no ground truth) — the transcribed
    text is returned so the user can judge quality themselves.
    """
    pcm = _load_stt_fixture()
    audio_secs = len(pcm) / 16000.0
    results = []
    for model_size in models:
        if model_size not in STT_MODEL_CHOICES:
            continue
        entry: dict = {"model": model_size, "compute_type": compute_type}
        try:
            stt = STT(
                model_size=model_size,
                compute_type=compute_type,
                download_root=download_root,
                cpu_threads=cpu_threads,
                language=language,
            )
            t0 = time.monotonic()
            text = stt.transcribe(pcm)
            transcribe_s = time.monotonic() - t0
            entry.update({
                "load_s": round(stt.load_time or 0.0, 2),
                "transcribe_s": round(transcribe_s, 2),
                "audio_s": round(audio_secs, 2),
                "rtf": round(transcribe_s / audio_secs, 3) if audio_secs else None,
                "text": text,
            })
        except Exception as e:  # noqa: BLE001 - surfaced to the UI, not fatal
            entry["error"] = f"{type(e).__name__}: {e}"
        results.append(entry)
        log.info("bench stt %s/%s: %s", model_size, compute_type, entry)
    return results


def bench_tts(
    voice: str, model_dir: str, data_dir: str, speed: float, cpu_threads_options: list[int]
) -> list[dict]:
    """Time load + voice-encode + synth for candidate thread counts.

    Each candidate gets a fresh TTS instance (the live engine is untouched),
    so this reloads the model once per thread count tested.
    """
    results = []
    seen: set[int] = set()
    for threads in cpu_threads_options:
        threads = max(1, int(threads))
        if threads in seen:
            continue
        seen.add(threads)
        entry: dict = {"cpu_threads": threads}
        try:
            tts = TTS(
                model_dir=model_dir, voice=voice, speed=speed, data_dir=data_dir, cpu_threads=threads
            )
            t0 = time.monotonic()
            tts.prime(voice)
            prime_s = time.monotonic() - t0
            if tts.warm_state != "ready":
                raise RuntimeError("engine failed to warm up")
            audio_s = synth_s = 0.0
            for text in TTS_SAMPLE_TEXTS:
                t1 = time.monotonic()
                audio = tts.synthesize(text)
                synth_s += time.monotonic() - t1
                audio_s += len(audio) / 48000.0
            entry.update({
                "load_s": round(tts.load_time or 0.0, 2),
                "prime_s": round(prime_s, 2),
                "synth_s": round(synth_s, 2),
                "audio_s": round(audio_s, 2),
                "rtf": round(synth_s / audio_s, 3) if audio_s else None,
            })
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
        results.append(entry)
        log.info("bench tts threads=%s: %s", threads, entry)
    return results


def suggest_stt(results: list[dict]) -> dict | None:
    """Largest model that stays comfortably faster than real time, falling
    back to the fastest available if none qualify."""
    ok = [r for r in results if r.get("rtf") is not None and "error" not in r]
    if not ok:
        return None
    order = {m: i for i, m in enumerate(STT_MODEL_CHOICES)}
    fast_enough = [r for r in ok if r["rtf"] <= STT_RTF_MARGIN]
    if fast_enough:
        best = max(fast_enough, key=lambda r: order.get(r["model"], -1))
        reason = "largest model that stayed comfortably faster than real time"
    else:
        best = min(ok, key=lambda r: r["rtf"])
        reason = "fastest model tested (none stayed comfortably faster than real time)"
    return {"model": best["model"], "compute_type": best["compute_type"], "reason": reason}


def suggest_tts(results: list[dict]) -> dict | None:
    """Fewest threads that don't cost meaningful synth time (diminishing
    returns beyond a small margin aren't worth the extra CPU)."""
    ok = [r for r in results if r.get("rtf") is not None and "error" not in r]
    if not ok:
        return None
    fastest = min(r["rtf"] for r in ok)
    good = [r for r in ok if r["rtf"] <= fastest * 1.1]
    best = min(good, key=lambda r: r["cpu_threads"])
    return {"cpu_threads": best["cpu_threads"], "reason": "fewest threads within 10% of the best synth time"}
