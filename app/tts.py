"""LuxTTS voice-cloning TTS wrapper (zipvoice, CPU, 48 kHz).

vivo speaks in a cloned voice: every named voice is a short reference clip
(a .wav in <data_dir>/voices, 3-30 s of clear speech). The first use of a
voice encodes the clip once (Whisper transcription + neural feature
extraction, ~1 s); every synthesis afterwards is just a flow-matching
generate (~0.2x realtime on CPU). The LuxTTS CPU loader resolves its models
through the Hugging Face cache (YatharthS/LuxTTS + openai/whisper-tiny),
pre-fetched by app/models.ensure_models.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

log = logging.getLogger("vivo.tts")

SAMPLE_RATE = 48000

LUX_MODEL = "YatharthS/LuxTTS"
REF_DURATION = 5  # seconds of the clip used for cloning
REF_RMS = 0.01
MIN_REF_SECS = 3.0
MAX_REF_SECS = 30.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (s or "voice")[:40]


class TTS:
    def __init__(
        self,
        model_dir: str = "models",
        voice: str = "default",
        speed: float = 1.0,
        sentence_pause: float = 0.2,
        data_dir: str = "data",
        cpu_threads: int = 8,
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self._voice = voice
        self.speed = speed
        self.sentence_pause = sentence_pause
        self.cpu_threads = cpu_threads
        self._engine = None
        self._engine_lock = threading.Lock()
        self._enc: Optional[dict] = None
        self._enc_voice: Optional[str] = None
        self._enc_lock = threading.RLock()
        self.load_time: Optional[float] = None

    # --- voices (reference clips) -------------------------------------------

    @property
    def voice(self) -> str:
        return self._voice

    @voice.setter
    def voice(self, name: str) -> None:
        self.set_voice(name)

    def set_voice(self, name: str) -> None:
        """Switch the active voice. The clip is encoded lazily on first use
        (or pre-encoded by prime), so this never blocks."""
        with self._enc_lock:
            if self._voice != name:
                log.info("tts voice: %s -> %s", self._voice, name)
                self._voice = name
                self._enc = None
                self._enc_voice = None

    def voices(self) -> list[str]:
        """All voice names available in the voices dir (T019 settings pane)."""
        d = self.voice_dir()
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.wav"))

    def voice_dir(self) -> Path:
        return Path(self.data_dir) / "voices"

    def voice_path(self, name: str) -> Path:
        return self.voice_dir() / f"{slugify(name)}.wav"

    def save_voice(self, data: bytes, name: str) -> tuple[str, float]:
        """Validate + store an uploaded reference clip as <data>/voices/<slug>.wav.

        Decodes any format libsndfile can read (wav, mp3, flac, ogg), downmixes
        to mono, resamples to 24 kHz int16 (the engine's feature rate), and
        enforces 3-30 s of audio. Returns (slug, duration_s).
        """
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"audio too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
        import soundfile as sf

        try:
            arr, sr = sf.read(io.BytesIO(data), dtype="float32")
        except Exception as e:
            raise ValueError(f"could not decode audio ({e})")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        dur = len(arr) / sr
        if dur < MIN_REF_SECS:
            raise ValueError(f"clip is {dur:.1f}s; need at least {MIN_REF_SECS:.0f}s of speech")
        if dur > MAX_REF_SECS:
            raise ValueError(f"clip is {dur:.1f}s; keep it under {MAX_REF_SECS:.0f}s")
        import librosa

        if sr != 24000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=24000)
        pcm16 = np.clip((arr * 32767).round(), -32768, 32767).astype("<i2")
        slug = slugify(name)
        self.voice_dir().mkdir(parents=True, exist_ok=True)
        sf.write(self.voice_path(slug), pcm16, 24000)
        log.info("voice %r stored (%.1fs)", slug, len(arr) / 24000)
        return slug, len(arr) / 24000

    def delete_voice(self, name: str) -> None:
        path = self.voice_path(name)
        if not path.exists():
            raise FileNotFoundError(name)
        path.unlink()
        log.info("voice %r deleted", name)

    # --- engine ----------------------------------------------------------------

    def _load(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from zipvoice.luxvoice import LuxTTS

                    t0 = time.monotonic()
                    self._engine = LuxTTS(LUX_MODEL, device="cpu", threads=self.cpu_threads)
                    self.load_time = time.monotonic() - t0
                    log.info("LuxTTS loaded in %.1fs", self.load_time)
        return self._engine

    def _prompt_for(self, engine, name: str | None = None) -> dict:
        """The encoded prompt for `name` (default: the active voice), encoding
        it on first use. Held under _enc_lock so a concurrent prime() and the
        TTS worker never double-encode or race the cache."""
        name = name or self._voice
        with self._enc_lock:
            if self._enc is not None and self._enc_voice == name:
                return self._enc
            path = self.voice_path(name)
            if not path.exists():
                raise FileNotFoundError(f"voice clip not found: {path}")
            t0 = time.monotonic()
            log.info("cloning voice %r (one-time encode)...", name)
            self._enc = engine.encode_prompt(str(path), duration=REF_DURATION, rms=REF_RMS)
            self._enc_voice = name
            log.info("voice %r encoded in %.1fs", name, time.monotonic() - t0)
            return self._enc

    def prime(self, name: str | None = None) -> None:
        """Load the engine and encode a clip now (run off the request path so
        the first utterance in the new voice is instant)."""
        try:
            engine = self._load()
            self._prompt_for(engine, name)
        except Exception:  # noqa: BLE001 - a failed prime just costs first use
            log.exception("could not pre-encode voice %r", name)

    # --- synthesis ----------------------------------------------------------------

    # LuxTTS's flow-matching length model can budget *fewer* frames than the
    # reference prompt for very short text (its default 1.3x speed makes this
    # common), which crashes the vocoder. We predict the frame budget and lower
    # the speed until enough frames are produced, so short sentences still work.
    _MIN_NEW_FRAMES = 24
    _SPEED_FLOOR = 0.6

    def _new_frames(self, engine, enc, text: str, internal_speed: float) -> int:
        """Predicted number of NEW feature frames `text` will generate at
        `internal_speed` (the speed LuxTTS's generate_cpu actually uses, i.e.
        the value after its internal *1.3)."""
        import torch

        te = engine.model.text_encoder
        inp = te.get_inputs()
        pt, _, pfeat, _ = enc.values()
        toks = engine.tokenizer.texts_to_token_ids([text])
        pf_len = int(pfeat.size(1))
        out = te.run(
            [te.get_outputs()[0].name],
            {
                inp[0].name: torch.tensor(toks, dtype=torch.int64).numpy(),
                inp[1].name: torch.tensor(pt, dtype=torch.int64).numpy(),
                inp[2].name: torch.tensor(pf_len, dtype=torch.int64).numpy(),
                inp[3].name: torch.tensor(float(internal_speed), dtype=torch.float32).numpy(),
            },
        )
        return int(out[0].shape[1]) - pf_len

    def _pick_speed(self, engine, enc, text: str) -> float:
        """A generate_speech `speed` arg (before LuxTTS's internal *1.3) that
        keeps the normal speaking rate for long text but slows short text down
        until it produces enough frames to be audible."""
        internal = self.speed * 1.3  # LuxTTS's "normal" internal speed
        for _ in range(10):
            if self._new_frames(engine, enc, text, internal) >= self._MIN_NEW_FRAMES:
                break
            if internal <= self._SPEED_FLOOR:
                break
            internal *= 0.85
        return internal / 1.3

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize one piece of text -> float32 mono PCM @ 48 kHz.

        Never raises for non-empty text: if the engine cannot produce audio
        (e.g. a pathological very-short text), it returns empty and the worker
        skips it rather than crashing the reply.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        engine = self._load()
        enc = self._prompt_for(engine)
        speed = self._pick_speed(engine, enc, text)
        try:
            wav = engine.generate_speech(text, enc, num_steps=4, speed=speed)
            audio = wav.detach().cpu().squeeze().numpy().astype(np.float32)
        except Exception:  # noqa: BLE001 - degrade to silence, never kill the worker
            log.exception("tts synthesis failed for %r; skipping", text)
            return np.zeros(0, dtype=np.float32)
        if audio.size < SAMPLE_RATE * 0.1:  # a sub-100 ms blip is not useful
            log.warning("tts produced only %d samples for %r", audio.size, text)
        if self.sentence_pause > 0:
            n = int(self.sentence_pause * SAMPLE_RATE)
            audio = np.concatenate([audio, np.zeros(n, dtype=np.float32)])
        return audio

    def stream(self, chunks: Iterator[str]) -> Iterator[np.ndarray]:
        """Synthesize a stream of text chunks (e.g. sentences) one by one."""
        for chunk in chunks:
            audio = self.synthesize(chunk)
            if audio.size:
                yield audio
