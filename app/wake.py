"""Wake-phrase session gating (T024).

With a non-empty wake phrase, vivo starts *asleep*: every VAD-endpointed
utterance is still transcribed, but only an utterance containing the wake
phrase reaches the LLM — everything else is dropped silently (no transcript
frame, no reply, nothing stored), so ambient speech can't flood the
transcript. The text after the wake phrase in the same utterance is processed
as a normal request; a wake-phrase-only utterance gets a short spoken ack
instead of an LLM round trip.

An awake session ends when the user speaks one of the configured end phrases,
or when the configured idle timeout elapses without an exchange (checked from
the audio path — no timer thread). An empty phrase disables the feature:
every utterance is answered, the legacy behavior.

Matching is done on the whisper transcript, not on a dedicated keyword-spotting
model (D020): the phrase words must occur in order with arbitrary
punctuation/whitespace between them, case-insensitively ("hey vivo" matches
"Hey, vivo!" and "HEY VIVO", not "hello vivo" or "hey vivov").
"""
from __future__ import annotations

import re
import threading
import time

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _words(phrase: str) -> list[str]:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", phrase.lower())).strip().split()


def _pattern(phrase: str) -> re.Pattern[str] | None:
    """Case-insensitive pattern for `phrase`: its words in order, with any run
    of non-alphanumerics (spaces, punctuation, apostrophes) allowed between
    them. Returns None for an empty phrase."""
    words = _words(phrase)
    if not words:
        return None
    return re.compile(
        r"\b" + r"[\W_]+".join(re.escape(w) for w in words) + r"\b",
        re.IGNORECASE,
    )


class WakeState:
    """Shared awake/asleep state for wake-phrase sessions (T024).

    One instance per process, shared by all connections (one mic in the
    room): a wake heard on any tab wakes vivo for all of them, and the idle
    timeout is checked from the audio path. Thread-safe; `now` is injectable
    for tests.
    """

    def __init__(
        self,
        phrase: str = "",
        end_phrases: tuple[str, ...] = (),
        timeout_s: float = 30.0,
        now=time.monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._now = now
        self._active = False
        self._last_exchange = 0.0
        self.update(phrase, end_phrases, timeout_s)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def enabled(self) -> bool:
        """True when a wake phrase is configured (the feature is on)."""
        with self._lock:
            return self._wake_pat is not None

    def update(self, phrase: str, end_phrases, timeout_s: float) -> None:
        """Hot-apply new settings (T019). An empty phrase disables the
        feature; it does not clear the awake flag (a live session ends by
        timeout or end phrase as before)."""
        with self._lock:
            self._wake_pat = _pattern(phrase)
            self._end_pats = tuple(
                (p, _pattern(p)) for p in end_phrases if _pattern(p)
            )
            self._timeout_s = max(0.0, float(timeout_s))

    def match_wake(self, text: str) -> re.Match | None:
        """The wake-phrase occurrence in `text`, if any. None when the feature
        is disabled or the phrase is absent."""
        with self._lock:
            pat = self._wake_pat
        return pat.search(text) if pat is not None else None

    def match_end(self, text: str) -> str | None:
        """The first configured end phrase found in `text` (its configured
        text), or None."""
        with self._lock:
            pats = self._end_pats
        for phrase, pat in pats:
            if pat.search(text):
                return phrase
        return None

    def wake(self) -> bool:
        """Mark awake and reset the idle clock. True if the state changed."""
        with self._lock:
            if self._active:
                return False
            self._active = True
            self._last_exchange = self._now()
            return True

    def sleep(self) -> bool:
        """Mark asleep. True if the state changed."""
        with self._lock:
            if not self._active:
                return False
            self._active = False
            return True

    def touch(self) -> None:
        """Reset the idle clock (a processed exchange)."""
        with self._lock:
            self._last_exchange = self._now()

    def claim_sleep(self) -> bool:
        """Atomically sleep if awake and idle past the timeout. True if this
        call ended the session (so the caller speaks the confirmation)."""
        with self._lock:
            if not self._active or self._now() - self._last_exchange < self._timeout_s:
                return False
            self._active = False
            return True
