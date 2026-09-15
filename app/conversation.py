"""Conversation memory: persistent turn history with idle-time compaction.

One shared conversation per process (one user at a time). History is a
`summary` plus a list of turns, where a turn is (user_text, assistant_text).
After each utterance the pipeline appends a turn; when history grows past a
character threshold, compaction summarises the older turns into a single
checkpoint via one non-streaming LLM call (nanobot's idle-session compaction
pattern, standalone). Compaction never blocks an utterance: it snapshots
under the lock, calls the LLM without the lock, and applies the result only
if nothing was appended in the meantime.

Persistence: JSON at <data_dir>/conversation.json, written atomically after
each turn and compaction. A corrupt or unreadable file is discarded (fresh
conversation) so a bad file can never kill startup.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Callable, List, Optional, Tuple

log = logging.getLogger("vivo.conversation")

Turn = Tuple[str, str]
Summarizer = Callable[[List[dict]], str]

SUMMARY_PREFIX = "Earlier in this conversation (summary): "


class Conversation:
    def __init__(
        self,
        data_path: Optional[str] = None,
        compact_after_chars: int = 12000,
        keep_recent_turns: int = 4,
    ):
        self.data_path = data_path
        self.compact_after_chars = compact_after_chars
        self.keep_recent_turns = keep_recent_turns
        self.lock = threading.RLock()
        self.summary: Optional[str] = None
        self.turns: List[Turn] = []
        self._compacting = False
        if data_path:
            self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.data_path, encoding="utf-8") as f:
                raw = json.load(f)
            turns = raw.get("turns")
            if not isinstance(turns, list):
                raise ValueError("turns is not a list")
            clean: List[Turn] = []
            for t in turns:
                if (
                    isinstance(t, list)
                    and len(t) == 2
                    and all(isinstance(x, str) for x in t)
                ):
                    clean.append((t[0], t[1]))
            summary = raw.get("summary")
            self.summary = summary if isinstance(summary, str) and summary else None
            self.turns = clean
            log.info(
                "loaded conversation: %d turns, summary=%s",
                len(clean),
                "yes" if self.summary else "no",
            )
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - a bad file must not kill startup
            log.exception("discarding unreadable conversation file %s", self.data_path)
            self.summary = None
            self.turns = []

    def _save(self) -> None:
        if not self.data_path:
            return
        tmp = self.data_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": self.summary, "turns": self.turns}, f, ensure_ascii=False
            )
        os.replace(tmp, self.data_path)

    # -- turns ------------------------------------------------------------
    def add_turn(self, user_text: str, assistant_text: str) -> None:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text or not assistant_text:
            return
        with self.lock:
            self.turns.append((user_text, assistant_text))
            self._save()

    def messages(self) -> List[dict]:
        """OpenAI-style messages: optional summary + flattened turns."""
        with self.lock:
            out: List[dict] = []
            if self.summary:
                out.append(
                    {"role": "system", "content": SUMMARY_PREFIX + self.summary}
                )
            for user_text, assistant_text in self.turns:
                out.append({"role": "user", "content": user_text})
                out.append({"role": "assistant", "content": assistant_text})
            return out

    # -- compaction ----------------------------------------------------------
    def size_chars(self) -> int:
        with self.lock:
            total = len(self.summary or "")
            for user_text, assistant_text in self.turns:
                total += len(user_text) + len(assistant_text)
            return total

    def needs_compaction(self) -> bool:
        with self.lock:
            return (
                self.size_chars() > self.compact_after_chars
                and len(self.turns) > self.keep_recent_turns
            )

    def compact(self, summarize: Summarizer) -> bool:
        """Summarise old turns into a checkpoint. The LLM call runs without
        holding the lock; the result applies only if nothing was appended in
        the meantime (otherwise it retries on the next idle cycle). Returns
        True if compaction happened."""
        with self.lock:
            if self._compacting or not self.needs_compaction():
                return False
            old = self.turns[:- self.keep_recent_turns]
            recent = self.turns[-self.keep_recent_turns:]
            old_summary = self.summary
            snapshot_len = len(self.turns)
            self._compacting = True
        try:
            messages: List[dict] = []
            if old_summary:
                messages.append(
                    {"role": "user", "content": f"Previous summary: {old_summary}"}
                )
            for user_text, assistant_text in old:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})
            new_summary = (summarize(messages) or "").strip()
            if not new_summary:
                return False
            with self.lock:
                if len(self.turns) != snapshot_len:
                    log.info("skip compaction: history grew while summarising")
                    return False
                self.summary = new_summary
                self.turns = recent
                self._save()
                log.info(
                    "compacted %d turns -> %d-char summary, kept %d turns",
                    len(old),
                    len(new_summary),
                    len(recent),
                )
                return True
        finally:
            with self.lock:
                self._compacting = False

    def maybe_compact(self, summarize: Summarizer) -> None:
        """Fire compaction in a background thread (no-op if not needed)."""
        threading.Thread(
            target=self.compact, args=(summarize,), daemon=True
        ).start()
