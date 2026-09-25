"""Conversation memory: named sessions with idle-time compaction (T021).

A `Conversation` is the turn history of one session: a `summary` plus a list
of turns, where a turn is (user_text, assistant_text). After each utterance
the pipeline appends a turn; when history grows past a character threshold
(or a token threshold, when one is configured), compaction summarises the
older turns into a single checkpoint via one non-streaming LLM call
(nanobot's idle-session compaction pattern, standalone). Compaction never
blocks an utterance: it snapshots under the lock, calls the LLM without the
lock, and applies the result only if nothing was appended in the meantime.

Guardrails keep the checkpoint small: every RESET_EVERY compactions the old
checkpoint is discarded and only the old turns are re-summarised, and a stored
summary is hard-capped at SUMMARY_MAX_CHARS. Completed compactions are
reported through an optional `on_compact(summary, turns_before, turns_after)`
callback so the UI can show what happened.

A `SessionStore` owns the sessions of one process: one is active, each is
persisted at <data_dir>/sessions/<id>.json with an index at
<data_dir>/sessions.json (id -> created/last_used/turns). Web connections
bind to a session (default: the active one) and may create or switch
sessions over the WS or via /api/sessions. Compaction is per-session.

Persistence: JSON, written atomically after each turn and compaction. A
corrupt or unreadable file is discarded (fresh conversation / fresh index)
so a bad file can never kill startup. A legacy single
<data_dir>/conversation.json (pre-T021) is imported as the first session
once, then removed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("vivo.conversation")

Turn = Tuple[str, str]
Summarizer = Callable[[List[dict]], str]
ChangeListener = Callable[["Conversation"], None]
# (summary, turns_before, turns_after)
CompactListener = Callable[[str, int, int], None]
# (session_id, summary, turns_before, turns_after)
CompactEvent = Callable[[str, str, int, int], None]

SUMMARY_PREFIX = "Earlier in this conversation (summary): "

# Every RESET_EVERY compactions the stored checkpoint is discarded and only
# the old turns are re-summarised, so the summary cannot grow without bound.
RESET_EVERY = 10
# Hard cap on a stored summary, even if the model ignores its max_tokens.
SUMMARY_MAX_CHARS = 500


_tiktoken_encoder = None  # lazy; None = unchecked, False = unavailable


def _count_tokens(text: str) -> int:
    """Best-effort token count: tiktoken cl100k_base when it is importable,
    otherwise ~4 chars/token. Good enough to decide when to compact; the
    local model's exact tokenizer is never required."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        try:
            import tiktoken

            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - fall back to the char estimate
            _tiktoken_encoder = False
    if _tiktoken_encoder:
        try:
            return len(_tiktoken_encoder.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


class Conversation:
    def __init__(
        self,
        data_path: Optional[str] = None,
        compact_after_chars: int = 12000,
        keep_recent_turns: int = 4,
        compact_after_tokens: int = 0,
        on_change: Optional[ChangeListener] = None,
        on_compact: Optional[CompactListener] = None,
    ):
        self.data_path = data_path
        self.compact_after_chars = compact_after_chars
        self.keep_recent_turns = keep_recent_turns
        self.compact_after_tokens = compact_after_tokens
        self.lock = threading.RLock()
        self.summary: Optional[str] = None
        self.turns: List[Turn] = []
        self._compactions = 0
        self._compacting = False
        self._on_change = on_change
        self._on_compact = on_compact
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
            try:
                self._compactions = max(0, int(raw.get("compactions") or 0))
            except (TypeError, ValueError):
                self._compactions = 0
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
                {
                    "summary": self.summary,
                    "turns": self.turns,
                    "compactions": self._compactions,
                },
                f,
                ensure_ascii=False,
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
            self._notify_change()

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

    def size_tokens(self) -> int:
        with self.lock:
            total = _count_tokens(self.summary or "")
            for user_text, assistant_text in self.turns:
                total += _count_tokens(user_text) + _count_tokens(assistant_text)
            return total

    def needs_compaction(self) -> bool:
        with self.lock:
            if len(self.turns) <= self.keep_recent_turns:
                return False
            if self.size_chars() > self.compact_after_chars:
                return True
            return (
                self.compact_after_tokens > 0
                and self.size_tokens() > self.compact_after_tokens
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
            # periodic checkpoint reset: on every RESET_EVERY-th compaction the
            # previous summary is dropped so the checkpoint cannot keep growing
            reset_checkpoint = (
                self._compactions > 0
                and (self._compactions + 1) % RESET_EVERY == 0
            )
            self._compacting = True
        try:
            messages: List[dict] = []
            if old_summary and not reset_checkpoint:
                messages.append(
                    {"role": "user", "content": f"Previous summary: {old_summary}"}
                )
            for user_text, assistant_text in old:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})
            new_summary = (summarize(messages) or "").strip()[:SUMMARY_MAX_CHARS]
            if not new_summary:
                return False
            with self.lock:
                if len(self.turns) != snapshot_len:
                    log.info("skip compaction: history grew while summarising")
                    return False
                self.summary = new_summary
                self.turns = recent
                self._compactions += 1
                self._save()
                log.info(
                    "compacted %d turns -> %d-char summary, kept %d turns"
                    " (checkpoint #%d%s)",
                    len(old),
                    len(new_summary),
                    len(recent),
                    self._compactions,
                    ", reset" if reset_checkpoint else "",
                )
                self._notify_compact(new_summary, snapshot_len, len(recent))
                self._notify_change()
                return True
        finally:
            with self.lock:
                self._compacting = False

    def maybe_compact(self, summarize: Summarizer) -> None:
        """Fire compaction in a background thread (no-op if not needed)."""
        threading.Thread(
            target=self.compact, args=(summarize,), daemon=True
        ).start()

    def _notify_compact(
        self, summary: str, turns_before: int, turns_after: int
    ) -> None:
        """Tell the store a compaction completed (while holding the lock). A
        callback error must never break the caller."""
        if self._on_compact is None:
            return
        try:
            self._on_compact(summary, turns_before, turns_after)
        except Exception:
            log.exception("compaction callback failed")

    def _notify_change(self) -> None:
        """Call the store callback (while holding the lock) so the session
        index stays fresh. A callback error must never break the caller."""
        if self._on_change is None:
            return
        try:
            self._on_change(self)
        except Exception:
            log.exception("conversation change callback failed")


class SessionStore:
    """Named conversation sessions for one process (T021).

    Exactly one session is active at a time. Each session is a Conversation
    persisted at <data_dir>/sessions/<id>.json; <data_dir>/sessions.json is
    the index {active, sessions: {id: {created, last_used, turns}}}. With
    data_dir=None the store is fully in-memory (tests).

    A legacy single <data_dir>/conversation.json (pre-T021) is imported as a
    session once, then removed; orphan sessions/*.json files not in the index
    are adopted, so history is not lost across upgrades or a lost index.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        compact_after_chars: int = 12000,
        keep_recent_turns: int = 4,
        compact_after_tokens: int = 0,
        on_compact: Optional[CompactEvent] = None,
    ):
        self.data_dir = data_dir
        self.compact_after_chars = compact_after_chars
        self.keep_recent_turns = keep_recent_turns
        self.compact_after_tokens = compact_after_tokens
        self._on_compact = on_compact
        self.lock = threading.RLock()
        self._conversations: Dict[str, Conversation] = {}
        self._retired: set = set()  # deleted ids: never reused, even in the same second
        self._index = {"active": None, "sessions": {}}
        if data_dir:
            os.makedirs(os.path.join(data_dir, "sessions"), exist_ok=True)
            self._load_index()
            self._adopt_orphans()
            self._migrate_legacy()
        with self.lock:
            if not self._index["sessions"]:
                self.create(activate=False)
            if self._index["active"] not in self._index["sessions"]:
                self._index["active"] = self._newest_id_locked()
                self._save_index()

    # -- paths -------------------------------------------------------------
    @property
    def index_path(self) -> Optional[str]:
        return os.path.join(self.data_dir, "sessions.json") if self.data_dir else None

    @property
    def sessions_dir(self) -> Optional[str]:
        return os.path.join(self.data_dir, "sessions") if self.data_dir else None

    def _session_path(self, session_id: str) -> Optional[str]:
        return (
            os.path.join(self.sessions_dir, f"{session_id}.json")
            if self.sessions_dir
            else None
        )

    # -- public API ---------------------------------------------------------
    @property
    def active_id(self) -> str:
        with self.lock:
            return self._index["active"]

    def ids(self) -> set:
        with self.lock:
            return set(self._index["sessions"])

    def list_sessions(self) -> List[dict]:
        """Session metadata, most recently used first."""
        with self.lock:
            rows = [
                {
                    "id": sid,
                    "name": e.get("name") or "",
                    "created": e["created"],
                    "last_used": e["last_used"],
                    "turns": e["turns"],
                    "active": sid == self._index["active"],
                }
                for sid, e in self._index["sessions"].items()
            ]
        rows.sort(key=lambda r: r["last_used"], reverse=True)
        return rows

    def conversation_for(self, session_id: str) -> Conversation:
        """Get (loading on demand) a session's Conversation. KeyError if unknown."""
        with self.lock:
            conv = self._conversations.get(session_id)
            if conv is None:
                if session_id not in self._index["sessions"]:
                    raise KeyError(session_id)
                conv = self._new_conversation(session_id)
                self._conversations[session_id] = conv
            return conv

    def create(self, activate: bool = True) -> str:
        with self.lock:
            session_id = self._next_id()
            conv = self._new_conversation(session_id)
            self._conversations[session_id] = conv
            now = time.time()
            self._index["sessions"][session_id] = {
                "name": "",
                "created": now,
                "last_used": now,
                "turns": 0,
            }
            if activate:
                self._index["active"] = session_id
            self._save_index()
        log.info("created session %s", session_id, extra={"event": "session_create"})
        return session_id

    def rename(self, session_id: str, name: str) -> None:
        """Set a human-friendly display name for a session (empty clears it)."""
        with self.lock:
            entry = self._index["sessions"].get(session_id)
            if entry is None:
                raise KeyError(session_id)
            entry["name"] = (name or "").strip()
            self._save_index()

    def set_active(self, session_id: str) -> None:
        with self.lock:
            if session_id not in self._index["sessions"]:
                raise KeyError(session_id)
            self._index["active"] = session_id
            self._save_index()

    def delete(self, session_id: str) -> None:
        """Remove a session. If it was active, the most recently used
        remaining session becomes active, or a fresh one is created."""
        with self.lock:
            if session_id not in self._index["sessions"]:
                raise KeyError(session_id)
            del self._index["sessions"][session_id]
            self._retired.add(session_id)
            self._conversations.pop(session_id, None)
            path = self._session_path(session_id)
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if self._index["active"] == session_id:
                if self._index["sessions"]:
                    self._index["active"] = self._newest_id_locked()
                else:
                    self._index["active"] = None
            self._save_index()
            if self._index["active"] is None:
                self._index["active"] = self.create(activate=False)
                self._save_index()
        log.info("deleted session %s", session_id, extra={"event": "session_delete"})

    def set_limits(
        self,
        compact_after_chars: int,
        keep_recent_turns: int,
        compact_after_tokens: int = 0,
    ) -> None:
        """Hot-apply compaction limits to the store and all loaded sessions."""
        with self.lock:
            self.compact_after_chars = compact_after_chars
            self.keep_recent_turns = keep_recent_turns
            self.compact_after_tokens = compact_after_tokens
            for conv in self._conversations.values():
                conv.compact_after_chars = compact_after_chars
                conv.keep_recent_turns = keep_recent_turns
                conv.compact_after_tokens = compact_after_tokens

    # -- internals -----------------------------------------------------------
    def _newest_id_locked(self) -> str:
        return max(
            self._index["sessions"],
            key=lambda i: self._index["sessions"][i]["last_used"],
        )

    def _next_id(self) -> str:
        base = time.strftime("%Y%m%d-%H%M%S")
        candidate, n = base, 2
        while candidate in self._index["sessions"] or candidate in self._retired:
            candidate = f"{base}-{n}"
            n += 1
        return candidate

    def _new_conversation(self, session_id: str) -> Conversation:
        conv = Conversation(
            data_path=self._session_path(session_id),
            compact_after_chars=self.compact_after_chars,
            keep_recent_turns=self.keep_recent_turns,
            compact_after_tokens=self.compact_after_tokens,
            on_change=lambda c, sid=session_id: self._note_change(sid, c),
            on_compact=lambda s, b, a, sid=session_id: self._note_compact(
                sid, s, b, a
            ),
        )
        conv._save()
        return conv

    def _note_change(self, session_id: str, conv: Conversation) -> None:
        with self.lock:
            entry = self._index["sessions"].get(session_id)
            if entry is None:
                return
            entry["last_used"] = time.time()
            entry["turns"] = len(conv.turns)
            self._save_index()

    def _note_compact(
        self, session_id: str, summary: str, turns_before: int, turns_after: int
    ) -> None:
        """Forward a completed compaction to the store callback (pipeline WS
        broadcast). A callback error must never break the caller."""
        if self._on_compact is None:
            return
        try:
            self._on_compact(session_id, summary, turns_before, turns_after)
        except Exception:
            log.exception("session compaction callback failed")

    def _load_index(self) -> None:
        if not self.index_path:
            return
        try:
            with open(self.index_path, encoding="utf-8") as f:
                raw = json.load(f)
            sessions = raw.get("sessions")
            if not isinstance(sessions, dict):
                raise ValueError("sessions is not a dict")
            clean = {}
            for sid, entry in sessions.items():
                if not isinstance(sid, str) or not isinstance(entry, dict):
                    continue
                clean[sid] = {
                    "name": str(entry.get("name") or ""),
                    "created": float(entry.get("created") or 0.0),
                    "last_used": float(entry.get("last_used") or 0.0),
                    "turns": int(entry.get("turns") or 0),
                }
            active = raw.get("active")
            self._index = {
                "active": active if isinstance(active, str) else None,
                "sessions": clean,
            }
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - a bad file must not kill startup
            log.exception("discarding unreadable session index %s", self.index_path)
            self._index = {"active": None, "sessions": {}}

    def _save_index(self) -> None:
        if not self.data_dir:
            return
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False)
        os.replace(tmp, self.index_path)

    def _adopt_orphans(self) -> None:
        if not self.sessions_dir:
            return
        for name in sorted(os.listdir(self.sessions_dir)):
            if not name.endswith(".json"):
                continue
            session_id = name[: -len(".json")]
            if session_id in self._index["sessions"]:
                continue
            path = os.path.join(self.sessions_dir, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            turns = 0
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                t = raw.get("turns")
                if isinstance(t, list):
                    turns = len(t)
            except Exception:
                pass
            self._index["sessions"][session_id] = {
                "name": "",
                "created": mtime,
                "last_used": mtime,
                "turns": turns,
            }
            log.info("adopted orphan session file %s (%d turns)", name, turns)

    def _migrate_legacy(self) -> None:
        legacy = os.path.join(self.data_dir, "conversation.json")
        if not os.path.exists(legacy):
            return
        old = Conversation(data_path=legacy)
        if old.turns or old.summary:
            session_id = self._next_id()
            conv = self._new_conversation(session_id)
            self._conversations[session_id] = conv
            conv.summary = old.summary
            conv.turns = list(old.turns)
            conv._save()
            now = time.time()
            self._index["sessions"][session_id] = {
                "name": "",
                "created": now,
                "last_used": now,
                "turns": len(old.turns),
            }
            self._index["active"] = session_id
            self._save_index()
            log.info(
                "migrated legacy conversation.json -> session %s (%d turns)",
                session_id,
                len(old.turns),
            )
        try:
            os.remove(legacy)
        except OSError:
            pass
