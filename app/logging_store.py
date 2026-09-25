"""System log store: a rolling buffer of vivo log events.

The pipeline, agent and API modules already log through the ``vivo.*`` loggers;
``VivoLogHandler`` bridges those records into a ``LogStore``, which keeps the
last MAX_ENTRIES events in memory and persists them (atomically) to
``<data dir>/log.json``.  The web console lists the buffer over
``GET /api/logs`` and receives new entries live over the WebSocket
(``{"type": "log_entry", ...}``).  A corrupt file is discarded, like the rest
of vivo's persistence.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Optional

MAX_ENTRIES = 1000
# one disk write per this many seconds; entries arriving sooner are batched
_SAVE_INTERVAL_S = 1.0
# keep stored messages short enough to stay readable in the console
_MAX_MESSAGE_CHARS = 500


class LogStore:
    """Thread-safe rolling buffer of log entries backed by a JSON file."""

    def __init__(
        self,
        path: Optional[str] = None,
        max_entries: int = MAX_ENTRIES,
        on_entry: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.path = path
        self.max_entries = max_entries
        self.on_entry = on_entry
        self.lock = threading.RLock()
        self.entries: deque[dict] = deque()
        self._last_save = 0.0
        if path:
            self._load()

    def add(
        self,
        level: str,
        message: str,
        event: Optional[str] = None,
        **fields,
    ) -> dict:
        """Append one entry, return it, and notify the (WS) callback."""
        message = str(message).replace("\r\n", "\n").replace("\r", "\n")
        entry = {
            "ts": time.time(),
            "level": str(level).upper(),
            "message": message[:_MAX_MESSAGE_CHARS],
        }
        if event:
            entry["event"] = str(event)
        for key, value in fields.items():
            if value is not None:
                entry[key] = value
        with self.lock:
            self.entries.append(entry)
            while len(self.entries) > self.max_entries:
                self.entries.popleft()
            self._maybe_save()
        if self.on_entry is not None:
            try:
                self.on_entry(entry)
            except Exception:
                pass  # a broken console must not break logging
        return entry

    def list(self, level: Optional[str] = None) -> list[dict]:
        """Return the buffer, newest last, optionally filtered by level."""
        with self.lock:
            entries = list(self.entries)
        if level:
            wanted = str(level).upper()
            entries = [e for e in entries if e["level"] == wanted]
        return entries

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self._save()

    def flush(self) -> None:
        """Force any pending entries to disk (called on shutdown)."""
        with self.lock:
            self._save()

    # -- persistence ---------------------------------------------------------
    def _maybe_save(self) -> None:
        now = time.time()
        if now - self._last_save >= _SAVE_INTERVAL_S:
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        self._last_save = time.time()
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"entries": list(self.entries)}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass  # log persistence must never take the assistant down

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        rows = raw.get("entries") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return
        clean = [
            item
            for item in rows
            if isinstance(item, dict)
            and isinstance(item.get("ts"), (int, float))
            and isinstance(item.get("message"), str)
        ]
        with self.lock:
            self.entries.extend(clean[-self.max_entries:])


class VivoLogHandler(logging.Handler):
    """Route ``vivo.*`` log records into a LogStore (and on to the console)."""

    def __init__(self, store: LogStore) -> None:
        super().__init__()
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.store.add(
                record.levelname,
                record.getMessage(),
                event=getattr(record, "event", None),
            )
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)
