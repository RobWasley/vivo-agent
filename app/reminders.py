from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app import config


DEFAULT_REMINDER_PATH = str(Path(os.environ.get("DATA_DIR", "data")) / "reminders.json")

_DEFAULT_STORE: ReminderStore | None = None
_DEFAULT_SCHEDULER: "ReminderScheduler | None" = None


def _resolve_reminder_path(path: str | None = None) -> Path:
    if path is not None:
        return Path(path)
    base = os.environ.get("DATA_DIR") or "data"
    return Path(base) / "reminders.json"


def get_default_store() -> "ReminderStore":
    global _DEFAULT_STORE
    target_path = str(_resolve_reminder_path())
    if _DEFAULT_STORE is None or str(_DEFAULT_STORE.path) != target_path:
        _DEFAULT_STORE = ReminderStore(path=target_path)
    return _DEFAULT_STORE


def get_default_scheduler(on_due: Callable[[dict[str, Any]], None] | None = None) -> "ReminderScheduler":
    global _DEFAULT_SCHEDULER
    store = get_default_store()
    if _DEFAULT_SCHEDULER is None or str(_DEFAULT_SCHEDULER.store.path) != str(store.path):
        _DEFAULT_SCHEDULER = ReminderScheduler(store=store, on_due=on_due)
    elif on_due is not None:
        _DEFAULT_SCHEDULER.on_due = on_due
    if not _DEFAULT_SCHEDULER._thread or not _DEFAULT_SCHEDULER._thread.is_alive():
        _DEFAULT_SCHEDULER.start()
    return _DEFAULT_SCHEDULER


def set_default_on_due(callback: Callable[[dict[str, Any]], None] | None) -> "ReminderScheduler":
    return get_default_scheduler(on_due=callback)


def reminder_message(text: str) -> str:
    """Friendly reminder wording that uses the configured user name when available."""
    reminder = str(text).strip()
    if not reminder:
        reminder = "your reminder"
    name = config.USER_NAME.strip()
    if name:
        return f"Hey {name}, this is your reminder to {reminder}"
    return f"This is your reminder to {reminder}"


def _coerce_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def parse_reminder(text: str, now: datetime | None = None) -> datetime:
    """Parse simple reminder expressions like 'in 5 minutes' or 'at 7:30 pm'."""
    now = now or datetime.now()
    normalized = text.strip().lower()

    m = re.search(r"\bin\s+(\d+)\s*(minute|minutes|hour|hours|day|days)\b", normalized)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(minutes=amount) if unit.startswith("minute") else timedelta(hours=amount) if unit.startswith("hour") else timedelta(days=amount)
        return now + delta

    time_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", normalized)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        meridiem = time_match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    raise ValueError(f"unsupported reminder text: {text!r}")


class ReminderStore:
    """Persistent reminder storage as a small JSON array."""

    def __init__(self, path: str | None = None):
        self.path = _resolve_reminder_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (OSError, ValueError):
            pass
        return []

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._items, f, indent=2, ensure_ascii=False)

    def add(self, text: str, due_at: str | datetime) -> dict[str, Any]:
        due = _coerce_datetime(due_at) if not isinstance(due_at, datetime) else due_at
        item = {
            "id": str(uuid4()),
            "text": text.strip(),
            "due_at": due.isoformat(timespec="seconds"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "pending",
            "fired_at": None,
        }
        self._items.append(item)
        self._save()
        return item

    def pending(self) -> list[dict[str, Any]]:
        return [it for it in self._items if it.get("status") == "pending"]

    def upcoming(self, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now()
        items = []
        for item in self.pending():
            try:
                when = _coerce_datetime(item["due_at"])
            except (TypeError, ValueError):
                continue
            if when >= now:
                items.append((when, item))
        items.sort(key=lambda pair: pair[0])
        return [item for _, item in items[:limit]]

    def due(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now()
        due_items: list[dict[str, Any]] = []
        for item in self.pending():
            try:
                when = _coerce_datetime(item["due_at"])
            except (TypeError, ValueError):
                continue
            if when <= now:
                due_items.append(item)
        return due_items

    def mark_fired(self, reminder_id: str) -> None:
        for item in self._items:
            if item.get("id") == reminder_id:
                item["status"] = "fired"
                item["fired_at"] = datetime.now().isoformat(timespec="seconds")
                self._save()
                return

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._items)


class ReminderScheduler:
    """Background scheduler that fires due reminders via a callback."""

    def __init__(
        self,
        store: ReminderStore | None = None,
        poll_interval: float = 1.0,
        on_due: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.store = store or ReminderStore()
        self.poll_interval = poll_interval
        self.on_due = on_due or (lambda reminder: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, text: str, *, in_minutes: float | None = None, at: datetime | str | None = None) -> dict[str, Any]:
        if in_minutes is not None and at is not None:
            raise ValueError("choose either in_minutes or at, not both")
        if in_minutes is not None:
            due_at = datetime.now() + timedelta(minutes=float(in_minutes))
        elif at is not None:
            due_at = _coerce_datetime(at) if isinstance(at, str) else at
        else:
            raise ValueError("reminder requires either in_minutes or at")
        return self.store.add(text, due_at)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval):
            for reminder in self.store.due():
                self.on_due(reminder)
                self.store.mark_fired(reminder["id"])


def schedule_reminder(text: str, *, minutes: float | None = None, at: str | datetime | None = None) -> dict[str, Any]:
    scheduler = get_default_scheduler()
    if minutes is not None:
        return scheduler.add(text, in_minutes=minutes)
    if at is not None:
        return scheduler.add(text, at=at)
    raise ValueError("provide either minutes or at")
