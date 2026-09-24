from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app import config

log = logging.getLogger("vivo.reminders")

DEFAULT_REMINDER_PATH = str(Path(os.environ.get("DATA_DIR", "data")) / "reminders.json")

TYPES = ("reminder", "task")
RESPONSES = ("spoken", "text")
REPEATS = ("once", "interval", "daily", "weekly")
MAX_INTERVAL_MINUTES = 24 * 60

# Legacy rows used a single `kind` field; map it onto (type, response).
LEGACY_KINDS = {
    "spoken": ("reminder", "spoken"),
    "silent": ("reminder", "text"),
    "system": ("task", "text"),
}

_DEFAULT_STORE: "ReminderStore | None" = None
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


def user_tz() -> tzinfo:
    """The user's configured IANA time zone (T023), or the server's local
    zone when none is set or the name is unknown. Schedules are written in
    the user's wall clock and stored in UTC, so a 09:00 daily item fires at
    09:00 *for the user* regardless of DST."""
    name = config.USER_TIMEZONE.strip()
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            log.warning("unknown user time zone %r; falling back to the server's zone", name)
    return datetime.now().astimezone().tzinfo


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}", text):
        text += ":00"
    return datetime.fromisoformat(text)


def _as_utc(dt: datetime, assume: str) -> datetime:
    """Make a datetime aware UTC. `assume` says what a naive value means:
    "local" = the user's wall clock (what they asked for), "utc" = absolute
    server time. Aware values are converted either way."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=user_tz() if assume == "local" else timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_at(value: str | datetime) -> datetime:
    """A one-time fire time from user input; a naive value is the user's local
    wall clock. Returns aware UTC."""
    return _as_utc(_coerce_datetime(value), "local")


def _parse_hhmm(value: object) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value or ""))
    if not m:
        raise ValueError("time must look like HH:MM")
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        raise ValueError("time must be between 00:00 and 23:59")
    return hour, minute


def _parse_weekdays(value: object) -> list[int]:
    days = [int(d) for d in (value or [])]
    if not days:
        raise ValueError("a weekly reminder needs at least one weekday")
    if any(d < 0 or d > 6 for d in days):
        raise ValueError("weekdays are 0=Monday .. 6=Sunday")
    return sorted(set(days))


def _parse_interval(value: object) -> int:
    try:
        minutes = int(value)
    except (TypeError, ValueError):
        raise ValueError("an interval reminder needs a whole number of minutes")
    if not 1 <= minutes <= MAX_INTERVAL_MINUTES:
        raise ValueError(f"interval must be between 1 minute and {MAX_INTERVAL_MINUTES // 60} hours")
    return minutes


def next_due(
    repeat: str,
    *,
    at: str | datetime | None = None,
    time: str | None = None,
    days: list[int] | None = None,
    every: int | None = None,
    now: datetime | None = None,
) -> datetime:
    """Next fire time, as an aware UTC datetime. `repeat` is
    once/interval/daily/weekly; interval `every` is a free-running span in
    minutes (next fire is `now` + span, so items roll over from their last
    fire), weekly `days` use 0=Monday .. 6=Sunday, matching Python's
    weekday(). Wall-clock inputs (`at`, `time`) are read in the user's
    configured time zone; a naive `now` is server (UTC) time."""
    now = now or utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if repeat == "once":
        if at is None:
            raise ValueError("a one-time reminder needs a date and time")
        when = _coerce_at(at)
        if when <= now:
            raise ValueError("a one-time reminder must be in the future")
        return when
    if repeat == "interval":
        return now + timedelta(minutes=_parse_interval(every))
    if repeat not in ("daily", "weekly"):
        raise ValueError(f"unknown repeat kind: {repeat!r}")
    hour, minute = _parse_hhmm(time)
    local_now = now.astimezone(user_tz())
    if repeat == "daily":
        candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= local_now:
            candidate += timedelta(days=1)
        return candidate.astimezone(timezone.utc)
    day_set = set(_parse_weekdays(days))
    for offset in range(8):
        day = local_now + timedelta(days=offset)
        if day.weekday() in day_set:
            candidate = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate > local_now:
                return candidate.astimezone(timezone.utc)
    raise ValueError("no matching weekday")  # unreachable: day_set is non-empty


def parse_reminder(text: str, now: datetime | None = None) -> datetime:
    """Parse simple reminder expressions like 'in 5 minutes' or 'at 7:30 pm'.
    A wall-clock time is read in the user's time zone; returns aware UTC."""
    now = now or utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(user_tz())
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
        candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= local_now:
            candidate += timedelta(days=1)
        return candidate.astimezone(timezone.utc)

    raise ValueError(f"unsupported reminder text: {text!r}")


def _is_aware(value: object) -> bool:
    try:
        return _coerce_datetime(value).tzinfo is not None
    except (TypeError, ValueError):
        return False


def _shape_from(item: dict) -> tuple[str, str]:
    """(type, response) for a row, mapping the legacy `kind` field when the
    row predates the split."""
    kind = str(item.get("kind") or "").strip().lower()
    legacy = LEGACY_KINDS.get(kind)
    item_type = str(item.get("type") or (legacy[0] if legacy else "reminder"))
    if item_type not in TYPES:
        item_type = legacy[0] if legacy else "reminder"
    response = str(item.get("response") or (legacy[1] if legacy else "spoken"))
    if response not in RESPONSES:
        response = legacy[1] if legacy else "spoken"
    return item_type, response


def _normalize_datetime(value: object, assume: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        return _as_utc(_coerce_datetime(value), assume).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def _normalize(item: dict) -> dict[str, Any]:
    """One stored reminder/task with every field present (legacy rows upgraded).
    Datetimes are canonical aware UTC. Naive rows predate the time-zone fix and
    hold the server's (UTC) wall clock, so they are read as absolute UTC —
    except legacy repeating items, whose intended user wall clock survives in
    `time`/`days` and is re-anchored in the user's zone."""
    item_type, response = _shape_from(item)
    repeat = str(item.get("repeat") or "once")
    due_at = _normalize_datetime(item["due_at"], "utc")
    if repeat in ("daily", "weekly") and not _is_aware(item.get("due_at")):
        try:
            due_at = next_due(repeat, time=item.get("time"), days=item.get("days")).isoformat(timespec="seconds")
        except ValueError:
            pass  # unparseable `time`: keep the absolute interpretation
    return {
        "id": str(item["id"]),
        "text": str(item["text"]),
        "type": item_type,
        "response": response,
        "repeat": repeat,
        "at": _normalize_datetime(item.get("at"), "utc"),
        "time": item.get("time"),
        "days": list(item.get("days") or []),
        "every": item.get("every"),
        "session_id": item.get("session_id") or None,
        "due_at": due_at,
        "created_at": _normalize_datetime(item.get("created_at"), "utc") or "",
        "status": item.get("status") or "pending",
        "fired_at": _normalize_datetime(item.get("fired_at"), "utc"),
        "last_fired_at": _normalize_datetime(item.get("last_fired_at"), "utc"),
        "last_response": item.get("last_response"),
    }


class ReminderStore:
    """Persistent reminder/task storage as a small JSON array.

    Each item: id, text, type (reminder|task), response (spoken|text), repeat
    (once|interval|daily|weekly), at (aware-UTC iso datetime, once), every
    (minutes, interval), time (HH:MM in the user's time zone, daily/weekly),
    days (0=Mon..6=Sun, weekly), session_id
    (optional conversation to attach to), due_at (next fire time, aware UTC),
    created_at, status (pending|fired), fired_at, last_fired_at,
    last_response (what a task last produced).
    """

    def __init__(self, path: str | None = None):
        self.path = _resolve_reminder_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._items: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [
                    _normalize(item)
                    for item in data
                    if isinstance(item, dict) and all(k in item for k in ("id", "text", "due_at"))
                ]
        except (OSError, ValueError):
            pass
        return []

    def _save(self) -> None:
        with self._lock:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self._items, f, indent=2, ensure_ascii=False)

    # -- creation ----------------------------------------------------------
    def add(self, text: str, due_at: str | datetime) -> dict[str, Any]:
        """Legacy one-time spoken reminder (tools, old tests). `due_at` is an
        absolute fire time; a naive datetime is server (UTC) time."""
        at = _as_utc(_coerce_datetime(due_at), "utc")
        return self._new(text, "reminder", "spoken", "once", at=at.isoformat(timespec="seconds"))

    def _new(self, text: str, item_type: str, response: str, repeat: str, *, at=None, time=None, days=None, every=None, session_id=None) -> dict[str, Any]:
        now = utcnow()
        due = next_due(repeat, at=at, time=time, days=days, every=every, now=now)
        item = {
            "id": str(uuid4()),
            "text": text.strip(),
            "type": item_type,
            "response": response,
            "repeat": repeat,
            "at": at if repeat == "once" else None,
            "time": str(time) if repeat in ("daily", "weekly") and time else None,
            "days": sorted(set(days or [])) if repeat == "weekly" else [],
            "every": every if repeat == "interval" else None,
            "session_id": session_id,
            "due_at": due.isoformat(timespec="seconds"),
            "created_at": now.isoformat(timespec="seconds"),
            "status": "pending",
            "fired_at": None,
            "last_fired_at": None,
            "last_response": None,
        }
        self._items.append(item)
        self._save()
        return item

    def create(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Validate a payload from the API/UI and add the reminder."""
        with self._lock:
            text = str(spec.get("text") or "").strip()
            if not text:
                raise ValueError("reminder text is required")
            item_type, response = self._spec_shape(spec, "reminder", "spoken")
            repeat = str(spec.get("repeat") or "once")
            if repeat not in REPEATS:
                raise ValueError(f"repeat must be one of: {', '.join(REPEATS)}")
            session_id = str(spec.get("session_id") or "").strip() or None
            every_minutes = None
            at_iso = None
            if repeat == "once":
                if not spec.get("at"):
                    raise ValueError("a one-time reminder needs a date and time")
                at_iso = _coerce_at(spec["at"]).isoformat(timespec="seconds")
                next_due("once", at=at_iso)
            elif repeat == "interval":
                every_minutes = _parse_interval(spec.get("every"))
            else:
                hour, minute = _parse_hhmm(spec.get("time"))
            return self._new(
                text, item_type, response, repeat,
                at=at_iso,
                time=f"{hour:02d}:{minute:02d}" if repeat in ("daily", "weekly") else None,
                days=spec.get("days") if repeat == "weekly" else None,
                every=every_minutes,
                session_id=session_id,
            )

    @staticmethod
    def _spec_shape(spec: dict[str, Any], default_type: str, default_response: str) -> tuple[str, str]:
        """(type, response) from a payload, tolerating the legacy `kind` key."""
        kind = str(spec.get("kind") or "").strip().lower()
        if kind and kind not in LEGACY_KINDS:
            raise ValueError(f"unknown kind {kind!r} (legacy kinds: {', '.join(sorted(LEGACY_KINDS))})")
        legacy = LEGACY_KINDS.get(kind)
        item_type = str(spec.get("type") or (legacy[0] if legacy else default_type))
        if item_type not in TYPES:
            raise ValueError(f"type must be one of: {', '.join(TYPES)}")
        response = str(spec.get("response") or (legacy[1] if legacy else default_response))
        if response not in RESPONSES:
            raise ValueError(f"response must be one of: {', '.join(RESPONSES)}")
        return item_type, response

    # -- lookup / mutation ---------------------------------------------------
    def get(self, reminder_id: str) -> dict[str, Any]:
        with self._lock:
            for item in self._items:
                if item["id"] == reminder_id:
                    return dict(item)
            raise KeyError(reminder_id)

    def update(self, reminder_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Edit fields and re-derive the next due time; re-arms a fired item."""
        with self._lock:
            item = None
            for candidate in self._items:
                if candidate["id"] == reminder_id:
                    item = candidate
                    break
            if item is None:
                raise KeyError(reminder_id)
            text = str(spec.get("text", item["text"]) or "").strip()
            if not text:
                raise ValueError("reminder text is required")
            item_type, response = self._spec_shape(
                spec, item["type"], item["response"]
            )
            repeat = str(spec.get("repeat") or item["repeat"])
            if repeat not in REPEATS:
                raise ValueError(f"repeat must be one of: {', '.join(REPEATS)}")
            session_id = spec.get("session_id", item["session_id"])
            session_id = str(session_id or "").strip() or None
            if repeat == "once":
                at = spec.get("at") or item.get("at")
                if not at:
                    raise ValueError("a one-time reminder needs a date and time")
                # A value in the payload is user wall clock; the stored one is
                # already canonical UTC.
                item["at"] = _as_utc(_coerce_datetime(at), "local" if spec.get("at") else "utc").isoformat(timespec="seconds")
                item["time"] = None
                item["days"] = []
                item["every"] = None
            elif repeat == "interval":
                item["every"] = _parse_interval(spec.get("every") or item.get("every"))
                item["at"] = None
                item["time"] = None
                item["days"] = []
            else:
                hour, minute = _parse_hhmm(spec.get("time") or item.get("time"))
                time_hhmm = f"{hour:02d}:{minute:02d}"
                item["time"] = time_hhmm
                item["at"] = None
                item["every"] = None
                if repeat == "weekly":
                    item["days"] = _parse_weekdays(spec.get("days") or item.get("days"))
                else:
                    item["days"] = []
            item["text"] = text
            item["type"] = item_type
            item["response"] = response
            item["repeat"] = repeat
            item["session_id"] = session_id
            item["due_at"] = next_due(
                repeat, at=item.get("at"), time=item.get("time"),
                days=item.get("days"), every=item.get("every"),
            ).isoformat(timespec="seconds")
            item["status"] = "pending"
            item["fired_at"] = None
            self._save()
            return dict(item)

    def delete(self, reminder_id: str) -> None:
        with self._lock:
            remaining = [item for item in self._items if item["id"] != reminder_id]
            if len(remaining) == len(self._items):
                raise KeyError(reminder_id)
            self._items = remaining
            self._save()

    def list_all(self) -> list[dict[str, Any]]:
        """Pending first (soonest due), then fired (most recently fired)."""
        with self._lock:
            pending = sorted(
                (dict(i) for i in self._items if i["status"] == "pending"),
                key=lambda i: i["due_at"],
            )
            fired = sorted(
                (dict(i) for i in self._items if i["status"] == "fired"),
                key=lambda i: i.get("fired_at") or "",
                reverse=True,
            )
            return pending + fired

    # -- scheduler support ----------------------------------------------------
    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(i) for i in self._items if i.get("status") == "pending"]

    def upcoming(self, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        with self._lock:
            items = []
            for item in self._items:
                if item.get("status") != "pending":
                    continue
                try:
                    when = _as_utc(_coerce_datetime(item["due_at"]), "local")
                except (TypeError, ValueError):
                    continue
                if when >= now:
                    items.append((when, item))
            items.sort(key=lambda pair: pair[0])
            return [dict(item) for _, item in items[:limit]]

    def due(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        with self._lock:
            due_items: list[dict[str, Any]] = []
            for item in self._items:
                if item.get("status") != "pending":
                    continue
                try:
                    when = _as_utc(_coerce_datetime(item["due_at"]), "local")
                except (TypeError, ValueError):
                    continue
                if when <= now:
                    due_items.append(dict(item))
            return due_items

    def mark_fired(self, reminder_id: str, now: datetime | None = None) -> None:
        now = now or utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        with self._lock:
            for item in self._items:
                if item.get("id") == reminder_id:
                    item["status"] = "fired"
                    item["fired_at"] = now.isoformat(timespec="seconds")
                    self._save()
                    return

    def advance(self, reminder_id: str, now: datetime | None = None) -> None:
        """After a fire: one-time items are done; repeating ones roll to
        their next occurrence and stay pending."""
        now = now or utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        with self._lock:
            item = None
            for candidate in self._items:
                if candidate.get("id") == reminder_id:
                    item = candidate
                    break
            if item is None:
                return
            if item.get("repeat") == "once":
                item["status"] = "fired"
                item["fired_at"] = now.isoformat(timespec="seconds")
            else:
                item["last_fired_at"] = now.isoformat(timespec="seconds")
                item["due_at"] = next_due(
                    item["repeat"], at=item.get("at"), time=item.get("time"),
                    days=item.get("days"), every=item.get("every"), now=now,
                ).isoformat(timespec="seconds")
            self._save()

    def set_last_response(self, reminder_id: str, text: str | None) -> dict[str, Any]:
        """Record what a task last produced, without touching the schedule or
        status (unlike update(), which re-arms the item)."""
        with self._lock:
            for item in self._items:
                if item.get("id") == reminder_id:
                    response = str(text or "").strip()
                    item["last_response"] = response[:1000] or None
                    self._save()
                    return dict(item)
            raise KeyError(reminder_id)


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
            due_at = utcnow() + timedelta(minutes=float(in_minutes))
        elif at is not None:
            # A string is user wall clock; a datetime is absolute (naive = UTC).
            due_at = _coerce_at(at) if isinstance(at, str) else _as_utc(_coerce_datetime(at), "utc")
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
                try:
                    self.on_due(reminder)
                except Exception:  # noqa: BLE001 - one bad callback must not kill the loop
                    log.exception("reminder callback failed for %r", reminder.get("text"))
                self.store.advance(reminder["id"])


def schedule_reminder(text: str, *, minutes: float | None = None, at: str | datetime | None = None) -> dict[str, Any]:
    scheduler = get_default_scheduler()
    if minutes is not None:
        return scheduler.add(text, in_minutes=minutes)
    if at is not None:
        return scheduler.add(text, at=at)
    raise ValueError("provide either minutes or at")
