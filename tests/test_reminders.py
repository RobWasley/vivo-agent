from __future__ import annotations

import time
from datetime import datetime, timedelta

from app.reminders import ReminderScheduler, ReminderStore, parse_reminder


def test_parse_relative_reminder():
    now = datetime(2026, 9, 18, 12, 0, 0)
    due = parse_reminder("in 5 minutes", now=now)
    assert due == now + timedelta(minutes=5)


def test_parse_time_reminder():
    now = datetime(2026, 9, 18, 11, 30, 0)
    due = parse_reminder("at 12:15", now=now)
    assert due == datetime(2026, 9, 18, 12, 15, 0)


def test_reminder_scheduler_fires_due(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    fired = []
    scheduler = ReminderScheduler(store=store, poll_interval=0.01, on_due=lambda reminder: fired.append(reminder["text"]))
    scheduler.add("check the build", in_minutes=0.02)
    scheduler.start()
    deadline = time.monotonic() + 2
    while not fired and time.monotonic() < deadline:
        time.sleep(0.01)
    scheduler.stop()
    assert fired == ["check the build"]


def test_schedule_reminder_uses_data_dir_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.reminders import schedule_reminder

    item = schedule_reminder("call mom", minutes=5)

    assert item["text"] == "call mom"
    path = tmp_path / "reminders.json"
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip().startswith("[")


def test_schedule_reminder_fires_default_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.reminders as reminders

    fired = []
    reminders.set_default_on_due(lambda reminder: fired.append(reminder["text"]))

    reminders.schedule_reminder("call mom", minutes=0.02)

    deadline = time.monotonic() + 2
    while not fired and time.monotonic() < deadline:
        time.sleep(0.01)

    assert fired == ["call mom"]


def test_reminder_store_lists_upcoming_notifications(tmp_path):
    from app.reminders import ReminderStore

    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    store.add("take a break", datetime.now() + timedelta(minutes=5))
    store.add("call mom", datetime.now() + timedelta(minutes=15))

    upcoming = store.upcoming(limit=10)

    assert len(upcoming) == 2
    assert {item["text"] for item in upcoming} == {"take a break", "call mom"}
    assert all(item["due_at"] for item in upcoming)


def test_memory_store_dream_keeps_high_value_fact(tmp_path):
    from app.memory import MemoryStore

    store = MemoryStore(path=str(tmp_path / "memory.md"))
    store.observe("User prefers short answers")
    store.observe("User prefers short answers")
    store.observe("The project is called vivo-agent")
    summary = store.dream(["User prefers short answers", "The project is called vivo-agent"])
    assert "short answers" in summary.lower()
    assert "vivo-agent" in summary.lower()


def test_dream_scheduler_reports_state_and_manual_trigger(tmp_path):
    from app.memory import DreamScheduler, MemoryStore

    store = MemoryStore(path=str(tmp_path / "memory.md"))
    states = []
    scheduler = DreamScheduler(store, interval_seconds=999, on_state=lambda active: states.append(active))

    store.observe("User prefers short answers")
    store.observe("The project is called vivo-agent")
    summary = scheduler.trigger()

    assert summary and "short answers" in summary.lower()
    assert states == [True, False]
