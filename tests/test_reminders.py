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


def test_reminder_text_uses_user_name_and_prompt(monkeypatch):
    import app.config as config
    from app.reminders import reminder_message

    monkeypatch.setattr(config, "USER_NAME", "Rob", raising=False)

    msg = reminder_message("take a break")

    assert "Hey Rob" in msg
    assert "this is your reminder to take a break" in msg.lower()


def test_due_reminder_uses_server_tts(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    import app.config as config

    monkeypatch.setattr(config, "USER_NAME", "Rob", raising=False)

    calls = []

    class FakeTTS:
        def synthesize(self, text):
            calls.append(text)
            return np.array([0.0, 0.1, 0.2], dtype=np.float32)

    class FakeSession:
        def __init__(self):
            self.closed = False
            self.sent = []
            self.engines = SimpleNamespace(tts=FakeTTS())

        def send(self, payload):
            self.sent.append(payload)

    session = FakeSession()
    engines = SimpleNamespace(
        memory=SimpleNamespace(observe=lambda *args, **kwargs: None),
        tts=session.engines.tts,
    )
    engines._handle_due_reminder = lambda reminder: None

    def trigger(reminder):
        text = str(reminder.get("text", "Reminder")).strip()
        spoken = __import__("app.reminders", fromlist=["reminder_message"]).reminder_message(text)
        session.send({"type": "reminder", "text": spoken})
        audio = session.engines.tts.synthesize(spoken)
        if audio.size:
            session.send((audio * 32767).astype("<i2").tobytes())

    trigger({"text": "take a break"})

    assert calls == ["Hey Rob, this is your reminder to take a break"]
    assert session.sent[0]["type"] == "reminder"
    assert isinstance(session.sent[1], bytes)


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
