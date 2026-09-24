from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from app.reminders import ReminderScheduler, ReminderStore, parse_reminder

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _utc_user_zone(monkeypatch):
    """Deterministic default: user zone = server zone (UTC in the container).
    Dedicated tests override this with a real IANA zone."""
    import app.config as config

    monkeypatch.setattr(config, "USER_TIMEZONE", "")


def test_parse_relative_reminder():
    now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
    due = parse_reminder("in 5 minutes", now=now)
    assert due == now + timedelta(minutes=5)
    assert due.tzinfo is not None


def test_parse_time_reminder():
    now = datetime(2026, 9, 18, 11, 30, 0, tzinfo=UTC)
    due = parse_reminder("at 12:15", now=now)
    assert due == datetime(2026, 9, 18, 12, 15, 0, tzinfo=UTC)


def test_parse_time_reminder_uses_user_zone(monkeypatch):
    import app.config as config

    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    # 12:00 UTC is 13:00 BST, so "at 09:00" means the next London morning
    now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
    assert parse_reminder("at 09:00", now=now) == datetime(2026, 9, 19, 8, 0, 0, tzinfo=UTC)
    # winter: London is on GMT, so 09:00 local is 09:00 UTC
    now = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
    assert parse_reminder("at 09:00", now=now) == datetime(2026, 1, 16, 9, 0, 0, tzinfo=UTC)


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
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    store.add("take a break", datetime.now(UTC) + timedelta(minutes=5))
    store.add("call mom", datetime.now(UTC) + timedelta(minutes=15))

    upcoming = store.upcoming(limit=10)

    assert len(upcoming) == 2
    assert {item["text"] for item in upcoming} == {"take a break", "call mom"}
    assert all(item["due_at"] for item in upcoming)
    assert all(item["due_at"].endswith("+00:00") for item in upcoming)


def test_reminder_text_uses_user_name_and_prompt(monkeypatch):
    import app.config as config
    from app.reminders import reminder_message

    monkeypatch.setattr(config, "USER_NAME", "Alex", raising=False)

    msg = reminder_message("take a break")

    assert "Hey Alex" in msg
    assert "this is your reminder to take a break" in msg.lower()


# -- next_due: wall-clock schedules in the user's time zone -------------------


def test_next_due_daily_rolls_to_next_day():
    from app.reminders import next_due

    # before the time today -> today; after -> tomorrow
    assert next_due("daily", time="09:00", now=datetime(2026, 9, 18, 8, 0, tzinfo=UTC)) == datetime(2026, 9, 18, 9, 0, tzinfo=UTC)
    assert next_due("daily", time="09:00", now=datetime(2026, 9, 18, 10, 0, tzinfo=UTC)) == datetime(2026, 9, 19, 9, 0, tzinfo=UTC)


def test_next_due_daily_uses_user_zone(monkeypatch):
    import app.config as config
    from app.reminders import next_due

    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    # BST: a 09:00 London item fires at 08:00 UTC
    assert next_due("daily", time="09:00", now=datetime(2026, 9, 18, 1, 0, tzinfo=UTC)) == datetime(2026, 9, 18, 8, 0, tzinfo=UTC)
    # winter GMT: the same wall clock fires at 09:00 UTC
    assert next_due("daily", time="09:00", now=datetime(2026, 1, 15, 1, 0, tzinfo=UTC)) == datetime(2026, 1, 15, 9, 0, tzinfo=UTC)


def test_next_due_weekly_morning_briefing():
    from app.reminders import next_due

    days = [0, 1, 2, 3, 4]  # Mon..Fri
    # Wednesday 10:00 -> Thursday 09:00
    assert next_due("weekly", time="09:00", days=days, now=datetime(2026, 9, 16, 10, 0, tzinfo=UTC)) == datetime(2026, 9, 17, 9, 0, tzinfo=UTC)
    # Saturday 10:00 -> Monday 09:00 (2026-09-19 is a Saturday)
    assert next_due("weekly", time="09:00", days=days, now=datetime(2026, 9, 19, 10, 0, tzinfo=UTC)) == datetime(2026, 9, 21, 9, 0, tzinfo=UTC)


def test_next_due_once_past_raises():
    from app.reminders import next_due

    with pytest.raises(ValueError):
        next_due("once", at="2026-09-18T09:00:00", now=datetime(2026, 9, 18, 10, 0, tzinfo=UTC))


def test_next_due_once_aware_at_is_absolute(monkeypatch):
    import app.config as config
    from app.reminders import next_due

    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    # an aware timestamp is absolute UTC regardless of the user zone
    assert next_due("once", at="2026-09-18T09:00:00+00:00", now=datetime(2026, 9, 18, 8, 0, tzinfo=UTC)) == datetime(2026, 9, 18, 9, 0, tzinfo=UTC)


def test_next_due_interval_is_now_plus_every():
    from app.reminders import next_due

    now = datetime(2026, 9, 18, 12, 7, 30, tzinfo=UTC)
    assert next_due("interval", every=30, now=now) == now + timedelta(minutes=30)
    assert next_due("interval", every=120, now=now) == now + timedelta(hours=2)


def test_next_due_interval_rejects_bad_every():
    from app.reminders import next_due

    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    for bad in (None, 0, -5, "soon", 1441):
        with pytest.raises(ValueError):
            next_due("interval", every=bad, now=now)


def _future_at(days: int = 1, hour: int = 15, minute: int = 30) -> str:
    when = (datetime.now(UTC) + timedelta(days=days)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return when.isoformat(timespec="seconds")


# -- store: create / update / advance ----------------------------------------


def test_create_once_stores_schedule(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    at = _future_at()
    item = store.create({
        "text": "call mom", "type": "reminder", "response": "text", "repeat": "once",
        "at": at, "session_id": "s1",
    })
    assert item["type"] == "reminder"
    assert item["response"] == "text"
    assert item["repeat"] == "once"
    assert item["at"] == at
    assert item["due_at"] == at
    assert item["session_id"] == "s1"
    assert item["status"] == "pending"


def test_create_defaults_to_spoken_reminder(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "call mom", "repeat": "once", "at": _future_at()})
    assert item["type"] == "reminder"
    assert item["response"] == "spoken"


def test_create_daily_normalizes_time(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "water plants", "type": "reminder", "response": "spoken", "repeat": "daily", "time": "9:05"})
    assert item["time"] == "09:05"
    assert item["days"] == []
    assert item["due_at"].endswith("09:05:00+00:00")


def test_create_weekly_requires_days(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    with pytest.raises(ValueError):
        store.create({"text": "briefing", "type": "task", "repeat": "weekly", "time": "09:00", "days": []})
    item = store.create({"text": "briefing", "type": "task", "response": "text", "repeat": "weekly", "time": "09:00", "days": [0, 4]})
    assert item["type"] == "task"
    assert item["response"] == "text"
    assert item["days"] == [0, 4]


def test_create_interval_stores_every(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "stretch", "response": "text", "repeat": "interval", "every": 45})
    assert item["repeat"] == "interval"
    assert item["every"] == 45
    assert item["time"] is None
    assert item["days"] == []
    assert item["status"] == "pending"
    due = datetime.fromisoformat(item["due_at"])
    assert timedelta(minutes=44) < due - datetime.now(UTC) <= timedelta(minutes=46)


def test_create_interval_requires_every(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    with pytest.raises(ValueError):
        store.create({"text": "stretch", "repeat": "interval"})
    with pytest.raises(ValueError):
        store.create({"text": "stretch", "repeat": "interval", "every": 0})
    with pytest.raises(ValueError):
        store.create({"text": "stretch", "repeat": "interval", "every": 1441})


def test_create_rejects_past_once_and_bad_shapes(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    with pytest.raises(ValueError):
        store.create({"text": "x", "repeat": "once", "at": "2020-01-01T09:00:00"})
    with pytest.raises(ValueError):
        store.create({"text": "x", "type": "shouty", "repeat": "once", "at": _future_at()})
    with pytest.raises(ValueError):
        store.create({"text": "x", "response": "shouty", "repeat": "once", "at": _future_at()})
    # an unrecognised legacy kind is rejected too
    with pytest.raises(ValueError):
        store.create({"text": "x", "kind": "shouty", "repeat": "once", "at": _future_at()})


def test_advance_interval_rolls_from_fire_time(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "hydrate", "repeat": "interval", "every": 30})
    now = datetime.now(UTC) + timedelta(minutes=31)
    store.advance(item["id"], now=now)
    advanced = store.get(item["id"])
    assert advanced["status"] == "pending"
    assert advanced["last_fired_at"] is not None
    assert advanced["due_at"] == (now + timedelta(minutes=30)).isoformat(timespec="seconds")


def test_update_switches_repeat_kinds(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "standup", "repeat": "daily", "time": "09:00"})

    as_interval = store.update(item["id"], {"repeat": "interval", "every": 90})
    assert as_interval["repeat"] == "interval"
    assert as_interval["every"] == 90
    assert as_interval["time"] is None

    as_daily = store.update(item["id"], {"repeat": "daily", "time": "09:00"})
    assert as_daily["repeat"] == "daily"
    assert as_daily["every"] is None
    assert as_daily["time"] == "09:00"

    with pytest.raises(ValueError):
        store.update(item["id"], {"repeat": "interval"})  # no every anywhere


def test_update_switches_type_and_response(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "standup", "repeat": "daily", "time": "09:00"})

    updated = store.update(item["id"], {"type": "task", "response": "text"})
    assert updated["type"] == "task"
    assert updated["response"] == "text"
    assert updated["due_at"] == item["due_at"]  # schedule untouched


def test_update_rearms_fired_item(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "old", "repeat": "once", "at": _future_at()})
    store.mark_fired(item["id"])
    assert store.get(item["id"])["status"] == "fired"

    new_at = _future_at(days=2, hour=8, minute=0)
    updated = store.update(item["id"], {"repeat": "once", "at": new_at, "response": "text"})
    assert updated["status"] == "pending"
    assert updated["fired_at"] is None
    assert updated["due_at"] == new_at
    assert updated["response"] == "text"
    with pytest.raises(KeyError):
        store.update("missing", {"repeat": "once", "at": new_at})


def test_advance_repeating_rolls_over(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "standup", "response": "text", "repeat": "daily", "time": "23:59"})
    assert item["due_at"].endswith("T23:59:00+00:00")
    now = datetime.now(UTC).replace(hour=23, minute=59, second=30, microsecond=0)
    store.advance(item["id"], now=now)
    advanced = store.get(item["id"])
    assert advanced["status"] == "pending"
    assert advanced["last_fired_at"] is not None
    # the next fire is the next wall-clock 23:59, not fire-time + 1 day
    assert advanced["due_at"] == (now + timedelta(days=1)).replace(second=0, microsecond=0).isoformat(timespec="seconds")


def test_advance_once_marks_fired(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    fired_now = (datetime.now(UTC) + timedelta(days=1)).replace(
        hour=15, minute=0, second=1, microsecond=0
    )
    item = store.create({"text": "once", "repeat": "once", "at": fired_now.isoformat(timespec="seconds")})
    store.advance(item["id"], now=fired_now)
    advanced = store.get(item["id"])
    assert advanced["status"] == "fired"
    assert advanced["fired_at"] == fired_now.isoformat(timespec="seconds")


def test_list_all_orders_pending_then_fired(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    a = store.create({"text": "sooner", "repeat": "once", "at": _future_at(hour=14, minute=0)})
    store.create({"text": "later", "repeat": "once", "at": _future_at(hour=16, minute=0)})
    store.mark_fired(a["id"])
    listing = store.list_all()
    assert [i["text"] for i in listing] == ["later", "sooner"]
    assert listing[1]["status"] == "fired"


# -- legacy rows: kind mapping and naive-UTC migration ------------------------


def test_legacy_items_normalized_on_load(tmp_path):
    path = tmp_path / "reminders.json"
    path.write_text(
        '[{"id": "a1", "text": "legacy", "due_at": "2026-09-18T12:00:00"}]',
        encoding="utf-8",
    )
    store = ReminderStore(path=str(path))
    item = store.get("a1")
    assert item["type"] == "reminder"
    assert item["response"] == "spoken"
    assert item["repeat"] == "once"
    assert item["days"] == []
    assert item["session_id"] is None
    assert item["status"] == "pending"
    # a naive legacy timestamp is the old server (UTC) wall clock, kept absolute
    assert item["due_at"] == "2026-09-18T12:00:00+00:00"


def test_legacy_kind_maps_to_type_and_response(tmp_path):
    path = tmp_path / "reminders.json"
    path.write_text(
        json.dumps([
            {"id": "a", "text": "s", "kind": "spoken", "due_at": "2026-09-18T12:00:00"},
            {"id": "b", "text": "s", "kind": "silent", "due_at": "2026-09-18T12:00:00"},
            {"id": "c", "text": "s", "kind": "system", "due_at": "2026-09-18T12:00:00", "repeat": "daily", "time": "09:00"},
        ]),
        encoding="utf-8",
    )
    store = ReminderStore(path=str(path))
    by_id = {i["id"]: i for i in store.list_all()}
    assert (by_id["a"]["type"], by_id["a"]["response"]) == ("reminder", "spoken")
    assert (by_id["b"]["type"], by_id["b"]["response"]) == ("reminder", "text")
    assert (by_id["c"]["type"], by_id["c"]["response"]) == ("task", "text")


def test_legacy_naive_repeating_due_reanchored_to_user_zone(tmp_path, monkeypatch):
    """Pre-fix repeating items stored a naive UTC due_at; on load the next
    fire is re-derived from the user's wall clock (`time`/`days`) so the item
    lands at the right local moment again."""
    import app.config as config
    from app.reminders import next_due

    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    path = tmp_path / "reminders.json"
    path.write_text(
        json.dumps([{
            "id": "w", "text": "briefing", "kind": "system", "repeat": "weekly",
            "time": "09:00", "days": [0, 1, 2, 3, 4], "due_at": "2026-09-19T09:00:00",
        }]),
        encoding="utf-8",
    )
    store = ReminderStore(path=str(path))
    item = store.get("w")
    expected = next_due("weekly", time="09:00", days=[0, 1, 2, 3, 4]).isoformat(timespec="seconds")
    assert item["due_at"] == expected
    assert item["due_at"].endswith("+00:00")


def test_legacy_naive_once_due_is_absolute_utc(tmp_path, monkeypatch):
    """A one-time legacy item fires at the stored instant, not one user hour
    off: 12:00 naive stays 12:00Z even for a London user."""
    import app.config as config

    monkeypatch.setattr(config, "USER_TIMEZONE", "Europe/London")
    path = tmp_path / "reminders.json"
    path.write_text(
        '[{"id": "o", "text": "call", "kind": "silent", "due_at": "2026-09-18T12:00:00"}]',
        encoding="utf-8",
    )
    assert ReminderStore(path=str(path)).get("o")["due_at"] == "2026-09-18T12:00:00+00:00"


# -- pipeline: delivery by type/response --------------------------------------


class _FakeSession:
    """Stands in for VoiceSession; mirrors its reply-slot contract."""

    def __init__(self, session_id="s1"):
        self.session_id = session_id
        self.closed = False
        self.sent = []
        self.generation = 0

    def send(self, payload):
        self.sent.append(payload)

    def alive(self, gen):
        return not self.closed and self.generation == gen

    def begin_scheduled_reply(self):
        import queue

        self.generation += 1
        return self.generation, queue.Queue()

    def end_scheduled_reply(self, gen):
        if self.generation != gen:
            return False
        self.generation += 1
        return True


def _fake_engines(tmp_path, sessions, agent=None):
    import numpy as np
    from types import SimpleNamespace
    from app.pipeline import Engines
    from app.skills import SkillStore

    observations = []
    tts_calls = []
    turns = []

    class FakeTTS:
        def synthesize(self, text):
            tts_calls.append(text)
            return np.array([0.0, 0.1], dtype=np.float32)

    engines = object.__new__(Engines)  # real methods, no __init__
    engines.memory = SimpleNamespace(
        observe=lambda text: observations.append(text),
        core_summary=lambda: "No important memory yet.",
    )
    engines.skills = SkillStore(path=str(tmp_path / "skills"))
    engines.tts = SimpleNamespace(synthesize=FakeTTS().synthesize)
    engines.agent = agent or SimpleNamespace(
        user_profile=None,
        reply=lambda *a, **k: iter([]),
        summarize=lambda messages: "summary",
    )
    engines.sessions = SimpleNamespace(
        ids=lambda: {s.session_id for s in sessions},
        conversation_for=lambda sid: SimpleNamespace(
            messages=lambda: [],
            add_turn=lambda user, assistant: turns.append((user, assistant)),
            maybe_compact=lambda summarize: None,
        ),
    )
    for s in sessions:  # the TTS worker reaches the engine via session.engines
        s.engines = engines
    return engines, {"tts_calls": tts_calls, "observations": observations, "turns": turns}


def test_pipeline_spoken_reminder_targets_attached_session(tmp_path, monkeypatch):
    import app.pipeline as pipeline
    from app.pipeline import Engines

    target = _FakeSession("s1")
    other = _FakeSession("s2")
    monkeypatch.setattr(pipeline, "_ACTIVE", {target, other})
    engines, hooks = _fake_engines(tmp_path, [target, other])
    Engines._handle_due_reminder(engines, {"text": "take a break", "type": "reminder", "response": "spoken", "session_id": "s1"})

    reminder = [p for p in target.sent if isinstance(p, dict) and p.get("type") == "reminder"]
    assert reminder and reminder[0]["text"]
    assert reminder[0]["response"] == "spoken"
    assert any(isinstance(p, bytes) for p in target.sent)  # audio for the attached session
    assert other.sent == []  # other conversations get nothing
    assert any("take a break" in o for o in hooks["observations"])


def test_pipeline_text_reminder_broadcasts_text_only(tmp_path, monkeypatch):
    import app.pipeline as pipeline
    from app.pipeline import Engines

    one = _FakeSession("s1")
    two = _FakeSession("s2")
    monkeypatch.setattr(pipeline, "_ACTIVE", {one, two})
    engines, hooks = _fake_engines(tmp_path, [one, two])
    Engines._handle_due_reminder(engines, {"text": "build passed", "type": "reminder", "response": "text"})

    for session in (one, two):
        frames = [p for p in session.sent if isinstance(p, dict)]
        assert frames[0]["type"] == "reminder"
        assert frames[0]["response"] == "text"
        assert frames[-1] == {"type": "reply_done"}
        assert not any(isinstance(p, bytes) for p in session.sent)
    assert hooks["tts_calls"] == []


def test_pipeline_text_task_runs_agent_without_audio(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import app.pipeline as pipeline
    from app.agent import ToolRound
    from app.pipeline import Engines

    target = _FakeSession("s1")
    monkeypatch.setattr(pipeline, "_ACTIVE", {target})
    agent = SimpleNamespace(
        user_profile=None,
        reply=lambda text, execute, history=None: iter([ToolRound(), "Checked ", "the logs."]),
        summarize=lambda messages: "summary",
    )
    engines, hooks = _fake_engines(tmp_path, [target], agent=agent)
    Engines._handle_due_reminder(engines, {"text": "summarize errors", "type": "task", "response": "text", "session_id": "s1"})

    assert hooks["turns"] == [("summarize errors", "Checked the logs.")]
    frames = [p for p in target.sent if isinstance(p, dict)]
    assert frames[0] == {"type": "agent_text", "delta": "Checked the logs.", "start": True}
    assert frames[-1] == {"type": "reply_done"}
    assert not any(isinstance(p, bytes) for p in target.sent)
    assert hooks["tts_calls"] == []


def test_pipeline_spoken_task_speaks_result(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import app.pipeline as pipeline
    from app.agent import ToolRound
    from app.pipeline import Engines

    target = _FakeSession("s1")
    monkeypatch.setattr(pipeline, "_ACTIVE", {target})
    agent = SimpleNamespace(
        user_profile=None,
        reply=lambda text, execute, history=None: iter([ToolRound(), "All quiet."]),
        summarize=lambda messages: "summary",
    )
    engines, hooks = _fake_engines(tmp_path, [target], agent=agent)
    Engines._handle_due_reminder(engines, {"text": "check the logs", "type": "task", "response": "spoken", "session_id": "s1"})

    assert hooks["tts_calls"] == ["All quiet."]
    frames = [p for p in target.sent if isinstance(p, dict)]
    assert frames[0] == {"type": "agent_text", "delta": "All quiet.", "start": True}
    assert frames[-1] == {"type": "reply_done"}
    assert any(isinstance(p, bytes) for p in target.sent)


def test_speakable_strips_markdown():
    from app.pipeline import _speakable

    out = _speakable(
        "# Title\n\n**bold** and `code` and [a link](http://x) here.\n"
        "- bullet one\n- bullet two\n*ital*"
    )
    assert out == "Title bold and code and a link here. bullet one bullet two ital"


def test_pipeline_spoken_task_chunks_long_answer(tmp_path, monkeypatch):
    """A long spoken answer is synthesized per sentence, never as one
    one-shot call (one-shot synthesis of a long answer garbles), and the
    caption keeps the raw markdown answer while the TTS input is clean."""
    from types import SimpleNamespace
    import app.pipeline as pipeline
    from app import config
    from app.agent import ToolRound
    from app.pipeline import Engines

    target = _FakeSession("s1")
    monkeypatch.setattr(pipeline, "_ACTIVE", {target})
    answer = "**ai-dashboard** - a Flask app. " + " ".join(
        f"Sentence number {i} is here." for i in range(1, 12)
    )
    agent = SimpleNamespace(
        user_profile=None,
        reply=lambda text, execute, history=None: iter([ToolRound(), answer]),
        summarize=lambda messages: "summary",
    )
    engines, hooks = _fake_engines(tmp_path, [target], agent=agent)
    Engines._handle_due_reminder(engines, {"text": "scan the workspace", "type": "task", "response": "spoken", "session_id": "s1"})

    assert len(hooks["tts_calls"]) > 1
    assert all(len(c) <= config.SENTENCE_MAX_CHARS for c in hooks["tts_calls"])
    assert not any("**" in c for c in hooks["tts_calls"])
    audio = [p for p in target.sent if isinstance(p, bytes)]
    assert len(audio) == len(hooks["tts_calls"])
    frames = [p for p in target.sent if isinstance(p, dict)]
    assert frames[0] == {"type": "agent_text", "delta": answer, "start": True}
    assert frames[-1] == {"type": "reply_done"}


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


# -- last_response (what a task last produced) --------------------------------


def test_set_last_response_keeps_schedule(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "briefing", "type": "task", "repeat": "daily", "time": "09:00"})
    due_before = item["due_at"]

    updated = store.set_last_response(item["id"], "All clear this morning.")

    assert updated["last_response"] == "All clear this morning."
    reloaded = store.get(item["id"])
    assert reloaded["status"] == "pending"
    assert reloaded["due_at"] == due_before
    assert reloaded["fired_at"] is None


def test_set_last_response_truncates_clears_and_missing(tmp_path):
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "briefing", "type": "task", "repeat": "daily", "time": "09:00"})

    store.set_last_response(item["id"], "x" * 1500)
    assert len(store.get(item["id"])["last_response"]) == 1000

    store.set_last_response(item["id"], "   ")
    assert store.get(item["id"])["last_response"] is None

    with pytest.raises(KeyError):
        store.set_last_response("missing", "hi")


def test_last_response_survives_reload_and_legacy_rows(tmp_path):
    path = tmp_path / "reminders.json"
    item = ReminderStore(path=str(path)).create(
        {"text": "b", "type": "task", "repeat": "daily", "time": "09:00"}
    )
    ReminderStore(path=str(path)).set_last_response(item["id"], "Done.")
    assert ReminderStore(path=str(path)).get(item["id"])["last_response"] == "Done."

    path.write_text(
        '[{"id": "a1", "text": "legacy", "due_at": "2026-09-18T12:00:00"}]',
        encoding="utf-8",
    )
    assert ReminderStore(path=str(path)).get("a1")["last_response"] is None


def test_task_records_last_response(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import app.pipeline as pipeline
    from app.agent import ToolRound
    from app.pipeline import Engines

    target = _FakeSession("s1")
    monkeypatch.setattr(pipeline, "_ACTIVE", {target})
    agent = SimpleNamespace(
        user_profile=None,
        reply=lambda text, execute, history=None: iter([ToolRound(), "Brief: all good."]),
        summarize=lambda messages: "summary",
    )
    engines, _hooks = _fake_engines(tmp_path, [target], agent=agent)
    store = ReminderStore(path=str(tmp_path / "reminders.json"))
    item = store.create({"text": "morning briefing", "type": "task", "repeat": "daily", "time": "09:00"})
    engines.reminders = store

    Engines._handle_due_reminder(
        engines, {"id": item["id"], "text": "morning briefing", "type": "task", "session_id": "s1"}
    )

    recorded = store.get(item["id"])
    assert recorded["last_response"] == "Brief: all good."
    assert recorded["status"] == "pending"  # schedule untouched


# -- LLM tools: set_reminder / list_reminders / delete_reminder --------------


def _tools_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.reminders import get_default_store

    return get_default_store()


def test_tool_set_reminder_daily(tmp_path, monkeypatch):
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)  # sets DATA_DIR first
    result = tools_execute("set_reminder", {"text": "water plants", "repeat": "daily", "time": "08:30"})

    assert not result.startswith("error:"), result
    items = store.list_all()
    assert items[0]["text"] == "water plants"
    assert items[0]["type"] == "reminder"
    assert items[0]["response"] == "spoken"
    assert items[0]["time"] == "08:30"
    assert "daily" in result


def test_tool_set_reminder_weekly_task(tmp_path, monkeypatch):
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)
    result = tools_execute(
        "set_reminder",
        {"text": "morning briefing", "type": "task", "response": "text", "repeat": "weekly", "time": "09:00", "days": [0, 4]},
    )

    assert not result.startswith("error:"), result
    item = store.list_all()[0]
    assert item["type"] == "task"
    assert item["response"] == "text"
    assert item["days"] == [0, 4]


def test_tool_set_reminder_interval(tmp_path, monkeypatch):
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)
    result = tools_execute(
        "set_reminder", {"text": "stretch", "response": "text", "repeat": "interval", "every_minutes": 45}
    )

    assert not result.startswith("error:"), result
    item = store.list_all()[0]
    assert item["repeat"] == "interval"
    assert item["every"] == 45
    assert "interval" in result
    assert "every 45 min" in tools_execute("list_reminders", {})


def test_tool_set_reminder_in_minutes(tmp_path, monkeypatch):
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)
    result = tools_execute("set_reminder", {"text": "call mom", "in_minutes": 10})

    assert not result.startswith("error:"), result
    item = store.list_all()[0]
    assert item["repeat"] == "once"
    assert item["at"] is not None


def test_tool_set_reminder_in_minutes_is_absolute_utc(tmp_path, monkeypatch):
    """in_minutes must not shift with the user zone: 10 minutes from *now*,
    whatever the local clock says."""
    import datetime as dt
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)
    before = dt.datetime.now(dt.timezone.utc)
    assert not tools_execute("set_reminder", {"text": "call mom", "in_minutes": 10}).startswith("error:")
    item = store.list_all()[0]
    due = dt.datetime.fromisoformat(item["due_at"])
    assert before + dt.timedelta(minutes=9, seconds=30) < due < before + dt.timedelta(minutes=10, seconds=30)


def test_tool_set_reminder_reports_missing_details(tmp_path, monkeypatch):
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)

    # no when at all (once needs at/in_minutes)
    assert tools_execute("set_reminder", {"text": "x"}).startswith("error:")
    # interval without a span
    assert tools_execute("set_reminder", {"text": "x", "repeat": "interval"}).startswith("error:")
    # daily without a time
    assert tools_execute("set_reminder", {"text": "x", "repeat": "daily"}).startswith("error:")
    # weekly without weekdays
    assert tools_execute(
        "set_reminder", {"text": "x", "repeat": "weekly", "time": "09:00"}
    ).startswith("error:")
    # invalid shapes are rejected by the store
    assert tools_execute(
        "set_reminder", {"text": "x", "type": "shouty", "repeat": "daily", "time": "09:00"}
    ).startswith("error:")
    assert tools_execute(
        "set_reminder", {"text": "x", "response": "shouty", "repeat": "daily", "time": "09:00"}
    ).startswith("error:")

    assert store.list_all() == []


def test_tool_delete_reminder(tmp_path, monkeypatch):
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)
    assert not tools_execute("set_reminder", {"text": "call mom", "in_minutes": 5}).startswith("error:")
    item_id = store.list_all()[0]["id"]

    assert tools_execute("delete_reminder", {"id": item_id}) == "deleted: call mom"
    assert store.list_all() == []
    assert tools_execute("delete_reminder", {"id": item_id}).startswith("error:")


def test_tool_list_reminders_includes_last_response(tmp_path, monkeypatch):
    from app.tools import execute as tools_execute

    store = _tools_store(tmp_path, monkeypatch)
    assert not tools_execute(
        "set_reminder", {"text": "briefing", "type": "task", "response": "text", "repeat": "daily", "time": "09:00"}
    ).startswith("error:")
    item_id = store.list_all()[0]["id"]
    store.set_last_response(item_id, "All quiet.")

    listing = tools_execute("list_reminders", {})

    assert item_id in listing
    assert "task/text" in listing
    assert "daily 09:00" in listing
    assert "last response: All quiet." in listing
