"""T021: named conversation sessions (offline).

SessionStore behaviour (create/activate/delete, persistence, index upkeep via
the Conversation change callback, legacy migration, orphan adoption, corrupt
file tolerance, set_limits), VoiceSession session binding/switching, the WS
session handshake + commands, and /api/sessions (offline fakes).
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main, pipeline
from app.conversation import Conversation, SessionStore
from app.pipeline import VoiceSession
from app.wake import WakeState


# ---------------- SessionStore: basics ----------------

def test_starts_with_one_active_session():
    store = SessionStore()
    rows = store.list_sessions()
    assert len(rows) == 1
    assert rows[0]["active"] is True
    assert rows[0]["turns"] == 0
    assert store.active_id == rows[0]["id"]


def test_create_makes_new_active_and_unique_ids():
    store = SessionStore()
    first = store.active_id
    second = store.create()
    third = store.create()
    assert len({first, second, third}) == 3
    assert store.active_id == third
    assert {r["id"] for r in store.list_sessions()} == {first, second, third}
    assert [r["id"] for r in store.list_sessions() if r["active"]] == [third]


def test_set_active_unknown_raises():
    store = SessionStore()
    with pytest.raises(KeyError):
        store.set_active("nope")


def test_rename_sets_display_name_and_can_clear():
    store = SessionStore()
    sid = store.active_id
    store.rename(sid, "Work Notes")
    row = [r for r in store.list_sessions() if r["id"] == sid][0]
    assert row["name"] == "Work Notes"

    store.rename(sid, "   ")
    row = [r for r in store.list_sessions() if r["id"] == sid][0]
    assert row["name"] == ""


def test_conversation_for_unknown_raises():
    store = SessionStore()
    with pytest.raises(KeyError):
        store.conversation_for("nope")


def test_delete_removes_and_reactivates_newest():
    store = SessionStore()
    first = store.active_id
    second = store.create()
    store.set_active(first)
    store.conversation_for(second).add_turn("u", "a")  # second becomes newest
    store.delete(second)
    assert second not in store.ids()
    assert store.active_id == first

    store.delete(first)  # last one: a fresh active session appears
    assert store.active_id not in (first, second)
    assert len(store.list_sessions()) == 1


def test_delete_unknown_raises():
    store = SessionStore()
    with pytest.raises(KeyError):
        store.delete("nope")


def test_list_sorted_by_last_used():
    store = SessionStore()
    a = store.active_id
    b = store.create()
    store.conversation_for(a).add_turn("u", "a")  # a becomes newest
    assert [r["id"] for r in store.list_sessions()] == [a, b]


def test_set_limits_propagates():
    store = SessionStore()
    conv = store.conversation_for(store.active_id)
    other = store.create()
    store.set_limits(500, 2)
    assert (store.compact_after_chars, store.keep_recent_turns) == (500, 2)
    assert (conv.compact_after_chars, conv.keep_recent_turns) == (500, 2)
    # already-loaded and not-yet-loaded sessions both pick up the new limits
    assert store.conversation_for(other).compact_after_chars == 500
    assert store.conversation_for(store.active_id).keep_recent_turns == 2


# ---------------- SessionStore: persistence ----------------

def test_persistence_roundtrip(tmp_path):
    store = SessionStore(data_dir=str(tmp_path))
    sid = store.active_id
    conv = store.conversation_for(sid)
    conv.add_turn("hello", "hi there")
    conv.add_turn("bye", "goodbye")

    reopened = SessionStore(data_dir=str(tmp_path))
    assert reopened.active_id == sid
    assert reopened.conversation_for(sid).turns == [("hello", "hi there"), ("bye", "goodbye")]
    assert reopened.list_sessions()[0]["turns"] == 2


def test_on_change_updates_index(tmp_path):
    store = SessionStore(data_dir=str(tmp_path))
    sid = store.active_id
    before = store.list_sessions()[0]["last_used"]
    store.conversation_for(sid).add_turn("hello", "hi")
    row = [r for r in store.list_sessions() if r["id"] == sid][0]
    assert row["turns"] == 1
    assert row["last_used"] >= before
    index = json.loads((tmp_path / "sessions.json").read_text())
    assert index["active"] == sid
    assert index["sessions"][sid]["turns"] == 1
    assert (tmp_path / "sessions" / f"{sid}.json").exists()


def test_corrupt_index_starts_fresh(tmp_path):
    (tmp_path / "sessions.json").write_text("not json {")
    store = SessionStore(data_dir=str(tmp_path))
    assert len(store.list_sessions()) == 1
    assert store.active_id in store.ids()


def test_corrupt_session_file_is_tolerated(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    (d / "bad.json").write_text("not json {")
    store = SessionStore(data_dir=str(tmp_path))
    rows = {r["id"]: r for r in store.list_sessions()}
    assert rows["bad"]["turns"] == 0
    assert store.active_id == "bad"  # the only session; loads empty, no crash
    assert store.conversation_for("bad").turns == []


def test_orphan_files_adopted(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    (d / "orphan1.json").write_text(
        json.dumps({"summary": "s", "turns": [["u", "a"], ["u2", "a2"]]})
    )
    (d / "orphan2.json").write_text("not json")
    (d / "ignore.txt").write_text("x")
    store = SessionStore(data_dir=str(tmp_path))
    rows = {r["id"]: r for r in store.list_sessions()}
    assert set(rows) == {"orphan1", "orphan2"}
    assert rows["orphan1"]["turns"] == 2
    assert rows["orphan2"]["turns"] == 0
    # a healthy orphan is still fully loadable
    conv = store.conversation_for("orphan1")
    assert conv.turns == [("u", "a"), ("u2", "a2")]
    assert conv.summary == "s"


def test_legacy_conversation_migrated(tmp_path):
    legacy = tmp_path / "conversation.json"
    legacy.write_text(
        json.dumps({"summary": "old summary", "turns": [["u1", "a1"], ["u2", "a2"]]})
    )
    store = SessionStore(data_dir=str(tmp_path))
    rows = store.list_sessions()
    assert len(rows) == 1
    sid = rows[0]["id"]
    assert rows[0]["active"] is True
    assert rows[0]["turns"] == 2
    conv = store.conversation_for(sid)
    assert conv.turns == [("u1", "a1"), ("u2", "a2")]
    assert conv.summary == "old summary"
    assert not legacy.exists()  # imported, then removed

    reopened = SessionStore(data_dir=str(tmp_path))
    assert len(reopened.list_sessions()) == 1  # not re-imported
    assert reopened.active_id == sid


def test_legacy_empty_file_removed(tmp_path):
    (tmp_path / "conversation.json").write_text(
        json.dumps({"summary": None, "turns": []})
    )
    store = SessionStore(data_dir=str(tmp_path))
    assert len(store.list_sessions()) == 1
    assert not (tmp_path / "conversation.json").exists()


# ---------------- Conversation change callback ----------------

def test_on_change_called_after_add_turn(tmp_path):
    calls = []
    conv = Conversation(data_path=str(tmp_path / "c.json"), on_change=lambda c: calls.append(c))
    conv.add_turn("u", "a")
    assert calls == [conv]
    calls.clear()
    conv.add_turn("   ", "a")  # empty turns are skipped, no callback
    assert calls == []


def test_on_change_called_after_compact():
    calls = []
    conv = Conversation(
        compact_after_chars=10, keep_recent_turns=1, on_change=lambda c: calls.append(1)
    )
    for i in range(5):
        conv.add_turn(f"u{i}", "aaaaa")
    assert len(calls) == 5
    assert conv.compact(lambda msgs: "the summary")
    assert len(calls) == 6
    assert conv.summary == "the summary"


def test_callback_error_does_not_break_add_turn():
    def boom(c):
        raise RuntimeError("nope")

    conv = Conversation(on_change=boom)
    conv.add_turn("u", "a")  # must not raise
    assert conv.turns == [("u", "a")]


# ---------------- VoiceSession binding ----------------

class _RecWS:
    async def send_text(self, t: str) -> None:
        pass

    async def send_bytes(self, b: bytes) -> None:
        pass


def _engines(store: SessionStore):
    return SimpleNamespace(
        stt=SimpleNamespace(transcribe=lambda samples: "hi"),
        tts=SimpleNamespace(synthesize=lambda text: None),
        agent=SimpleNamespace(summarize=lambda msgs: ""),
        sessions=store,
        wake=WakeState(),  # disabled (no phrase): legacy always-answer behavior
    )


def test_voice_session_binding():
    store = SessionStore()
    engines = _engines(store)
    original = store.active_id

    async def go() -> None:
        s1 = VoiceSession(_RecWS(), engines)
        assert s1.session_id == original
        new_id = store.create()
        # a fresh connection binds to the (new) active session
        s2 = VoiceSession(_RecWS(), engines)
        assert s2.session_id == new_id
        # pinning by id; same Conversation object as s1
        s3 = VoiceSession(_RecWS(), engines, session_id=original)
        assert s3.session_id == original
        assert s3.conversation is s1.conversation
        # unknown id falls back to the active one
        s4 = VoiceSession(_RecWS(), engines, session_id="nope")
        assert s4.session_id == new_id
        # switch_session
        assert not s2.switch_session("nope")
        assert s2.session_id == new_id
        assert s2.switch_session(original)
        assert s2.session_id == original
        assert s2.conversation is s1.conversation

    asyncio.run(go())

def test_conversation_tools_manage_current_session():
    store = SessionStore()
    engines = _engines(store)

    async def go() -> None:
        session = VoiceSession(_RecWS(), engines)
        original = session.session_id
        created = session.execute_conversation_tool(
            "create_conversation", {"name": "Planning"}
        )
        current = session.session_id
        assert current != original and "created and switched" in created
        assert store.list_sessions()[0]["name"] == "Planning"
        assert "current" in session.execute_conversation_tool(
            "list_conversations", {}
        )
        assert "renamed" in session.execute_conversation_tool(
            "rename_conversation", {"id": "current", "name": "Today"}
        )
        assert store.list_sessions()[0]["name"] == "Today"
        assert "deleted" in session.execute_conversation_tool(
            "delete_conversation", {"id": "current"}
        )
        assert session.session_id == store.active_id
        assert current not in store.ids()

    asyncio.run(go())


def test_execute_tool_reports_timing_and_safe_error_kind(monkeypatch):
    store = SessionStore()
    engines = _engines(store)

    async def go() -> None:
        session = VoiceSession(_RecWS(), engines)
        monkeypatch.setattr(pipeline.tools, "execute", lambda name, args: "error: fetch failed (HTTP 503)")
        result, event = pipeline.execute_tool("web_fetch", {"url": "https://example.com"}, session)

        assert result.startswith("error:")
        assert event["status"] == "error"
        assert event["error_kind"] == "fetch_failed"
        assert isinstance(event["duration_ms"], int) and event["duration_ms"] >= 0

    asyncio.run(go())


# ---------------- serve_session handshake + commands ----------------

class _CmdWS:
    """Scripted WebSocket: feeds queued JSON commands, then disconnects."""

    def __init__(self, frames: list[str], query_params: dict | None = None) -> None:
        self.frames = list(frames)
        self.sent: list = []
        self.query_params = query_params or {}

    async def send_text(self, t: str) -> None:
        self.sent.append(json.loads(t))

    async def send_bytes(self, b: bytes) -> None:
        self.sent.append(b)

    async def receive(self) -> dict:
        # a real ws.receive() suspends on the network, giving the loop a
        # chance to run the send tasks VoiceSession.send schedules
        await asyncio.sleep(0)
        if self.frames:
            return {"text": self.frames.pop(0)}
        return {"type": "websocket.disconnect"}


def test_serve_session_handshake_and_commands():
    store = SessionStore()
    engines = _engines(store)
    original = store.active_id

    async def go() -> None:
        ws = _CmdWS(
            [
                json.dumps({"type": "session"}),  # create a new one
                json.dumps({"type": "session", "id": original}),  # switch back
                json.dumps({"type": "session", "id": "nope"}),  # unknown
                json.dumps({"type": "wake"}),  # no-op: fake engines have no phrase
            ]
        )
        await pipeline.serve_session(ws, engines)
        for _ in range(10):  # drain send tasks scheduled on this loop
            await asyncio.sleep(0)

        msgs = [m for m in ws.sent if isinstance(m, dict)]
        # handshake: session frame first, then the config frame, then wake state
        assert msgs[0] == {"type": "session", "id": original}
        assert msgs[1]["type"] == "config"
        assert msgs[2] == {"type": "wake", "active": False}
        sess = [m for m in msgs if m["type"] == "session"]
        created = [m for m in sess if m.get("created")]
        assert len(created) == 1
        new_id = created[0]["id"]
        assert [m["id"] for m in sess] == [original, new_id, original]
        assert [m.get("created") for m in sess] == [None, True, None]
        errors = [m for m in msgs if m["type"] == "error"]
        assert len(errors) == 1 and "nope" in errors[0]["message"]
        # the last real command switched back: the store agrees
        assert store.active_id == original

    asyncio.run(go())


def test_serve_session_pins_query_param():
    store = SessionStore()
    engines = _engines(store)
    original = store.active_id
    other = store.create()

    async def go() -> None:
        ws = _CmdWS([], query_params={"session": original})
        await pipeline.serve_session(ws, engines)
        for _ in range(10):  # drain send tasks scheduled on this loop
            await asyncio.sleep(0)
        first = next(m for m in ws.sent if isinstance(m, dict))
        assert first == {"type": "session", "id": original}

    asyncio.run(go())


# ---------------- /api/sessions (offline) ----------------

def _offline_app() -> FastAPI:
    app = FastAPI()
    app.add_api_route("/api/sessions", main.list_sessions, methods=["GET"])
    app.add_api_route(
        "/api/sessions/{session_id}/transcript",
        main.get_session_transcript,
        methods=["GET"],
    )
    app.add_api_route("/api/sessions", main.create_session, methods=["POST"])
    app.add_api_route(
        "/api/sessions/{session_id}/activate", main.activate_session, methods=["POST"]
    )
    app.add_api_route(
        "/api/sessions/{session_id}/rename", main.rename_session, methods=["POST"]
    )
    app.add_api_route(
        "/api/sessions/{session_id}", main.delete_session, methods=["DELETE"]
    )
    app.state.engines = SimpleNamespace(sessions=SessionStore())
    return app


def test_sessions_api():
    client = TestClient(_offline_app())

    j = client.get("/api/sessions").json()
    active = j["active"]
    assert len(j["sessions"]) == 1
    assert j["sessions"][0]["id"] == active
    assert j["sessions"][0]["active"] is True

    t = client.get(f"/api/sessions/{active}/transcript")
    assert t.status_code == 200
    assert t.json() == {"id": active, "summary": None, "turns": []}

    store = client.app.state.engines.sessions
    store.conversation_for(active).add_turn("hello", "hi")
    t = client.get(f"/api/sessions/{active}/transcript")
    assert t.status_code == 200
    assert t.json()["turns"] == [{"user": "hello", "assistant": "hi"}]

    r = client.post(f"/api/sessions/{active}/rename", json={"name": "Daily standup"})
    assert r.status_code == 200
    assert r.json()["name"] == "Daily standup"
    j = client.get("/api/sessions").json()
    row = [s for s in j["sessions"] if s["id"] == active][0]
    assert row["name"] == "Daily standup"

    r = client.post("/api/sessions")
    assert r.status_code == 200
    new_id = r.json()["session_id"]
    assert r.json()["active"] == new_id

    r = client.post(f"/api/sessions/{active}/activate")
    assert r.status_code == 200
    assert r.json()["active"] == active

    r = client.delete(f"/api/sessions/{new_id}")
    assert r.status_code == 200
    assert r.json()["active"] == active

    # deleting the last session creates a fresh active one
    r = client.delete(f"/api/sessions/{active}")
    assert r.status_code == 200
    fresh = r.json()["active"]
    assert fresh not in (active, new_id)
    assert client.get("/api/sessions").json()["active"] == fresh

    assert client.post("/api/sessions/nope/activate").status_code == 404
    assert client.post("/api/sessions/nope/rename", json={"name": "x"}).status_code == 404
    assert client.delete("/api/sessions/nope").status_code == 404
    assert client.get("/api/sessions/nope/transcript").status_code == 404
