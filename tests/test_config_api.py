"""Settings-pane API tests (T019): schema/validation/writer, /api/config, hot-apply, broadcast."""
import asyncio
import json
import threading
import time
import tomllib
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app import config, config_schema, main, pipeline

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["llm", "persona", "filler", "voice", "stt", "vad", "barge_in", "memory", "agent"]


@pytest.fixture()
def api_env(tmp_path, monkeypatch):
    """Point CONFIG_PATH at a temp copy of the shipped vivo.toml.

    Restores the original effective values on teardown (finalizer runs
    before monkeypatch reverts CONFIG_PATH).
    """
    cfg_path = tmp_path / "vivo.toml"
    cfg_path.write_bytes((ROOT / "vivo.toml").read_bytes())
    monkeypatch.setattr(config, "CONFIG_PATH", str(cfg_path))
    config.refresh()
    saved = config.effective_dict()
    yield cfg_path
    config.write_config(saved)
    config.refresh()


# ---------------- schema ----------------

def test_schema_matches_shipped_toml():
    file_toml = tomllib.loads((ROOT / "vivo.toml").read_text())
    file_keys = {s: set(d.keys()) for s, d in file_toml.items()}
    schema_keys = {s: set(spec["keys"]) for s, spec in config_schema.SCHEMA.items()}
    assert file_keys == schema_keys
    assert list(config_schema.SCHEMA) == SECTIONS


def test_schema_attrs_match_effective_values():
    eff = config.effective_dict()
    for sec, spec in config_schema.SCHEMA.items():
        for key, k in spec["keys"].items():
            val = getattr(config, k["attr"])
            if k["type"] == "str[]":
                val = list(val)
            assert eff[sec][key] == val, (sec, key)
            assert k["apply"] in ("now", "next", "restart")


def test_validate_accepts_effective_values():
    # the live voice comes from the user-editable vivo.toml (T019)
    assert config_schema.validate(config.effective_dict(), voices=(config.TTS_VOICE,)) == []


def test_validate_rejects_bad_values():
    bad = {
        "llm": {"max_tokens": "nope", "bogus": 1},
        "vad": {"threshold": 0.0},
        "filler": {"phrases": ["a", "   ", "b"]},
        "voice": {"tts_voice": "nope"},
        "agent": {"exec_timeout_s": 5000},
        "unknown_section": {},
    }
    joined = "; ".join(config_schema.validate(bad, voices=("af_heart",)))
    for frag in (
        "llm.bogus: unknown key",
        "must be a number",
        "must be at least 0.1",
        "non-empty list of strings",
        "not a known voice",
        "must be at most 120",
        "unknown section",
    ):
        assert frag in joined, joined


# ---------------- writer + refresh ----------------

def test_write_config_roundtrip(api_env):
    base = config.effective_dict()
    config.write_config(base)
    config.refresh()
    assert config.effective_dict() == base
    text = api_env.read_text()
    assert "# vivo settings (T018)." in text
    assert "# Language model" in text  # section title from the schema
    assert "# Kokoro voice used for all speech." in text  # per-key help from the schema


def test_write_config_escapes_special_chars(api_env):
    base = config.effective_dict()
    base["persona"]["system_prompt"] = 'He said "hello"\nline two\ttab \\ slash'
    config.write_config(base)
    config.refresh()
    assert config.SYSTEM_PROMPT == base["persona"]["system_prompt"]


def test_refresh_picks_up_edits(api_env):
    d = config.effective_dict()
    d["stt"]["language"] = "de"
    config.write_config(d)
    config.refresh()
    assert config.STT_LANGUAGE == "de"


def test_refresh_tolerates_corrupt_file(api_env):
    api_env.write_text("this is [ not valid toml")
    config.refresh()  # must not raise
    assert config.LLM_MODEL == "qwen3.8-27b"  # built-in default, not a crash


# ---------------- /api/config (offline fakes) ----------------

class _FakeEngines:
    def __init__(self):
        self.agent = type("A", (), {})()
        self.agent.base_url = "http://x/v1"
        self.agent.model = "m"
        self.agent.persona = "p"
        self.agent.system_prompt = "s"
        self.agent.thinking = True
        self.agent.max_tokens = 600
        self.agent.max_tool_rounds = 8
        self.tts = type("T", (), {"voice": "af_heart", "speed": 1.0, "sentence_pause": 0.2})()
        self.tts.voices = lambda: ["af_heart", "am_michael"]
        self.stt = type("S", (), {"language": "en", "beam_size": 1})()
        self.conversation = type("C", (), {"compact_after_chars": 12000, "keep_recent_turns": 4})()


def _offline_app():
    app = FastAPI()
    app.add_api_route("/api/config", main.get_config, methods=["GET"])
    app.add_api_route("/api/config", main.post_config, methods=["POST"])
    app.state.engines = _FakeEngines()
    return app


def test_api_get_returns_values_schema_voices(api_env):
    with TestClient(_offline_app()) as c:
        j = c.get("/api/config").json()
    assert set(j["schema"]) == set(SECTIONS)
    assert j["voices"] == ["af_heart", "am_michael"]
    assert j["values"]["voice"]["tts_voice"] == config.TTS_VOICE
    assert j["schema"]["voice"]["keys"]["tts_voice"]["type"] == "voices"


def test_api_post_validates_writes_and_applies(api_env):
    with TestClient(_offline_app()) as c:
        r = c.post("/api/config", json={"values": {"voice": {"tts_voice": "am_michael", "tts_speed": 1.2}}})
        assert r.status_code == 200, r.text
        assert r.json()["values"]["voice"]["tts_voice"] == "am_michael"
        assert config.TTS_VOICE == "am_michael"
        assert 'tts_voice = "am_michael"' in api_env.read_text()
        assert "tts_speed = 1.2" in api_env.read_text()

        bad = c.post("/api/config", json={"values": {"voice": {"tts_voice": "nope"}}})
        assert bad.status_code == 400
        assert "not a known voice" in str(bad.json()["detail"])

        bad2 = c.post("/api/config", json={"values": {"vad": {"threshold": 0.0}}})
        assert bad2.status_code == 400
        assert "must be at least" in str(bad2.json()["detail"])


# ---------------- broadcast to open sessions ----------------

class _FakeWS:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def accept(self):
        pass

    async def receive_json(self):
        raise RuntimeError("no more messages")

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def send_bytes(self, data):
        self.sent.append(data)


def _wait_for_config(ws, timeout=5.0):
    """send() is thread-safe-async; poll until the frame lands."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cfgs = [m for m in ws.sent if isinstance(m, dict) and m["type"] == "config"]
        if cfgs:
            return cfgs[-1]
        time.sleep(0.02)
    return None


def test_broadcast_config_reaches_open_sessions(api_env):
    engines = _FakeEngines()
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    try:
        started.wait(5)
        ws = _FakeWS()
        session = asyncio.run_coroutine_threadsafe(_make_session(ws, engines), loop).result(10)
        pipeline._ACTIVE.add(session)  # serve_session does this in production
        config.BARGE_LEVEL_THRESHOLD = 0.3  # barge_config_msg() reads it live
        pipeline.broadcast_config()
        msg = _wait_for_config(ws)
        assert msg and msg["barge_in"]["level_threshold"] == 0.3

        session.closed = True  # closed sessions are skipped, not crashed on
        pipeline.broadcast_config()
        time.sleep(0.1)
    finally:
        pipeline._ACTIVE.discard(session)
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()


async def _make_session(ws, engines):
    return pipeline.VoiceSession(ws, engines)


# ---------------- live (real engines) ----------------

def test_api_config_live(api_env):
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
        j = c.get("/api/config").json()
        assert set(j["schema"]) == set(SECTIONS)
        assert "af_heart" in j["voices"]
        assert j["values"]["stt"]["model"] == "small"

        other = [v for v in j["voices"] if v != "af_heart"][0]
        r = c.post("/api/config", json={"values": {"voice": {"tts_voice": other}}})
        assert r.status_code == 200, r.text
        assert config.TTS_VOICE == other
        assert f'tts_voice = "{other}"' in api_env.read_text()

        bad = c.post("/api/config", json={"values": {"vad": {"threshold": 0.0}}})
        assert bad.status_code == 400
