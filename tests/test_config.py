"""Config layer (T016): vivo.toml loading and env-var precedence.

app.config is read at import time, so these tests reload the module against a
test-owned CONFIG_PATH and restore the real environment + configuration on
teardown (the suite shares one process).
"""
import importlib
import os
import tomllib
from pathlib import Path

import pytest

from app import config as config_mod

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg(tmp_path):
    """Reload app.config with CONFIG_PATH pointing at an empty tmp dir."""
    saved_env = dict(os.environ)
    os.environ["CONFIG_PATH"] = str(tmp_path / "vivo.toml")
    importlib.reload(config_mod)
    yield config_mod
    os.environ.clear()
    os.environ.update(saved_env)
    importlib.reload(config_mod)  # back to the shipped vivo.toml values


def _write(tmp_path, text: str) -> None:
    (tmp_path / "vivo.toml").write_text(text, encoding="utf-8")
    importlib.reload(config_mod)


def test_missing_file_uses_built_in_defaults(cfg, tmp_path):
    assert cfg.LLM_MODEL == "qwen3.8-27b"
    assert cfg.LLM_THINKING is True
    assert cfg.LLM_MAX_TOKENS == 600
    assert cfg.SENTENCE_MAX_CHARS == 90
    assert cfg.VAD_REOPEN_MS == 600
    assert cfg.FILLER_PHRASES == (
        "Let me think about that.",
        "Working on it.",
        "One moment.",
        "Hmm, give me a second.",
        "Still thinking.",
        "Let me work that out.",
    )
    assert "hands-free voice assistant" in cfg.SYSTEM_PROMPT


def test_file_values_loaded(cfg, tmp_path):
    _write(tmp_path, """
[llm]
model = "custom-9b"
thinking = false
max_tokens = 250

[persona]
blurb = "Custom persona."
system_prompt = "Be very brief."

[filler]
first_after_s = 1.5
interval_s = 3.0
phrases = ["Hmm."]

[voice]
tts_voice = "af_bella"
tts_speed = 1.2
sentence_pause_s = 0.5
sentence_max_chars = 60
tts_queue_size = 4

[stt]
model = "base"
beam_size = 3

[vad]
threshold = 0.4
min_speech_ms = 150
min_silence_ms = 300
reopen_ms = 500
speech_pad_ms = 100
max_speech_s = 20.0

[barge_in]
level_threshold = 0.3
sustain_ms = 200
cooldown_ms = 500

[memory]
compact_after_chars = 9999
keep_recent_turns = 2

[agent]
max_tool_rounds = 5
exec_timeout_s = 30
""")
    assert cfg.LLM_MODEL == "custom-9b"
    assert cfg.LLM_THINKING is False
    assert cfg.LLM_MAX_TOKENS == 250
    assert cfg.PERSONA == "Custom persona."
    assert cfg.SYSTEM_PROMPT == "Be very brief."
    assert cfg.THINK_FILLER_FIRST_AFTER == 1.5
    assert cfg.THINK_FILLER_INTERVAL == 3.0
    assert cfg.FILLER_PHRASES == ("Hmm.",)
    assert cfg.TTS_VOICE == "af_bella"
    assert cfg.TTS_SPEED == 1.2
    assert cfg.TTS_SENTENCE_PAUSE == 0.5
    assert cfg.SENTENCE_MAX_CHARS == 60
    assert cfg.TTS_QUEUE_SIZE == 4
    assert cfg.STT_MODEL == "base"
    assert cfg.STT_BEAM_SIZE == 3
    assert cfg.VAD_THRESHOLD == 0.4
    assert cfg.VAD_MIN_SPEECH_MS == 150
    assert cfg.VAD_MIN_SILENCE_MS == 300
    assert cfg.VAD_REOPEN_MS == 500
    assert cfg.VAD_SPEECH_PAD_MS == 100
    assert cfg.VAD_MAX_SPEECH_S == 20.0
    assert cfg.BARGE_LEVEL_THRESHOLD == 0.3
    assert cfg.BARGE_SUSTAIN_MS == 200
    assert cfg.BARGE_COOLDOWN_MS == 500
    assert cfg.COMPACT_AFTER_CHARS == 9999
    assert cfg.KEEP_RECENT_TURNS == 2
    assert cfg.MAX_TOOL_ROUNDS == 5
    assert cfg.EXEC_TIMEOUT == 30
    # untouched keys keep their defaults
    assert cfg.EXEC_MAX_TIMEOUT == 120


def test_env_overrides_file(cfg, tmp_path):
    _write(tmp_path, '[llm]\nmodel = "custom-9b"\nthinking = true\n')
    os.environ["LLM_MODEL"] = "env-model"
    os.environ["LLM_THINKING"] = "0"
    importlib.reload(config_mod)
    assert cfg.LLM_MODEL == "env-model"
    assert cfg.LLM_THINKING is False


def test_empty_env_counts_as_unset(cfg, tmp_path):
    # compose passes ${VAR:-} (empty string) for unset overrides
    _write(tmp_path, '[llm]\nmodel = "custom-9b"\n')
    os.environ["LLM_MODEL"] = ""
    importlib.reload(config_mod)
    assert cfg.LLM_MODEL == "custom-9b"


def test_bool_env_parsing(cfg, tmp_path):
    os.environ["LLM_THINKING"] = "0"
    importlib.reload(config_mod)
    assert cfg.LLM_THINKING is False
    os.environ["LLM_THINKING"] = "true"
    importlib.reload(config_mod)
    assert cfg.LLM_THINKING is True


def test_list_env_is_comma_separated(cfg, tmp_path):
    os.environ["THINK_FILLER_PHRASES"] = "A, B ,C"
    importlib.reload(config_mod)
    assert cfg.FILLER_PHRASES == ("A", "B", "C")


def test_shipped_vivo_toml_is_complete():
    # vivo.toml is user-editable (settings pane, T019): assert it carries every
    # schema key with schema-valid values, rather than pinning individual
    # choices like voice or model.
    from app import config_schema

    path = REPO_ROOT / "vivo.toml"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    schema_keys = {s: set(spec["keys"]) for s, spec in config_schema.SCHEMA.items()}
    file_keys = {s: set(d.keys()) for s, d in data.items()}
    assert file_keys == schema_keys
    assert config_schema.validate(data, voices=(data["voice"]["tts_voice"],)) == []
