"""Settings schema: the single source of truth for every vivo.toml key (T019).

Drives three things:
- the web settings pane (static/app.js renders its controls from `SCHEMA`)
- validation of POST /api/config (`validate` / `coerce`)
- the comments written into vivo.toml on save (`dump_toml`)

Per-key spec fields:
  type     str | textarea | bool | int | float | str[] | voices
  label    short display name in the pane
  help     shown in the pane's (?) tooltip
  apply    now | next | restart — when a saved value takes effect
  attr     app.config module attribute holding the value (consistency tests)
  min/max  bounds for numeric types
  step     slider step for numeric types
  choices  static dropdown values (for `str` with a fixed set)
"""

SCHEMA = {
    "llm": {
        "title": "Language model",
        "keys": {
            "base_url": {
                "type": "str",
                "label": "Base URL",
                "help": "OpenAI-compatible endpoint serving the chat model (the llama.cpp server).",
                "apply": "now",
                "attr": "LLM_BASE_URL",
            },
            "model": {
                "type": "str",
                "label": "Model",
                "help": "Model name the LLM server serves, e.g. qwen3.8-27b.",
                "apply": "now",
                "attr": "LLM_MODEL",
            },
            "thinking": {
                "type": "bool",
                "label": "Thinking mode",
                "help": "Let the model reason before answering. Adds latency but improves answers, "
                        "especially with tools. vivo speaks filler phrases while it thinks.",
                "apply": "now",
                "attr": "LLM_THINKING",
            },
            "max_tokens": {
                "type": "int",
                "label": "Max tokens",
                "help": "Generation budget per reply (thinking + answer together). Too low truncates "
                        "answers; too high wastes latency.",
                "apply": "now",
                "min": 64,
                "max": 4096,
                "step": 32,
                "attr": "LLM_MAX_TOKENS",
            },
        },
    },
    "persona": {
        "title": "Persona & prompt",
        "keys": {
            "blurb": {
                "type": "textarea",
                "label": "Persona blurb",
                "help": "One or two sentences describing who vivo is. Prepended to the system prompt.",
                "apply": "now",
                "attr": "PERSONA",
            },
            "system_prompt": {
                "type": "textarea",
                "label": "System prompt",
                "help": "Voice-UX instructions appended to the persona: how to speak, when to use "
                        "tools, how to summarise results.",
                "apply": "now",
                "attr": "SYSTEM_PROMPT",
            },
        },
    },
    "user": {
        "title": "User profile",
        "keys": {
            "name": {
                "type": "str",
                "label": "Your name",
                "help": "vivo addresses you by this name.",
                "apply": "now",
                "attr": "USER_NAME",
            },
            "location": {
                "type": "str",
                "label": "Location",
                "help": "Your place, e.g. \"Bristol, UK\". Default for weather questions; "
                        "ask about other places by name.",
                "apply": "now",
                "attr": "USER_LOCATION",
            },
            "timezone": {
                "type": "str",
                "label": "Time zone",
                "help": "IANA time zone, e.g. \"Europe/London\". Used for time questions; "
                        "empty means the container's time zone (UTC).",
                "apply": "now",
                "attr": "USER_TIMEZONE",
            },
            "units": {
                "type": "str",
                "label": "Units",
                "help": "Units in weather answers: metric is °C and km/h, imperial is °F and mph.",
                "apply": "now",
                "choices": ["metric", "imperial"],
                "attr": "USER_UNITS",
            },
        },
    },
    "filler": {
        "title": "Thinking fillers",
        "keys": {
            "first_after_s": {
                "type": "float",
                "label": "First filler after (s)",
                "help": "How long vivo waits without speakable text before starting to fill the "
                        "silence with a phrase.",
                "apply": "now",
                "min": 0.0,
                "max": 30.0,
                "step": 0.5,
                "attr": "THINK_FILLER_FIRST_AFTER",
            },
            "interval_s": {
                "type": "float",
                "label": "Filler interval (s)",
                "help": "Gap between repeated filler phrases while the model keeps thinking.",
                "apply": "now",
                "min": 1.0,
                "max": 60.0,
                "step": 1.0,
                "attr": "THINK_FILLER_INTERVAL",
            },
            "phrases": {
                "type": "str[]",
                "label": "Filler phrases",
                "help": "Short phrases vivo speaks while thinking, one per line. Keep them brief — "
                        "each one is synthesised separately.",
                "apply": "now",
                "attr": "FILLER_PHRASES",
            },
        },
    },
    "voice": {
        "title": "Voice (TTS)",
        "keys": {
            "tts_voice": {
                "type": "voices",
                "label": "Voice (clone)",
                "help": "The voice vivo speaks with: a reference clip (3-30 s of clear speech) "
                        "that is cloned. Upload or record one below.",
                "apply": "now",
                "attr": "TTS_VOICE",
            },
            "tts_speed": {
                "type": "float",
                "label": "Speed",
                "help": "Speech-rate multiplier applied to the cloned voice: 1.0 is the model's "
                        "natural pace, higher is faster.",
                "apply": "now",
                "min": 0.5,
                "max": 2.0,
                "step": 0.05,
                "attr": "TTS_SPEED",
            },
            "sentence_pause_s": {
                "type": "float",
                "label": "Sentence pause (s)",
                "help": "Silence appended after each spoken sentence.",
                "apply": "now",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "attr": "TTS_SENTENCE_PAUSE",
            },
            "sentence_max_chars": {
                "type": "int",
                "label": "Max sentence chars",
                "help": "Hard split for long unpunctuated text. Shorter means the first audio arrives "
                        "sooner at the cost of choppier phrasing.",
                "apply": "next",
                "min": 40,
                "max": 400,
                "step": 10,
                "attr": "SENTENCE_MAX_CHARS",
            },
            "clause_max_chars": {
                "type": "int",
                "label": "Clause split (chars)",
                "help": "Split at clause punctuation (comma, semicolon, colon) once a sentence exceeds "
                        "this length, so TTS can start before the whole sentence is generated. 0 "
                        "disables it. Lower cuts first-audio latency; higher sounds smoother.",
                "apply": "next",
                "min": 0,
                "max": 200,
                "step": 5,
                "attr": "CLAUSE_MAX_CHARS",
            },
            "tts_queue_size": {
                "type": "int",
                "label": "TTS queue size",
                "help": "Sentences buffered between the LLM and TTS. Bigger hides TTS hiccups, "
                        "smaller cuts latency.",
                "apply": "next",
                "min": 1,
                "max": 8,
                "step": 1,
                "attr": "TTS_QUEUE_SIZE",
            },
        },
    },
    "stt": {
        "title": "Speech-to-text",
        "keys": {
            "model": {
                "type": "str",
                "label": "Model size",
                "help": "faster-whisper model. Bigger is more accurate but slower to transcribe and "
                        "slower to load.",
                "apply": "restart",
                "choices": ["tiny", "base", "small", "medium", "large-v3"],
                "attr": "STT_MODEL",
            },
            "compute_type": {
                "type": "str",
                "label": "Compute type",
                "help": "Precision for inference. int8 is the fast CPU default; float16 and "
                        "int8_float16 need a GPU.",
                "apply": "restart",
                "choices": ["int8", "int8_float16", "float16", "float32"],
                "attr": "STT_COMPUTE_TYPE",
            },
            "cpu_threads": {
                "type": "int",
                "label": "CPU threads",
                "help": "Threads used for transcription.",
                "apply": "restart",
                "min": 1,
                "max": 32,
                "step": 1,
                "attr": "STT_CPU_THREADS",
            },
            "language": {
                "type": "str",
                "label": "Language",
                "help": "Transcription language code (en, de, fr, …).",
                "apply": "now",
                "attr": "STT_LANGUAGE",
            },
            "beam_size": {
                "type": "int",
                "label": "Beam size",
                "help": "Search width: 1 (greedy) is fastest; higher is slightly more accurate.",
                "apply": "now",
                "min": 1,
                "max": 5,
                "step": 1,
                "attr": "STT_BEAM_SIZE",
            },
        },
    },
    "vad": {
        "title": "Voice activity detection",
        "keys": {
            "threshold": {
                "type": "float",
                "label": "Speech threshold",
                "help": "Level above which audio counts as speech. Raise it if ambient noise "
                        "triggers vivo, lower it if she misses quiet speech.",
                "apply": "next",
                "min": 0.1,
                "max": 0.9,
                "step": 0.05,
                "attr": "VAD_THRESHOLD",
            },
            "min_speech_ms": {
                "type": "int",
                "label": "Min speech (ms)",
                "help": "Shortest burst of sound kept as speech; shorter blips are ignored.",
                "apply": "next",
                "min": 50,
                "max": 2000,
                "step": 50,
                "attr": "VAD_MIN_SPEECH_MS",
            },
            "min_silence_ms": {
                "type": "int",
                "label": "Min silence (ms)",
                "help": "Silence that puts an utterance on pause.",
                "apply": "next",
                "min": 100,
                "max": 3000,
                "step": 50,
                "attr": "VAD_MIN_SILENCE_MS",
            },
            "reopen_ms": {
                "type": "int",
                "label": "Reopen window (ms)",
                "help": "Extra silence an utterance survives: speech resuming within this window "
                        "continues the same utterance (breath pauses). 0 disables it.",
                "apply": "next",
                "min": 0,
                "max": 3000,
                "step": 50,
                "attr": "VAD_REOPEN_MS",
            },
            "speech_pad_ms": {
                "type": "int",
                "label": "Speech padding (ms)",
                "help": "Context added around each captured utterance.",
                "apply": "next",
                "min": 0,
                "max": 1000,
                "step": 50,
                "attr": "VAD_SPEECH_PAD_MS",
            },
            "max_speech_s": {
                "type": "float",
                "label": "Max speech (s)",
                "help": "Hard cap on a single utterance; longer speech is cut off and processed.",
                "apply": "next",
                "min": 5.0,
                "max": 120.0,
                "step": 5.0,
                "attr": "VAD_MAX_SPEECH_S",
            },
        },
    },
    "barge_in": {
        "title": "Barge-in (UI)",
        "keys": {
            "level_threshold": {
                "type": "float",
                "label": "Mic level threshold",
                "help": "While vivo is speaking, sustained mic level above this interrupts her. The "
                        "browser's echo cancellation has already removed her own voice.",
                "apply": "now",
                "min": 0.05,
                "max": 0.9,
                "step": 0.05,
                "attr": "BARGE_LEVEL_THRESHOLD",
            },
            "sustain_ms": {
                "type": "int",
                "label": "Sustain (ms)",
                "help": "How long the mic level must stay above the threshold before it interrupts.",
                "apply": "now",
                "min": 50,
                "max": 2000,
                "step": 50,
                "attr": "BARGE_SUSTAIN_MS",
            },
            "cooldown_ms": {
                "type": "int",
                "label": "Cooldown (ms)",
                "help": "Grace period after a barge-in before another one can trigger.",
                "apply": "now",
                "min": 0,
                "max": 5000,
                "step": 50,
                "attr": "BARGE_COOLDOWN_MS",
            },
        },
    },
    "wake": {
        "title": "Wake phrase",
        "keys": {
            "phrase": {
                "type": "str",
                "label": "Wake phrase",
                "help": "Say this to wake vivo: she listens silently for it and ignores everything "
                        "else, so ambient speech doesn't flood the transcript. Anything after the "
                        "phrase in the same utterance is answered too. An empty phrase disables the "
                        "wake word (she answers every utterance).",
                "apply": "now",
                "attr": "WAKE_PHRASE",
            },
            "session_timeout_s": {
                "type": "float",
                "label": "Session timeout (s)",
                "help": "Once awake, vivo returns to silent listening after you stay quiet this long. "
                        "Any exchange restarts the clock; it never fires while she is replying.",
                "apply": "now",
                "min": 5.0,
                "max": 600.0,
                "step": 5.0,
                "attr": "WAKE_SESSION_TIMEOUT_S",
            },
            "end_phrases": {
                "type": "str[]",
                "label": "End phrases",
                "help": "Saying one of these (one per line) ends the session early: vivo confirms and "
                        "returns to silent listening.",
                "apply": "now",
                "attr": "WAKE_END_PHRASES",
            },
            "ack": {
                "type": "str",
                "label": "Wake acknowledgement",
                "help": "What vivo says right after the wake phrase when nothing else was said.",
                "apply": "now",
                "attr": "WAKE_ACK",
            },
            "goodnight": {
                "type": "str",
                "label": "Goodnight",
                "help": "What vivo says when a session ends (timeout or end phrase).",
                "apply": "now",
                "attr": "WAKE_GOODNIGHT",
            },
        },
    },
    "memory": {
        "title": "Conversation memory",
        "keys": {
            "compact_after_chars": {
                "type": "int",
                "label": "Compact after (chars)",
                "help": "Once the conversation history grows past this, older turns are summarised "
                        "to keep the context small.",
                "apply": "now",
                "min": 1000,
                "max": 100000,
                "step": 500,
                "attr": "COMPACT_AFTER_CHARS",
            },
            "keep_recent_turns": {
                "type": "int",
                "label": "Keep recent turns",
                "help": "How many recent turns stay verbatim during compaction.",
                "apply": "now",
                "min": 1,
                "max": 20,
                "step": 1,
                "attr": "KEEP_RECENT_TURNS",
            },
            "dream_interval_s": {
                "type": "int",
                "label": "Dream interval (s)",
                "help": "How often the background pass reviews local memory and keeps only high-value facts.",
                "apply": "now",
                "min": 60,
                "max": 86400,
                "step": 60,
                "attr": "DREAM_INTERVAL_S",
            },
        },
    },
    "agent": {
        "title": "Agent tools",
        "keys": {
            "max_tool_rounds": {
                "type": "int",
                "label": "Max tool rounds",
                "help": "How many tool-call rounds one reply may use before it must end with an "
                        "answer.",
                "apply": "now",
                "min": 1,
                "max": 20,
                "step": 1,
                "attr": "MAX_TOOL_ROUNDS",
            },
            "exec_timeout_s": {
                "type": "int",
                "label": "Exec timeout (s)",
                "help": "Default timeout for shell commands.",
                "apply": "now",
                "min": 5,
                "max": 120,
                "step": 5,
                "attr": "EXEC_TIMEOUT",
            },
            "exec_max_timeout_s": {
                "type": "int",
                "label": "Exec max timeout (s)",
                "help": "Longest command timeout the model may request.",
                "apply": "now",
                "min": 30,
                "max": 600,
                "step": 30,
                "attr": "EXEC_MAX_TIMEOUT",
            },
            "exec_max_output_chars": {
                "type": "int",
                "label": "Exec max output (chars)",
                "help": "Command output longer than this is truncated before it reaches the model.",
                "apply": "now",
                "min": 1000,
                "max": 50000,
                "step": 500,
                "attr": "EXEC_MAX_OUTPUT",
            },
            "search_max_results": {
                "type": "int",
                "label": "Search results",
                "help": "Maximum web-search results returned per query.",
                "apply": "now",
                "min": 1,
                "max": 10,
                "step": 1,
                "attr": "SEARCH_MAX_RESULTS",
            },
            "fetch_max_chars": {
                "type": "int",
                "label": "Fetch max chars",
                "help": "Maximum characters read from a web page.",
                "apply": "now",
                "min": 500,
                "max": 16000,
                "step": 500,
                "attr": "FETCH_MAX_CHARS",
            },
        },
    },
}


def _validate_value(spec: dict, v) -> str | None:
    t = spec["type"]
    if t in ("str", "textarea", "voices"):
        if not isinstance(v, str):
            return "must be a string"
        if t == "voices" and not v.strip():
            return "must not be empty"
        return None
    if t == "bool":
        return None if isinstance(v, bool) else "must be true or false"
    if t in ("int", "float"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return "must be a number"
        if t == "int" and not float(v).is_integer():
            return "must be a whole number"
        if "min" in spec and v < spec["min"]:
            return f"must be at least {spec['min']}"
        if "max" in spec and v > spec["max"]:
            return f"must be at most {spec['max']}"
        return None
    if t == "str[]":
        if (
            not isinstance(v, list)
            or not v
            or not all(isinstance(x, str) and x.strip() for x in v)
        ):
            return "must be a non-empty list of strings"
        return None
    return f"unknown type {t!r}"


def validate(values: dict, voices: tuple = ()) -> list[str]:
    """Validate a (possibly partial) settings payload. Returns a list of errors."""
    if not isinstance(values, dict):
        return ["values must be an object"]
    errors = []
    for section, keys in values.items():
        spec = SCHEMA.get(section)
        if spec is None:
            errors.append(f"{section}: unknown section")
            continue
        if not isinstance(keys, dict):
            errors.append(f"{section}: must be an object")
            continue
        for key, v in keys.items():
            kspec = spec["keys"].get(key)
            if kspec is None:
                errors.append(f"{section}.{key}: unknown key")
                continue
            err = _validate_value(kspec, v)
            if err is None and kspec["type"] == "voices" and v not in voices:
                err = f"{v!r} is not a known voice"
            if err is not None:
                errors.append(f"{section}.{key}: {err}")
    return errors


def coerce(values: dict) -> dict:
    """Normalise a validated payload (integral floats -> int)."""
    out = {}
    for section, keys in values.items():
        spec = SCHEMA.get(section)
        if spec is None or not isinstance(keys, dict):
            continue
        clean = {}
        for key, v in keys.items():
            kspec = spec["keys"].get(key)
            if kspec and kspec["type"] == "int" and isinstance(v, float) and v.is_integer():
                v = int(v)
            clean[key] = v
        out[section] = clean
    return out


# --- TOML writer (used by save; keeps the file documented) ------------------


def _fmt_str(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _fmt(v) -> str:
    if isinstance(v, bool):  # before int: bool is a subclass of int
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _fmt_str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_str(str(x)) for x in v) + "]"
    raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def dump_toml(data: dict) -> str:
    """Render a settings dict as vivo.toml, with a comment per key from SCHEMA."""
    lines = [
        "# vivo settings (T018).",
        "#",
        "# Editable by hand or from the web settings pane (gear button). On save the",
        "# file is regenerated from these values; comments come from the schema.",
        "# Precedence at runtime: environment variable > this file > built-in default.",
    ]
    emitted = set()
    for section, spec in SCHEMA.items():
        if section not in data:
            continue
        lines.append("")
        lines.append(f"[{section}]")
        lines.append(f"# {spec['title']}")
        for key, kspec in spec["keys"].items():
            if key not in data[section]:
                continue
            if kspec.get("help"):
                lines.append(f"# {kspec['help']}")
            lines.append(f"{key} = {_fmt(data[section][key])}")
            emitted.add(key)
    # Safety net: any extra sections/keys not in the schema are written as-is.
    for section, keys in data.items():
        if section in SCHEMA:
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, v in keys.items():
            lines.append(f"{key} = {_fmt(v)}")
    return "\n".join(lines) + "\n"
