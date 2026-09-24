"""Tools for the smol agent: get_time, weather (open-meteo, geocoded
place names), file tools (sandboxed to WORK_DIR), shell exec (app/shell.py),
web search/fetch (app/web.py).

get_time and weather honour the [user] profile (T023): the user's time zone
for time answers, and their location + units for weather."""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app import config
from app.memory import MemoryStore
from app import shell as shell_tool
from app.skills import SkillStore
from app import web as web_tool
from app.reminders import get_default_store, user_tz

MAX_RESULT_CHARS = 16000
IGNORED_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".cache",
    "target",
}
LIST_LIMIT = 200

WMO_CODES = {
    0: "clear sky",
    1: "mainly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with light hail",
    99: "a thunderstorm with heavy hail",
}

MAX_READ_CHARS = 4000
HTTP_RETRIES = 1


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Minimal nanobot-style registry for tool discovery and validation."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def execute(self, name: str, args: dict | None) -> str:
        tool = self.get(name)
        if tool is None:
            return f"error: unknown tool: {name}"

        payload = args if isinstance(args, dict) else {}
        params = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        missing = [key for key in required if key not in payload]
        if missing:
            return "error: missing required parameter(s): " + ", ".join(missing)

        kwargs: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in params:
                if params[key].get("type") == "integer" and value is not None and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        return f"error: invalid parameter '{key}': expected integer"
                if params[key].get("type") == "number" and value is not None and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        return f"error: invalid parameter '{key}': expected number"
                if params[key].get("type") == "string" and value is not None and not isinstance(value, str):
                    value = str(value)
                kwargs[key] = value

        try:
            return str(tool.func(**kwargs))
        except TypeError as exc:
            return f"error: invalid arguments: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @property
    def tool_names(self) -> List[str]:
        return sorted(self._tools)

    def get_definitions(self) -> List[dict]:
        return [tool.to_schema() for tool in sorted(self._tools.values(), key=lambda tool: tool.name)]


def get_time() -> str:
    """Current date/time in the user's configured time zone (T023); falls back
    to the local (container) time zone when none is set or the name is bad."""
    tz_name = config.USER_TIMEZONE.strip()
    tz = None
    if tz_name:
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            tz = None
    now = datetime.datetime.now(tz) if tz else datetime.datetime.now().astimezone()
    label = tz_name if tz else (now.tzname() or "local time")
    return now.strftime(f"It is %A, %B %d, %Y, %H:%M in {label}.")


_GEO_CACHE: dict[str, tuple[float, float]] = {}


def _http_get_with_retry(url: str, **kwargs) -> httpx.Response:
    """Retry only transient network and server failures once."""
    last_error = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = httpx.get(url, **kwargs)
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < HTTP_RETRIES:
                continue
            raise
        if getattr(response, "status_code", 200) >= 500 and attempt < HTTP_RETRIES:
            continue
        return response
    raise last_error  # pragma: no cover - loop always returns or raises


def _geocode(place: str) -> tuple[float, float] | None:
    """Place name -> (lat, lon) via open-meteo's free geocoding API (no key).
    Successful lookups are cached for the process lifetime."""
    key = place.strip().lower()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    r = _http_get_with_retry(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place.strip(), "count": 1},
        timeout=10.0,
    )
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    coords = (float(results[0]["latitude"]), float(results[0]["longitude"]))
    _GEO_CACHE[key] = coords
    return coords


def weather(location: str = "") -> str:
    """Current weather for `location` (a place name), or the user's default
    location from the [user] profile when omitted. Units follow the profile
    (metric: °C/km/h, imperial: °F/mph)."""
    place = location.strip() or config.USER_LOCATION.strip()
    if not place:
        return (
            "error: no location given and no default location is set — ask the "
            "user where they are, or set their location in the settings"
        )
    coords = _geocode(place)
    if coords is None:
        return f"error: could not find a place called {place!r} — try a different name"
    imperial = config.USER_UNITS.strip().lower() == "imperial"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "current": (
            "temperature_2m,apparent_temperature,weather_code,"
            "wind_speed_10m,relative_humidity_2m"
        ),
        "timezone": "auto",
    }
    if imperial:
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
    r = _http_get_with_retry("https://api.open-meteo.com/v1/forecast", params=params, timeout=15.0)
    r.raise_for_status()
    cur = r.json()["current"]
    desc = WMO_CODES.get(int(cur["weather_code"]), "unknown conditions")
    temp_unit = "degrees Fahrenheit" if imperial else "degrees Celsius"
    wind_unit = "miles per hour" if imperial else "kilometers per hour"
    return (
        f"It is {round(cur['temperature_2m'])} {temp_unit}, feels like "
        f"{round(cur['apparent_temperature'])} {temp_unit}, with {desc}, wind "
        f"{round(cur['wind_speed_10m'])} {wind_unit}, and "
        f"{round(cur['relative_humidity_2m'])} percent humidity."
    )


def _resolve(path: str) -> str | None:
    """Path inside the workspace root, or None."""
    base = os.path.realpath(config.WORK_DIR)
    target = os.path.realpath(os.path.join(base, path))
    if target != base and not target.startswith(base + os.sep):
        return None
    return target


def read_file(path: str) -> str:
    target = _resolve(str(path))
    if target is None:
        return "error: path is outside the workspace"
    if not os.path.isfile(target):
        return f"error: no such file: {path}"
    with open(target, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_READ_CHARS)
    return text.strip() or "(empty file)"


def read_memory() -> str:
    """Read the detailed local-memory archive outside the workspace sandbox."""
    return MemoryStore(os.path.join(config.DATA_DIR, "memory.md")).archive_text()


def remember(fact: str) -> str:
    """Persist a user-approved fact in the local memory store."""
    fact = str(fact).strip()
    if not fact:
        return "error: memory fact is required"
    MemoryStore(os.path.join(config.DATA_DIR, "memory.md")).observe(fact)
    return "memory saved"


def _skills() -> SkillStore:
    return SkillStore(os.path.join(config.DATA_DIR, "skills"))


def list_skills() -> str:
    skills = _skills().list()
    if not skills:
        return "(no skills)"
    return "\n".join(f"{item['name']}: {item['description']}" for item in skills)


def read_skill(name: str) -> str:
    return _skills().read(name)


def save_skill(name: str, description: str, instructions: str) -> str:
    action = _skills().save(name, description, instructions)
    return f"skill {action}: {name}"


def delete_skill(name: str) -> str:
    _skills().delete(name)
    return f"skill deleted: {name}"


def write_file(path: str, content: str) -> str:
    target = _resolve(str(path))
    if target is None:
        return "error: path is outside the workspace"
    if os.path.dirname(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(str(content))
    return f"wrote {len(content)} chars to {os.path.relpath(target, os.path.realpath(config.WORK_DIR))}"


def list_dir(path: str = "") -> str:
    target = _resolve(str(path or "."))
    if target is None:
        return "error: path is outside the workspace"
    if not os.path.isdir(target):
        return f"error: no such directory: {path}"
    try:
        entries = sorted(e for e in os.listdir(target) if e not in IGNORED_DIRS)
    except OSError as e:
        return f"error: {e}"
    if not entries:
        return "(empty directory)"
    lines = [
        f"{e}/" if os.path.isdir(os.path.join(target, e)) else e for e in entries[:LIST_LIMIT]
    ]
    note = f"\n({LIST_LIMIT} of {len(entries)} entries)" if len(entries) > LIST_LIMIT else ""
    return "\n".join(lines) + note


TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": (
                "Get the current weather. With no location it uses the user's "
                "default location from their profile; pass a place name for "
                "another place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Place name, e.g. \"Paris\". Omit for the user's location.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_conversations",
            "description": "List saved conversations with their IDs, names, turn counts, and active status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_conversation",
            "description": "Create and switch to a new conversation. Optionally give it a name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_conversation",
            "description": "Switch to a saved conversation by ID. Use list_conversations first to find IDs.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_conversation",
            "description": "Rename a saved conversation. Use id 'current' for the current conversation.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                "required": ["id", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_conversation",
            "description": "Delete a saved conversation only when the user explicitly asks. Use id 'current' for the current conversation.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace. Path is relative to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read the complete local memory when the user asks what you remember.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a durable fact only when the user explicitly asks you to remember it, "
                "such as a preference or an ongoing project detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List available task-specific skills and their descriptions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Load the full instructions for a named skill when it is relevant.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_skill",
            "description": (
                "Create or update a task-specific skill when the user asks. "
                "Use a concise description and complete Markdown instructions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Lowercase hyphenated skill name."},
                    "description": {"type": "string", "description": "One-line selection hint."},
                    "instructions": {"type": "string", "description": "Full Markdown instructions."},
                },
                "required": ["name", "description", "instructions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Delete a named skill only when the user explicitly asks.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a text file in the workspace. "
                "Path is relative to the workspace root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in the workspace (default: the root).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": (
                "Run a shell command in the workspace (bash). Use for builds, "
                "git, scripts, and browser automation with agent-browser. "
                "For browser work, open a page, snapshot interactive refs, "
                "act by ref, and re-snapshot after changes. The user can watch "
                "the browser panel. Keep commands short; output is truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_dir": {
                        "type": "string",
                        "description": "Optional subdirectory of the workspace.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60, max 120).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_view",
            "description": (
                "Open or close the live browser viewport in the user's UI. "
                "Call with action 'open' before starting agent-browser work so "
                "the user can watch, and action 'close' when the user asks to "
                "close the browser or the browsing task is finished (this also "
                "shuts the browser down and returns the UI to the voice view)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "close"],
                        "description": "'open' shows the viewport, 'close' closes it and the browser.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (DuckDuckGo). Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer", "description": "Results to return (1-5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and return its readable text/markdown. Use to read "
                "a specific page found via web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "description": "Max characters to return."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": (
                "Create a reminder or scheduled task. type: 'reminder' (vivo tells "
                "the user something at the set time) or 'task' (you carry it out "
                "with your tools and return the result, e.g. a daily news "
                "briefing). response: 'spoken' (said aloud and shown as text) or "
                "'text' (shown as text only). repeat: 'once' (needs at or "
                "in_minutes), 'interval' (needs every_minutes; fires repeatedly "
                "every N minutes, multiple times a day), 'daily' (needs time), "
                "'weekly' (needs time and days). Times are in the user's own time "
                "zone; use get_time first to resolve words like 'tomorrow' or "
                "'tonight' to a concrete ISO datetime in that zone. session_id "
                "optionally attaches the item to a conversation (see "
                "list_conversations). Only call this once every schedule detail "
                "is known; if anything is missing, ask the user first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "What to remind about, or the task to carry out."},
                    "type": {"type": "string", "enum": ["reminder", "task"], "description": "'reminder' to tell the user something, 'task' to carry it out with your tools and report back."},
                    "response": {"type": "string", "enum": ["spoken", "text"], "description": "'spoken' says the result aloud (and shows it), 'text' shows it as text only."},
                    "repeat": {"type": "string", "enum": ["once", "interval", "daily", "weekly"]},
                    "at": {"type": "string", "description": "ISO datetime for a one-time item, in the user's time zone (see get_time), e.g. '2026-09-24T10:00:00'."},
                    "in_minutes": {"type": "number", "description": "Delay in minutes for a one-time item."},
                    "every_minutes": {"type": "integer", "description": "Repeat span in minutes for an 'interval' item, 1-1440, e.g. 30 for every half hour or 120 for every 2 hours."},
                    "time": {"type": "string", "description": "Clock time 'HH:MM' for daily or weekly items, e.g. '09:00'."},
                    "days": {"type": "array", "items": {"type": "integer"}, "description": "Weekdays for a weekly item: 0=Monday..6=Sunday."},
                    "session_id": {"type": "string", "description": "Optional conversation ID to attach the item to."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": (
                "List the user's reminders and scheduled tasks with their IDs, types, "
                "schedules, next fire times (in the user's time zone), and (for tasks) "
                "the last response. Use it to answer what is scheduled and to find IDs "
                "for delete_reminder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of items to return."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_reminder",
            "description": (
                "Delete a reminder or scheduled task by its ID. Use list_reminders "
                "first to find IDs. Only use when the user asks to cancel or remove one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The reminder ID from list_reminders."},
                },
                "required": ["id"],
            },
        },
    },
]


def _format_every(minutes) -> str:
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        return "?"
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60} h"
    return f"{minutes} min"


def _local_when(value) -> str:
    """An iso datetime (stored as UTC) as the user's wall clock, so the LLM —
    which thinks in the user's time zone — sees the time they asked for."""
    try:
        dt = datetime.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(user_tz()).strftime("%Y-%m-%d %H:%M")


def execute(name: str, args: dict) -> str:
    registry = ToolRegistry()
    for schema in TOOLS:
        fn = schema["function"]
        tool_name = fn["name"]
        if tool_name == "get_time":
            func = lambda: get_time()
        elif tool_name == "weather":
            func = lambda location="": weather(str(location))
        elif tool_name == "read_file":
            func = lambda path: read_file(str(path))
        elif tool_name == "read_memory":
            func = lambda: read_memory()
        elif tool_name == "remember":
            func = lambda fact: remember(str(fact))
        elif tool_name == "list_skills":
            func = lambda: list_skills()
        elif tool_name == "read_skill":
            func = lambda name: read_skill(str(name))
        elif tool_name == "save_skill":
            func = lambda name, description, instructions: save_skill(
                str(name), str(description), str(instructions)
            )
        elif tool_name == "delete_skill":
            func = lambda name: delete_skill(str(name))
        elif tool_name == "write_file":
            func = lambda path, content: write_file(str(path), str(content))
        elif tool_name == "list_dir":
            func = lambda path="": list_dir(str(path))
        elif tool_name == "exec":
            func = lambda command="", working_dir=None, timeout=None: shell_tool.run_shell(
                str(command),
                working_dir=working_dir,
                timeout=timeout,
            )
        elif tool_name == "web_search":
            func = lambda query="", count=None: web_tool.web_search(str(query), count=count)
        elif tool_name == "web_fetch":
            func = lambda url="", max_chars=None: web_tool.web_fetch(str(url), max_chars=max_chars)
        elif tool_name == "set_reminder":
            def func(text="", type=None, response=None, repeat=None, at=None, in_minutes=None, every_minutes=None, time=None, days=None, session_id=None):
                if not str(text or "").strip():
                    return "error: reminder text is required"
                item_type = str(type or "reminder")
                item_response = str(response or "spoken")
                repeat = str(repeat or "once")
                spec = {"text": str(text), "type": item_type, "response": item_response, "repeat": repeat, "session_id": session_id or None}
                if repeat == "once":
                    if in_minutes is not None and at is not None:
                        return "error: use either in_minutes or at, not both"
                    if in_minutes is not None:
                        due = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=float(in_minutes))
                        spec["at"] = due.isoformat(timespec="seconds")
                    elif at is not None:
                        spec["at"] = str(at)
                    else:
                        return (
                            "error: a one-time reminder needs 'at' (an ISO datetime) "
                            "or 'in_minutes' — ask the user when they want it"
                        )
                elif repeat == "interval":
                    if every_minutes is None:
                        return (
                            "error: an interval item needs 'every_minutes' "
                            "(1-1440) — ask the user how often it should repeat"
                        )
                    spec["every"] = int(every_minutes)
                else:
                    if not time:
                        return (
                            "error: a daily or weekly item needs a 'time' like '09:00' "
                            "— ask the user what time they want it"
                        )
                    spec["time"] = str(time)
                    if repeat == "weekly" and not days:
                        return (
                            "error: a weekly item needs 'days' (0=Monday..6=Sunday) "
                            "— ask the user which days"
                        )
                    if repeat == "weekly":
                        spec["days"] = [int(d) for d in days]
                try:
                    item = get_default_store().create(spec)
                except ValueError as exc:
                    return f"error: {exc}"
                return f"reminder created: {item['text']} ({item['type']}, {item['response']}, {repeat}) — next fire {_local_when(item['due_at'])}"
        elif tool_name == "list_reminders":
            def func(limit=10):
                limit = int(limit or 10)
                items = get_default_store().list_all()[:limit]
                if not items:
                    return "No reminders or tasks."
                lines = []
                for item in items:
                    if item["repeat"] == "once":
                        schedule = f"once {_local_when(item['at'] or item['due_at'])}"
                    elif item["repeat"] == "interval":
                        schedule = f"every {_format_every(item.get('every'))}"
                    elif item["repeat"] == "weekly":
                        schedule = f"weekly {item['time']} on days {','.join(str(d) for d in item['days'])}"
                    else:
                        schedule = f"daily {item['time']}"
                    when = (
                        f"next fire {_local_when(item['due_at'])}" if item["status"] == "pending"
                        else f"fired {_local_when(item['fired_at']) if item['fired_at'] else 'earlier'}"
                    )
                    line = f"- {item['id']}: {item['text']} ({item['type']}/{item['response']}, {schedule}, {when})"
                    if item.get("last_fired_at"):
                        line += f" [last fired {_local_when(item['last_fired_at'])}]"
                    if item.get("last_response"):
                        line += f" | last response: {str(item['last_response'])[:200]}"
                    lines.append(line)
                return "\n".join(lines)
        elif tool_name == "delete_reminder":
            def func(id=""):
                store = get_default_store()
                try:
                    item = store.get(str(id))
                except KeyError:
                    return "error: no reminder with that ID — use list_reminders to see the current ones"
                store.delete(str(id))
                return f"deleted: {item['text']}"
        else:
            continue
        registry.register(
            ToolDefinition(
                name=tool_name,
                description=fn["description"],
                parameters=fn["parameters"],
                func=func,
            )
        )

    result = registry.execute(name, args)
    if result.startswith("error:"):
        return result
    return _cap_result(result)


def _cap_result(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    half = MAX_RESULT_CHARS // 2
    return (
        text[:half]
        + f"\n\n... ({len(text) - MAX_RESULT_CHARS:,} chars truncated) ...\n\n"
        + text[-half:]
    )
