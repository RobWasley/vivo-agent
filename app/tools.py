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
from app import shell as shell_tool
from app import web as web_tool

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


def _geocode(place: str) -> tuple[float, float] | None:
    """Place name -> (lat, lon) via open-meteo's free geocoding API (no key).
    Successful lookups are cached for the process lifetime."""
    key = place.strip().lower()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    r = httpx.get(
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
    r = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15.0)
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
                "git, scripts, and anything else you can do on the machine. "
                "Keep commands short; output is truncated."
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
]


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
