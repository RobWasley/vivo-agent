"""Tools for the smol agent: get_time, weather (open-meteo), file tools
(sandboxed to WORK_DIR), shell exec (app/shell.py), web search/fetch
(app/web.py)."""
from __future__ import annotations

import datetime
import os
from typing import List

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


def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("It is %A, %B %d, %Y, %H:%M.")


def weather(latitude: float, longitude: float) -> str:
    r = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,apparent_temperature,weather_code,"
                "wind_speed_10m,relative_humidity_2m"
            ),
            "timezone": "auto",
        },
        timeout=15.0,
    )
    r.raise_for_status()
    cur = r.json()["current"]
    desc = WMO_CODES.get(int(cur["weather_code"]), "unknown conditions")
    return (
        f"It is {round(cur['temperature_2m'])} degrees, feels like "
        f"{round(cur['apparent_temperature'])}, with {desc}, wind "
        f"{round(cur['wind_speed_10m'])} kilometers per hour, and "
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
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
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
    try:
        if name == "get_time":
            result = get_time()
        elif name == "weather":
            result = weather(float(args["latitude"]), float(args["longitude"]))
        elif name == "read_file":
            result = read_file(str(args["path"]))
        elif name == "write_file":
            result = write_file(str(args["path"]), str(args["content"]))
        elif name == "list_dir":
            result = list_dir(str(args.get("path", "")))
        elif name == "exec":
            result = shell_tool.run_shell(
                str(args.get("command", "")),
                working_dir=args.get("working_dir"),
                timeout=args.get("timeout"),
            )
        elif name == "web_search":
            result = web_tool.web_search(
                str(args.get("query", "")), count=args.get("count")
            )
        elif name == "web_fetch":
            result = web_tool.web_fetch(
                str(args.get("url", "")), max_chars=args.get("max_chars")
            )
        else:
            return f"error: unknown tool: {name}"
    except Exception as e:  # noqa: BLE001 - tool errors become model-visible text
        return f"error: {e}"
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
