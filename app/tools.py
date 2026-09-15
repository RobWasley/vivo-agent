"""Tools for the smol agent: get_time, weather (open-meteo), read_file
(sandboxed to DATA_DIR)."""
from __future__ import annotations

import datetime
import os
from typing import List

import httpx

from app import config

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


def read_file(path: str, data_dir: str = None) -> str:
    base = os.path.realpath(data_dir or config.DATA_DIR)
    target = os.path.realpath(os.path.join(base, path))
    if target != base and not target.startswith(base + os.sep):
        return "error: path is outside the sandbox"
    if not os.path.isfile(target):
        return f"error: no such file: {path}"
    with open(target, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_READ_CHARS)
    return text.strip() or "(empty file)"


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
            "description": (
                "Read a text file from the sandboxed data directory. "
                "Path is relative to that directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def execute(name: str, args: dict) -> str:
    try:
        if name == "get_time":
            return get_time()
        if name == "weather":
            return weather(float(args["latitude"]), float(args["longitude"]))
        if name == "read_file":
            return read_file(str(args["path"]))
        return f"error: unknown tool: {name}"
    except Exception as e:  # noqa: BLE001 - tool errors become model-visible text
        return f"error: {e}"
