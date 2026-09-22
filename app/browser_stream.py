"""Relay the agent-browser viewport stream to the Vivo web client."""
from __future__ import annotations

import asyncio
import json
import math
import os

import websockets
from fastapi import WebSocket, WebSocketDisconnect

STREAM_PORT = int(os.environ.get("AGENT_BROWSER_STREAM_PORT", "9248"))
STREAM_URL = f"ws://127.0.0.1:{STREAM_PORT}"
RETRY_SECONDS = 3
MAX_FPS = 30
MOUSE_EVENT_TYPES = {"mouseMoved", "mousePressed", "mouseReleased", "mouseWheel"}
MOUSE_BUTTONS = {"none", "left", "right", "middle", "back", "forward"}
KEYBOARD_EVENT_TYPES = {"keyDown", "keyUp"}


def apply_config(config) -> None:
    """Update settings inherited by browser daemons launched after a save."""
    os.environ["AGENT_BROWSER_STREAM_MAX_FPS"] = str(config.BROWSER_MAX_FPS)
    os.environ["AGENT_BROWSER_STREAM_QUALITY"] = str(config.BROWSER_QUALITY)
    os.environ["AGENT_BROWSER_STREAM_MAX_WIDTH"] = str(config.BROWSER_MAX_WIDTH)
    os.environ["AGENT_BROWSER_STREAM_MAX_HEIGHT"] = str(config.BROWSER_MAX_HEIGHT)


def _fps_message(message: str) -> str | None:
    try:
        payload = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None
    if payload.get("type") != "browser_fps":
        return None
    try:
        fps = max(1, min(int(payload.get("maxFps", 10)), MAX_FPS))
    except (TypeError, ValueError):
        return None
    return json.dumps({"type": "config", "maxFps": fps})


def _input_message(message: str) -> str | None:
    """Normalize trusted viewer mouse and keyboard events for agent-browser."""
    try:
        payload = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    modifiers = payload.get("modifiers", 0)
    if isinstance(modifiers, bool) or not isinstance(modifiers, int) or not 0 <= modifiers <= 15:
        return None

    if payload.get("type") == "input_mouse":
        x, y = payload.get("x"), payload.get("y")
        if (
            isinstance(x, bool) or isinstance(y, bool)
            or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
            or not math.isfinite(x) or not math.isfinite(y)
            or x < 0 or y < 0
        ):
            return None
        event_type = payload.get("eventType")
        button = payload.get("button", "left")
        if event_type not in MOUSE_EVENT_TYPES or button not in MOUSE_BUTTONS:
            return None
        if event_type == "mouseWheel":
            delta_x, delta_y = payload.get("deltaX"), payload.get("deltaY")
            if (
                isinstance(delta_x, bool) or isinstance(delta_y, bool)
                or not isinstance(delta_x, (int, float)) or not isinstance(delta_y, (int, float))
                or not math.isfinite(delta_x) or not math.isfinite(delta_y)
            ):
                return None
            return json.dumps({
                "type": "input_mouse", "x": round(x), "y": round(y),
                "eventType": event_type, "button": "none", "modifiers": modifiers,
                "clickCount": 0, "deltaX": delta_x, "deltaY": delta_y,
            })
        return json.dumps({
            "type": "input_mouse", "x": round(x), "y": round(y),
            "eventType": event_type, "button": button, "modifiers": modifiers,
            "clickCount": 1 if event_type == "mousePressed" else 0,
        })

    if payload.get("type") == "input_keyboard":
        key = payload.get("key")
        event_type = payload.get("eventType")
        code = payload.get("code", "")
        text = payload.get("text", "")
        if (
            not isinstance(key, str) or not key or len(key) > 128
            or event_type not in KEYBOARD_EVENT_TYPES
            or not isinstance(code, str) or len(code) > 128
            or not isinstance(text, str) or len(text) > 128
        ):
            return None
        return json.dumps({
            "type": "input_keyboard", "key": key,
            "eventType": event_type, "code": code, "text": text,
            "modifiers": modifiers,
        })
    return None


async def _send_status(ws: WebSocket, status: str) -> None:
    await ws.send_text(json.dumps({"type": "browser_status", "status": status}))


async def serve_browser_stream(ws: WebSocket) -> None:
    """Connect a browser viewer to agent-browser, retrying until it appears."""
    await ws.accept()
    try:
        while True:
            await _send_status(ws, "connecting")
            try:
                async with websockets.connect(STREAM_URL, max_size=None) as upstream:
                    await _send_status(ws, "online")

                    async def to_viewer():
                        async for message in upstream:
                            if isinstance(message, str):
                                await ws.send_text(message)

                    async def from_viewer():
                        while True:
                            message = await ws.receive_text()
                            upstream_message = _fps_message(message) or _input_message(message)
                            if upstream_message:
                                await upstream.send(upstream_message)

                    tasks = [asyncio.create_task(to_viewer()), asyncio.create_task(from_viewer())]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        task.result()
            except (OSError, websockets.WebSocketException, WebSocketDisconnect):
                await _send_status(ws, "offline")
                await asyncio.sleep(RETRY_SECONDS)
    except (WebSocketDisconnect, RuntimeError):
        return