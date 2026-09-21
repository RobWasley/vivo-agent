"""Relay the agent-browser viewport stream to the Vivo web client."""
from __future__ import annotations

import asyncio
import json
import os

import websockets
from fastapi import WebSocket, WebSocketDisconnect

STREAM_PORT = int(os.environ.get("AGENT_BROWSER_STREAM_PORT", "9248"))
STREAM_URL = f"ws://127.0.0.1:{STREAM_PORT}"
RETRY_SECONDS = 3
MAX_FPS = 30


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
                            config = _fps_message(message)
                            if config:
                                await upstream.send(config)

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