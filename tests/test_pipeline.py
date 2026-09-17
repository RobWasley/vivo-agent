"""T007 integration test: scripted WS client against the live pipeline.

Needs the app running (WS_URL env points at it; inside compose, run with
-e WS_URL=ws://vivo:8000/ws). Uses the live llama.cpp for the agent.
"""

import asyncio
import json
import os
import wave

import numpy as np
import websockets

WS_URL = os.environ.get("WS_URL", "ws://127.0.0.1:8000/ws")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
EXPECTED_WORDS = {"quick", "brown", "fox", "jumps", "over", "lazy", "dog"}
TIMEOUT = 120.0


def load_pcm16(path: str) -> np.ndarray:
    with wave.open(path) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n16 = int(len(pcm) * 16000 / sr)
    x = np.linspace(0, len(pcm) - 1, len(pcm))
    xi = np.linspace(0, n16 - 1, n16)
    pcm = np.interp(xi, x, pcm)
    return (np.clip(pcm, -1, 1) * 32767).astype("<i2")


def ftype(frame) -> str:
    if isinstance(frame, (bytes, bytearray)):
        return "binary"
    try:
        return json.loads(frame).get("type")
    except (json.JSONDecodeError, AttributeError):
        return "text"


def fjson(frame) -> dict:
    return json.loads(frame)


def tail_silence_samples() -> int:
    """Tail silence for endpointing must exceed the *running service's* VAD
    min_silence + reopen (the file is user-editable, T019), plus a margin."""
    import urllib.parse
    import urllib.request

    parts = urllib.parse.urlsplit(WS_URL)
    scheme = "https" if parts.scheme == "wss" else "http"
    try:
        with urllib.request.urlopen(
            f"{scheme}://{parts.netloc}/api/config", timeout=5
        ) as r:
            vad = json.load(r)["values"]["vad"]
        ms = vad["min_silence_ms"] + vad["reopen_ms"]
    except Exception:
        ms = 1000  # built-in defaults: 400 + 600
    return int((ms / 1000.0 + 0.5) * 16000)


async def send_utterance(ws, pcm16: np.ndarray, frame: int = 3200) -> None:
    for i in range(0, len(pcm16), frame):
        await ws.send(pcm16[i : i + frame].tobytes())
    await ws.send(np.zeros(tail_silence_samples(), dtype="<i2").tobytes())


async def recv_until(ws, predicate, timeout: float):
    frames = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=deadline - loop.time())
        except asyncio.TimeoutError:
            return frames, False
        frames.append(msg)
        if predicate(frames):
            return frames, True
    return frames, False


async def _utterance_reply(pcm16: np.ndarray) -> None:
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await send_utterance(ws, pcm16)
        frames, ok = await recv_until(ws, lambda fs: any(ftype(f) == "reply_done" for f in fs), TIMEOUT)
        assert ok, f"no reply_done in {len(frames)} frames: {[ftype(f) for f in frames]}"
        types = [ftype(f) for f in frames]
        assert "start" in types, types
        assert "end" in types, types
        transcript = next(fjson(f)["text"] for f in frames if ftype(f) == "transcript")
        hit = EXPECTED_WORDS & set(transcript.lower().replace(",", " ").split())
        assert len(hit) >= 5, f"transcript was: {transcript!r}"
        assert any(ftype(f) == "agent_text" for f in frames), "no agent text streamed"
        audio = b"".join(f for f in frames if isinstance(f, (bytes, bytearray)))
        assert len(audio) > 2 * 24000, f"only {len(audio) // 2} TTS samples"


async def _barge_in(pcm16: np.ndarray) -> None:
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await send_utterance(ws, pcm16)
        frames, ok = await recv_until(
            ws,
            lambda fs: any(isinstance(f, (bytes, bytearray)) and len(f) > 1000 for f in fs),
            TIMEOUT,
        )
        assert ok, f"no TTS audio arrived: {[ftype(f) for f in frames]}"
        await ws.send(json.dumps({"type": "barge_in"}))
        frames2, ok = await recv_until(ws, lambda fs: any(ftype(f) == "barge_ack" for f in fs), 10)
        assert ok, f"no barge_ack: {[ftype(f) for f in frames2]}"
        frames3, got_more = await recv_until(
            ws,
            lambda fs: any(isinstance(f, (bytes, bytearray)) and len(f) > 1000 for f in fs),
            3.0,
        )
        assert not got_more, "TTS audio kept flowing after barge_in"

        # a fresh utterance still works after the interruption
        await send_utterance(ws, pcm16)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TIMEOUT
        seen_end = False
        frames4 = []
        while loop.time() < deadline:
            msg = await asyncio.wait_for(ws.recv(), timeout=deadline - loop.time())
            frames4.append(msg)
            t = ftype(msg)
            if t == "end":
                seen_end = True
            if seen_end and t == "reply_done":
                break
        assert seen_end, f"no new utterance captured: {[ftype(f) for f in frames4]}"
        assert any(ftype(f) == "transcript" for f in frames4)


def test_utterance_reply():
    pcm16 = load_pcm16(os.path.join(FIXTURES, "stt_sample.wav"))
    asyncio.run(_utterance_reply(pcm16))


def test_barge_in_cancels_reply():
    pcm16 = load_pcm16(os.path.join(FIXTURES, "prompt_long.wav"))
    asyncio.run(_barge_in(pcm16))
