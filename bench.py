#!/usr/bin/env python3
"""vivo pipeline benchmark (run on the host, not in the container).

Drives the live WS pipeline with the TTS-generated speech fixtures while
sampling container CPU/memory (docker stats) and GPU utilisation/VRAM
(nvidia-smi), then correlates everything with the server's per-utterance
bench event log (`bench gen=<n> {json}` lines in the vivo container log;
see app/pipeline.py) to produce a before/after comparable report:

  client-observed : endpoint latency, end->first LLM token (as seen by the
                    client, incl. network), end->first audio, total, barge-in
  server events   : utterance_end, stt_end, llm_first_token, sentence_N_ready,
                    tts_N_start/end/sent, generation_complete, reply_done,
                    barge_in, stale_stop  (all seconds relative to speech end)
  resources       : container CPU% and VRAM per phase window (stt / tts / llm)

Usage:
  python3 bench.py --mode reply            short fixture, full reply
  python3 bench.py --mode long             long fixture (story), full reply
  python3 bench.py --mode barge            long fixture, barge-in after first audio
  python3 bench.py --mode all --repeats 2

Reports go to benchmarks/<mode>-<timestamp>.json (plus a summary on stdout).
The server must be running (docker compose up -d). The LLM must be reachable
from the container. If the container predates the bench logging, the server
events section is simply absent (client + resources still reported).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime, timezone

import numpy as np
import websockets

REPO = os.path.dirname(os.path.abspath(__file__))
SHORT_FIX = os.path.join(REPO, "tests", "fixtures", "stt_sample.wav")
LONG_FIX = os.path.join(REPO, "tests", "fixtures", "prompt_long.wav")
OUT_DIR = os.path.join(REPO, "benchmarks")
BENCH_RE = re.compile(r"bench gen=(\d+) (\{.*\})\s*$")
SAMPLE_RATE_TTS = 24000
FRAME = 3200  # mic frame size, samples (like the UI: 16 kHz frames)


# ---------------------------------------------------------------- resources

def parse_mem_mb(s: str) -> float:
    s = s.strip()
    m = re.match(r"([\d.]+)\s*(KiB|MiB|GiB|kB|MB|GB)?", s)
    if not m:
        return float("nan")
    v = float(m.group(1))
    unit = m.group(2) or "B"
    return v * {"B": 1 / 1048576, "KiB": 1 / 1024, "kB": 1 / 1048576 * 1000,
                "MiB": 1.0, "MB": 1000 / 1024, "GiB": 1024, "GB": 1000 * 1024 / 1048576}[unit]


class Sampler(threading.Thread):
    """Samples container CPU/mem + per-GPU utilisation/VRAM every `interval` s."""

    def __init__(self, container: str, interval: float = 0.1):
        super().__init__(daemon=True)
        self.container = container
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            s = {"t": time.time(), "cpu": None, "mem_mb": None, "gpu": []}
            try:
                out = subprocess.run(
                    ["docker", "stats", "--no-stream",
                     "--format", "{{.CPUPerc}}|{{.MemUsage}}", self.container],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                cpu_s, mem_s = out.split("|", 1)
                s["cpu"] = float(cpu_s.rstrip("%"))
                s["mem_mb"] = parse_mem_mb(mem_s.split("/")[0])
            except Exception:
                pass
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                for line in out.splitlines():
                    idx, util, mem = (x.strip() for x in line.split(","))
                    s["gpu"].append([int(idx), int(util), int(mem)])
            except Exception:
                pass
            self.samples.append(s)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def windows(self, t0: float, t1: float) -> list[dict]:
        return [s for s in self.samples if t0 <= s["t"] <= t1]

    @staticmethod
    def _stats(vals: list[float]) -> dict | None:
        if not vals:
            return None
        return {"max": round(max(vals), 1), "avg": round(sum(vals) / len(vals), 1), "n": len(vals)}

    def cpu(self, t0: float, t1: float) -> dict | None:
        return self._stats([s["cpu"] for s in self.windows(t0, t1) if s["cpu"] is not None])

    def gpu(self, t0: float, t1: float) -> dict | None:
        utils, vrams = [], []
        for s in self.windows(t0, t1):
            if s["gpu"]:
                utils.append(max(g[1] for g in s["gpu"]))
                vrams.append(max(g[2] for g in s["gpu"]))
        out = {}
        st = self._stats(utils)
        if st:
            out["util_pct"] = st
        st = self._stats(vrams)
        if st:
            out["vram_mb"] = st
        return out or None

    def mem_max(self, t0: float, t1: float) -> float | None:
        vals = [s["mem_mb"] for s in self.windows(t0, t1) if s["mem_mb"] is not None]
        return round(max(vals), 1) if vals else None


# ---------------------------------------------------------------- WS client

def load_pcm16(path: str) -> np.ndarray:
    with wave.open(path) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n16 = int(len(pcm) * 16000 / sr)
    x = np.linspace(0, len(pcm) - 1, len(pcm))
    xi = np.linspace(0, n16 - 1, n16)
    pcm = np.interp(xi, x, pcm)
    return (np.clip(pcm, -1, 1) * 32767).astype("<i2")


async def run_ws(url: str, pcm16: np.ndarray, barge: bool, timeout: float = 240.0) -> dict:
    """One scripted session. Returns client-observed metrics (monotonic secs)."""
    t: dict = {}
    out: dict = {}
    audio = []  # (t, nbytes)
    w0, m0 = time.time(), time.monotonic()  # wall anchor (resource-window fallback)
    async with websockets.connect(url, max_size=None) as ws:
        for i in range(0, len(pcm16), FRAME):
            await ws.send(pcm16[i:i + FRAME].tobytes())
        await ws.send(np.zeros(16000, dtype="<i2").tobytes())  # tail silence
        t["mic_done"] = time.monotonic()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        barged = False
        while loop.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=deadline - loop.time())
            except asyncio.TimeoutError:
                break
            now = time.monotonic()
            if isinstance(msg, (bytes, bytearray)):
                audio.append((now, len(msg)))
                if "first_audio" not in t:
                    t["first_audio"] = now
                t["last_audio"] = now
                if barge and not barged and len(msg) > 1000:
                    t["barge"] = now
                    await ws.send(json.dumps({"type": "barge_in"}))
                    barged = True
                continue
            try:
                m = json.loads(msg)
            except json.JSONDecodeError:
                out.setdefault("unparsed_frames", []).append(repr(msg[:200]))
                continue
            kind = m.get("type")
            if kind == "end" and "end" not in t:
                t["end"] = now
            elif kind == "agent_text" and "first_text" not in t:
                t["first_text"] = now
            elif kind == "barge_ack" and "barge_ack" not in t:
                t["barge_ack"] = now
                if barge:
                    break  # barge mode: the rest is stale-leak detection
            elif kind == "error":
                t["error"] = m.get("message")
                break
            elif kind == "reply_done":
                t["reply_done"] = now
                break
        if barge and barged:
            # after barge_ack: any further audio is a stale-audio leak
            deadline2 = loop.time() + 4.0
            leaked = 0
            while loop.time() < deadline2:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=deadline2 - loop.time())
                except asyncio.TimeoutError:
                    break
                now = time.monotonic()
                if isinstance(msg, (bytes, bytearray)):
                    leaked += len(msg)
                else:
                    try:
                        m = json.loads(msg)
                    except json.JSONDecodeError:
                        m = {}
                    if m.get("type") == "reply_done" and "reply_done" not in t:
                        t["reply_done"] = now
            t["stale_audio_leaked"] = leaked
    out.update({"audio_bytes": sum(b for _, b in audio), "audio_frames": len(audio),
                "audio_secs": round(sum(b for _, b in audio) / 2 / SAMPLE_RATE_TTS, 2)})
    out["wall"] = {k: round(w0 + (v - m0), 3) for k, v in t.items() if isinstance(v, float)}
    e = t.get("end")
    if e:
        for k, name in (("first_text", "llm_first_token"), ("first_audio", "first_audio"),
                        ("reply_done", "total")):
            if k in t:
                out[name] = round(t[k] - e, 3)
        out["endpoint_secs"] = round(e - t["mic_done"], 3)
    if barged:
        out["barge_to_ack_secs"] = round(t.get("barge_ack", t["barge"]) - t["barge"], 3)
        if "last_audio" in t:
            out["last_audio_to_barge_secs"] = round(t["barge"] - t["last_audio"], 3)
            out["barge_to_last_audio_secs"] = round(t["last_audio"] - t["barge"], 3) \
                if t["last_audio"] >= t["barge"] else 0.0
    if "error" in t:
        out["error"] = t["error"]
    return out


# ---------------------------------------------------------------- correlation

def server_bench_lines(container: str, since_wall: float,
                       expected: int = 0, timeout: float = 0.0) -> list[dict]:
    """Parse `bench gen=<n>` lines logged after since_wall (poll up to timeout s).

    In barge mode the server logs its report only after finishing and
    discarding the in-flight synthesis, which lags barge_ack by seconds.
    """
    deadline = time.time() + timeout
    while True:
        # explicit UTC: docker parses naive --since timestamps as UTC, but
        # fromtimestamp() without a tz would return local wall time. Logs go
        # to the container's stderr (python logging), so read both streams.
        r = subprocess.run(
            ["docker", "logs", "--since",
             datetime.fromtimestamp(since_wall, tz=timezone.utc).isoformat(), container],
            capture_output=True, text=True, timeout=30,
        )
        lines = []
        for ln in (r.stdout + r.stderr).splitlines():
            m = BENCH_RE.search(ln)
            if m:
                try:
                    lines.append(json.loads(m.group(2)))
                except json.JSONDecodeError:
                    pass
        out = [b for b in lines if b.get("t0_wall", 0) >= since_wall]
        if expected and len(out) >= expected:
            return out
        if time.time() >= deadline:
            return out
        time.sleep(0.5)


def correlate_client(sampler: Sampler, client: dict) -> dict:
    """Resource windows from client-observed times (container predates bench logging)."""
    w = client.get("wall", {})
    res: dict = {}
    if "end" not in w:
        return res
    w_start = w.get("mic_done", w["end"] - 1.0)
    w_end = w.get("reply_done") or w.get("barge") or w["end"]
    res["stt"] = sampler.cpu(w_start, w["end"])
    if "first_text" in w:
        res["llm"] = {"cpu": sampler.cpu(w["first_text"], w_end),
                      "gpu": sampler.gpu(w["first_text"], w_end)}
    if "first_audio" in w:
        res["tts"] = sampler.cpu(w["first_audio"], w_end)
    res["mem_max_mb"] = sampler.mem_max(w_start, w_end + 1)
    return res


def correlate(sampler: Sampler, client: dict, server: list[dict]) -> dict:
    res: dict = {}
    for b in server:
        t0 = b["t0_wall"]
        ev = b.get("events", {})
        t_end = t0 + ev.get("stt_end", 0)
        res[f"stt"] = sampler.cpu(t0, t_end)
        tts_win = [(t0 + ev[f"tts_{i}_start"], t0 + ev[f"tts_{i}_end"])
                   for i in range(1, 10) if f"tts_{i}_start" in ev and f"tts_{i}_end" in ev]
        cpu_parts = [sampler.cpu(a, b) for a, b in tts_win]
        cpu_vals = [c["max"] for c in cpu_parts if c]
        avg_vals = [c["avg"] for c in cpu_parts if c]
        res["tts"] = ({"max": round(max(cpu_vals), 1),
                       "avg": round(sum(avg_vals) / len(avg_vals), 1)} if cpu_vals else None)
        llm0, llm1 = ev.get("llm_first_token"), ev.get("generation_complete")
        if llm0 is not None and llm1 is not None:
            res["llm"] = {"cpu": sampler.cpu(t0 + llm0, t0 + llm1),
                          "gpu": sampler.gpu(t0 + llm0, t0 + llm1)}
        t1 = t0 + ev.get("reply_done", max((v for v in ev.values()), default=0))
        res["mem_max_mb"] = sampler.mem_max(t0, t1 + 1)
    return res


# ---------------------------------------------------------------- report

def human(mode: str, client: dict, server: list[dict], res: dict) -> str:
    L = [f"== bench {mode} =="]
    for k in ("endpoint_secs", "llm_first_token", "first_audio", "total",
              "barge_to_ack_secs", "barge_to_last_audio_secs", "stale_audio_leaked",
              "audio_secs", "audio_frames", "error"):
        if k in client:
            L.append(f"  client {k:26s} {client[k]}")
    if not server:
        L.append("  server bench events: (not logged by this container build)")
    for i, b in enumerate(server, 1):
        ev = b.get("events", {})
        L.append(f"  server gen={b.get('gen')} cancelled={b.get('cancelled')} "
                 f"sentences={b.get('sentences')} queue_max={b.get('queue_max')}")
        order = ["utterance_end", "stt_end", "llm_first_token", "generation_complete",
                 "barge_in", "stale_stop", "reply_done"]
        order += [f"sentence_{n}_ready" for n in range(1, 10)]
        order += [f"tts_{n}_{p}" for n in range(1, 10) for p in ("start", "end", "sent")]
        for k in order:
            if k in ev:
                L.append(f"    {k:22s} {ev[k]:8.3f}")
        for k in ("llm_gen_total", "tts_total", "llm_blocked_on_tts", "first_audio"):
            if k in b:
                L.append(f"    {k:22s} {b[k]:8.3f}")
    for phase in ("stt", "tts", "llm"):
        if phase in res and res[phase]:
            L.append(f"  res {phase:26s} {res[phase]}")
    if "mem_max_mb" in res and res["mem_max_mb"] is not None:
        L.append(f"  res {'mem_max_mb':26s} {res['mem_max_mb']}")
    return "\n".join(L)


MODES = {
    "reply": dict(fix=SHORT_FIX, barge=False),
    "long": dict(fix=LONG_FIX, barge=False),
    "barge": dict(fix=LONG_FIX, barge=True),
}


async def one_mode(name: str, args) -> dict:
    spec = MODES[name]
    pcm16 = load_pcm16(spec["fix"])
    runs, servers, ress = [], [], []
    for _ in range(args.repeats):
        sampler = Sampler(args.container, interval=args.sample_every)
        since = time.time() - 2
        sampler.start()
        client = await run_ws(args.url, pcm16, barge=spec["barge"])
        await asyncio.sleep(1.0)  # let the final log line flush
        server = server_bench_lines(args.container, since, expected=1,
                                    timeout=15.0 if spec["barge"] else 0.0)
        sampler.stop()
        res = correlate(sampler, client, server) if server else correlate_client(sampler, client)
        runs.append(client)
        servers.append(server)
        ress.append(res)
        print(human(f"{name} run {len(runs)}", client, server, res))
        await asyncio.sleep(2.0)
    return {
        "mode": name,
        "fixture": os.path.relpath(spec["fix"], REPO),
        "barge": spec["barge"],
        "url": args.url,
        "container": args.container,
        "client": runs,
        "server": servers,
        "resources": ress,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="ws://127.0.0.1:8600/ws")
    ap.add_argument("--container", default="vivo")
    ap.add_argument("--mode", choices=["reply", "long", "barge", "all"], default="all")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sample-every", type=float, default=0.1,
                    help="resource sampling interval seconds")
    ap.add_argument("--out", default=None, help="report path (default benchmarks/<mode>-<ts>.json)")
    args = ap.parse_args()

    names = ["reply", "long", "barge"] if args.mode == "all" else [args.mode]
    report = {"ts": datetime.now().isoformat(timespec="seconds"),
              "host_llm": "llama.cpp (see nvidia-smi in report for VRAM)",
              "modes": []}
    for name in names:
        report["modes"].append(asyncio.run(one_mode(name, args)))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = args.out or os.path.join(
        OUT_DIR, "bench-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
