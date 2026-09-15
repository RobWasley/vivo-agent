# SPEC: voice-toy

> Status: draft (awaiting review)
> Last updated: 2026-09-15 15:50

## Objective

A CPU-only, single-container voice toy: talk to it hands-free through a browser, it listens (Silero VAD), transcribes (faster-whisper small int8), answers via a stateless smol agent pointed at the existing llama.cpp v1 endpoint, and speaks back (kokoro-onnx 82M). The UI is a wobbly audio-reactive dot plus a transcript sidebar. Design borrows Pithagoras's voice pipeline (speculative transcription, sentence-chunked streaming TTS, barge-in) but is a fresh CPU build — Pithagoras's own voice stack is GPU-locked.

## Confirmed Requirements
- CPU-only; single Docker container.
- Hands-free from day one (Silero VAD), not push-to-talk.
- Web UI: wobbly audio-reactive dot + transcript sidebar.
- STT: faster-whisper `small` int8, in-process.
- TTS: kokoro-onnx 82M, in-process.
- Agent: stateless smol agent (no context/memory between turns), pointed at the user's existing llama.cpp v1 endpoint.
- Models stored on a mounted volume (survive rebuilds).
- Config via env: `LLM_BASE_URL`, `LLM_MODEL`, `PERSONA`.
- Non-thinking LLM responses per request (instant voice replies) — see D005.

## Assumptions
- A1: llama.cpp server is reachable from the container at the host gateway (e.g. `http://host.docker.internal:8080/v1` or a LAN IP). Verify actual URL/port before T004.
- A2: The loaded model is a Qwen3-style chat model whose template supports `enable_thinking` (or accepts `/no_think`). Verify against the live server's model.
- A3: The host has a decent CPU (≥8 cores) so small-int8 STT + 82M TTS run at usable latency. Verify with `nproc`/bench.
- A4: Browser mic capture works over the network (HTTPS or localhost). If the UI is served on plain HTTP from a non-localhost origin, `getUserMedia` will be blocked — may need a local tunnel or `localhost` access.
- A5: Kokoro voice `af_heart` (or similar default) is acceptable for v1.

## Constraints
- No GPU. All inference on CPU.
- No API keys anywhere (open-meteo, local llama.cpp).
- Single container; no multi-service compose.
- Low-power-friendly: no heavy frameworks, no GPU images.

## Dependencies
- Existing llama.cpp server (v1 OpenAI-compatible endpoint) — user's, already running.
- faster-whisper (CTranslate2) — CPU wheels.
- kokoro-onnx — CPU.
- Silero VAD (bundled with faster-whisper? No — separate `silero-vad` or onnxruntime model; verify).
- open-meteo API (no key) for the weather tool.
- Python 3.11 base image.

## Non-Goals
- No conversation memory/context between turns (stateless agent).
- No multi-user, no auth, no persistence of transcripts.
- No GPU path, no Breeze/audio.cpp.
- No mobile app; browser only.
- No fine-tuning or custom voices.

## Unknowns
- U1: Exact llama.cpp server URL/port reachable from the container.
- U2: Whether the live model's template honours `chat_template_kwargs.enable_thinking=false` (test with curl before building the agent).
- U3: Silero VAD integration path: `silero-vad` pip package vs onnxruntime + model file.
- U4: Kokoro voice quality/latency on this CPU (bench in T003).
- U5: Mic access over plain HTTP from a LAN origin (A4).

## Acceptance Criteria
- [ ] AC1: Container starts with `docker compose up -d` and serves the UI on a fixed port.
- [ ] AC2: Speaking a sentence into the browser mic produces a transcript in the sidebar within ~2s of finishing speech (VAD endpointing).
- [ ] AC3: The agent replies via the llama.cpp endpoint with thinking disabled (no `reasoning_content`, no `think` tags) and the reply is spoken by Kokoro.
- [ ] AC4: TTS streams sentence-by-sentence: first audio chunk starts playing before the full reply is generated.
- [ ] AC5: Barge-in: starting to speak while the bot is talking cancels playback and the in-flight LLM request.
- [ ] AC6: The dot wobbles/reacts to audio levels (mic input and/or TTS output).
- [ ] AC7: Tools work: `get_time`, `weather` (open-meteo), `read_file` (sandboxed to a mounted dir) — agent can call them and fold results into the spoken reply.
- [ ] AC8: Models live on a mounted volume; `docker compose down && up` does not re-download them.
- [ ] AC9: `LLM_BASE_URL`, `LLM_MODEL`, `PERSONA` env vars are honoured (change persona → different voice character).

## Proposed Architecture

Single Python container, one process, three in-process engines + a web server:

```
Browser (mic + audio playback + dot UI)
   │  WebSocket (PCM 16k mono up, PCM 24k mono down)
   ▼
FastAPI app (uvicorn)
   ├── VAD loop: Silero VAD on mic stream → speech segments
   │     └── speculative transcription: faster-whisper small int8 on segment
   │           └── on final transcript → Agent
   ├── Agent: stateless tool loop (hand-rolled, no framework)
   │     ├── system prompt = PERSONA + tool schemas
   │     ├── POST {LLM_BASE_URL}/chat/completions
   │     │     chat_template_kwargs={"enable_thinking": false}, stream=true
   │     ├── parse tool calls → execute (get_time / weather / read_file) → append → re-call
   │     └── final text → sentence chunker → TTS queue
   ├── TTS: kokoro-onnx 82M, sentence-chunked streaming
   │     └── synthesize next sentence while current plays (independent queues)
   └── Barge-in: VAD speech-start during playback → cancel playback + abort LLM stream
```

Key data flow: mic PCM → VAD → (speculative) STT → agent (LLM + tools) → sentence chunks → TTS → browser. Barge-in cuts the loop at playback and aborts upstream.

Components:
- `app/main.py` — FastAPI app, WS endpoint, static UI.
- `app/vad.py` — Silero VAD wrapper (speech start/end detection, endpointing).
- `app/stt.py` — faster-whisper wrapper (lazy load, int8, small).
- `app/tts.py` — kokoro-onnx wrapper (lazy load, sentence streaming).
- `app/agent.py` — stateless tool loop against llama.cpp v1.
- `app/tools.py` — get_time, weather (open-meteo), read_file (sandboxed).
- `app/chunker.py` — sentence chunker (≤600 chars, Pithagoras-style).
- `static/index.html` + `static/app.js` + `static/style.css` — dot UI, mic capture, audio playback, transcript sidebar.
- `docker-compose.yml` — single service, volume for models, env passthrough.
- `Dockerfile` — python:3.11-slim, CPU wheels.

Why this design: mirrors Pithagoras's proven pipeline (speculative STT, sentence-chunked TTS, barge-in) on a CPU-friendly stack; hand-rolled agent keeps it small and auditable; single process avoids IPC latency.
