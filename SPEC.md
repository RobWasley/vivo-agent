# SPEC: vivo

> Status: approved — implementation in progress
> Last updated: 2026-09-17 (T022: TTS engine swapped kokoro-onnx → LuxTTS voice cloning)

## Objective

A CPU-only, single-container voice assistant: talk to it hands-free through a browser, it listens (Silero VAD), transcribes (faster-whisper small int8), answers via a stateless smol agent pointed at the existing llama.cpp v1 endpoint, and speaks back in a cloned voice (LuxTTS voice cloning). The UI is a wobbly audio-reactive dot plus a transcript sidebar.

## Confirmed Requirements
- CPU-only; single Docker container.
- Hands-free from day one (Silero VAD), not push-to-talk.
- Web UI: wobbly audio-reactive dot + transcript sidebar.
- STT: faster-whisper `small` int8, in-process.
- TTS: LuxTTS (zipvoice) voice cloning, in-process, 48 kHz. Voices are 3–30 s reference clips; a `default` clip ships with the repo and users add their own via the settings pane (upload or mic record). (T022, D018 — supersedes kokoro-onnx 82M.)
- Agent: smol agent pointed at the user's existing llama.cpp v1 endpoint, with persistent conversation memory: turn history shared across utterances, idle-time summary compaction when history grows (T011, D009). Full tool set (T012, D010): sandboxed shell `exec` in the mounted workspace, file tools (`read_file`, `write_file`, `list_dir`), keyless web search (`web_search`, DuckDuckGo) and page reading (`web_fetch`).
- Models stored on a mounted volume (survive rebuilds).
- Config via env: `LLM_BASE_URL`, `LLM_MODEL`, `PERSONA`, `COMPACT_AFTER_CHARS`, `KEEP_RECENT_TURNS`, `WORK_DIR`, `EXEC_TIMEOUT`, `EXEC_MAX_OUTPUT`, `SEARCH_MAX_RESULTS`, `FETCH_MAX_CHARS`.
- Non-thinking LLM responses per request (instant voice replies) — see D005.

## Assumptions
- A1: llama.cpp server is reachable from the container at the host gateway (e.g. `http://host.docker.internal:8080/v1` or a LAN IP). Verify actual URL/port before T004.
- A2: The loaded model is a Qwen3-style chat model whose template supports `enable_thinking` (or accepts `/no_think`). Verify against the live server's model.
- A3: The host has a decent CPU (≥8 cores) so small-int8 STT + the LuxTTS flow-matching model run at usable latency. Verify with `nproc`/bench.
- A4: Browser mic capture works over the network (HTTPS or localhost). If the UI is served on plain HTTP from a non-localhost origin, `getUserMedia` will be blocked — may need a local tunnel or `localhost` access.
- A5: A bundled `default` reference clip (Kokoro `af_heart` speech) is acceptable out of the box; users can record/upload their own clip to be cloned (T022).

## Constraints
- No GPU. All inference on CPU.
- No API keys anywhere (open-meteo, local llama.cpp).
- Single container; no multi-service compose.
- Low-power-friendly: no heavy frameworks, no GPU images.

## Dependencies
- Existing llama.cpp server (v1 OpenAI-compatible endpoint) — user's, already running.
- faster-whisper (CTranslate2) — CPU wheels.
- LuxTTS (zipvoice) — CPU: torch CPU wheels + onnxruntime; models pulled from the Hugging Face cache (`YatharthS/LuxTTS`, `openai/whisper-tiny`), pinned via `HF_HOME`.
- Silero VAD: `silero_vad_v6.onnx` bundled with faster-whisper, run via onnxruntime (verified 2026-09-15, see D006).
- open-meteo API (no key) for the weather tool.
- Python 3.11 base image.

## Non-Goals
- No conversation memory/context between turns (stateless agent). — **superseded 2026-09-15 by T011/D009**: turn history + idle compaction now in scope.
- No multi-user, no auth. Conversation history persists (compact, on the mounted `./data` volume), but no per-user accounts or transcript archive.
- No host-level command execution: `exec` runs inside the container, sandboxed to the mounted workspace (D010). No background/long-running command sessions, no MCP, no plugin system.
- No GPU path, no Breeze/audio.cpp.
- No mobile app; browser only.
- No fine-tuning of the TTS model. Custom voices are reference-clip cloning (zero-shot), not training — a 3–30 s clip is encoded once and reused (T022).

## Unknowns
- ~~U1~~ (resolved 2026-09-15): llama.cpp reachable from container at `http://host.docker.internal:8080/v1` (server on `0.0.0.0:8080`).
- ~~U2~~ (resolved 2026-09-15): `qwen3.8-27b` honours `chat_template_kwargs={"enable_thinking": false}` — no `reasoning_content` in response.
- ~~U3~~ (resolved 2026-09-15): Silero VAD runs via onnxruntime using the `silero_vad_v6.onnx` bundled inside faster-whisper — no separate package (D006).
- ~~U4~~ (resolved 2026-09-17): LuxTTS voice quality/latency on this CPU — RTF ~0.2 (≈5× faster than kokoro's ~1.0), clone quality verified on real speech (T022, bench in `NOTES.md`).
- U5: Mic access over plain HTTP from a LAN origin (A4).

## Acceptance Criteria
- [ ] AC1: Container starts with `docker compose up -d` and serves the UI on a fixed port.
- [ ] AC2: Speaking a sentence into the browser mic produces a transcript in the sidebar within ~2s of finishing speech (VAD endpointing).
- [ ] AC3: The agent replies via the llama.cpp endpoint with thinking disabled (no `reasoning_content`, no `think` tags) and the reply is spoken by LuxTTS in the active cloned voice.
- [ ] AC4: TTS streams sentence-by-sentence: first audio chunk starts playing before the full reply is generated.
- [ ] AC5: Barge-in: starting to speak while the bot is talking cancels playback and the in-flight LLM request.
- [ ] AC6: The dot wobbles/reacts to audio levels (mic input and/or TTS output).
- [ ] AC7: Tools work: `get_time`, `weather` (open-meteo), `read_file` (sandboxed to a mounted dir) — agent can call them and fold results into the spoken reply.
- [ ] AC8: Models live on a mounted volume; `docker compose down && up` does not re-download them.
- [ ] AC9: `LLM_BASE_URL`, `LLM_MODEL`, `PERSONA` env vars are honoured (change persona → different vivo identity).

## Proposed Architecture

Single Python container, one process, three in-process engines + a web server:

```
Browser (mic + audio playback + dot UI)
   │  WebSocket (PCM 16k mono up, PCM 48k mono down)
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
    ├── TTS: LuxTTS voice cloning (48 kHz), sentence-chunked streaming
    │     └── synthesize next sentence while current plays (independent queues)
   └── Barge-in: VAD speech-start during playback → cancel playback + abort LLM stream
```

Key data flow: mic PCM → VAD → (speculative) STT → agent (LLM + tools) → sentence chunks → TTS → browser. Barge-in cuts the loop at playback and aborts upstream.

Components:
- `app/main.py` — FastAPI app, WS endpoint, static UI.
- `app/vad.py` — Silero VAD wrapper (speech start/end detection, endpointing).
- `app/stt.py` — faster-whisper wrapper (lazy load, int8, small).
- `app/tts.py` — LuxTTS voice-cloning wrapper (lazy load, sentence streaming, reference-clip voices, 48 kHz).
- `app/agent.py` — stateless tool loop against llama.cpp v1.
- `app/tools.py` — get_time, weather (open-meteo), read_file (sandboxed).
- `app/chunker.py` — sentence chunker (≤600 chars).
- `static/index.html` + `static/app.js` + `static/style.css` — dot UI, mic capture, audio playback, transcript sidebar.
- `docker-compose.yml` — single service, volume for models, env passthrough.
- `Dockerfile` — python:3.11-slim, CPU wheels.

Why this design: proven pipeline (speculative STT, sentence-chunked TTS, barge-in) on a CPU-friendly stack; hand-rolled agent keeps it small and auditable; single process avoids IPC latency.
