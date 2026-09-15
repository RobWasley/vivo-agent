# DECISIONS: voice-toy

> Append-only log of significant decisions. Newest at the bottom.

## Decision Log

### D001 — faster-whisper over whisper.cpp for STT
- **Date**: 2026-09-15
- **Context**: Need CPU STT in-process. whisper.cpp is fast but requires a separate process/HTTP or cffi binding; faster-whisper (CTranslate2) is a clean Python API with int8 quantisation.
- **Decision**: Use faster-whisper `small` int8, in-process.
- **Rationale**: Simpler integration (pure Python, no IPC), int8 is CPU-friendly, small model is a good accuracy/latency tradeoff.
- **Alternatives Considered**: whisper.cpp (faster but needs separate process or cffi), whisperX (heavier, GPU-oriented), Vosk (lower accuracy).

### D002 — kokoro-onnx 82M for TTS
- **Date**: 2026-09-15
- **Context**: Need CPU TTS, in-process, no API key.
- **Decision**: Use kokoro-onnx 82M, in-process.
- **Rationale**: Small (82M), good quality, ONNX runs on CPU, no API key, Python API.
- **Alternatives Considered**: piper (faster but lower quality), Coqui TTS (heavier, GPU-oriented), Breeze-TTS (GPU-locked, Pithagoras's choice).

### D003 — Hand-rolled tool loop, no agent framework
- **Date**: 2026-09-15
- **Context**: Need a stateless smol agent with tools (get_time, weather, read_file) pointed at llama.cpp v1.
- **Decision**: Hand-roll the tool loop (no LangChain/LlamaIndex/etc).
- **Rationale**: Keeps it small and auditable (user's local-first preference); the loop is ~50 lines; no framework bloat or version churn.
- **Alternatives Considered**: LangChain (heavy, opinionated), LlamaIndex (heavy), raw OpenAI SDK only (no tool loop).

### D004 — Single container, single process
- **Date**: 2026-09-15
- **Context**: User wants a single-container CPU toy.
- **Decision**: One Python process hosts FastAPI + VAD + STT + TTS + agent.
- **Rationale**: Avoids IPC latency and multi-service complexity; all engines are in-process Python libs.
- **Alternatives Considered**: Multi-service compose (more moving parts), separate STT/TTS microservices (IPC overhead).

### D005 — Non-thinking LLM responses per request
- **Date**: 2026-09-15
- **Context**: Voice replies must be instant; thinking models add latency and `reasoning_content` noise.
- **Decision**: Send `chat_template_kwargs={"enable_thinking": false}` per request (Qwen3-style templates); fallback: append `/no_think` to the user message. Cap `max_tokens`. Use a small system prompt. Stream.
- **Rationale**: Per-request toggle keeps the same server/model but forces non-thinking for voice. Untoggleable reasoning models (R1/QwQ) can't be switched — assume the live model is a Qwen3-style chat model (verify U2).
- **Alternatives Considered**: Separate non-thinking model on the server (extra model load), client-side stripping of think tags (wastes latency), always-on thinking (too slow for voice).
