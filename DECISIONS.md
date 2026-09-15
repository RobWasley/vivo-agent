# DECISIONS: vivo

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
- **Verified (2026-09-15)**: Live `qwen3.8-27b` honours `chat_template_kwargs={"enable_thinking": false}` — returned `content` only, no `reasoning_content` (U2 resolved).

### D006 — Silero VAD via faster-whisper's bundled onnx (no torch)
- **Date**: 2026-09-15
- **Context**: U3 — need a CPU VAD. Options were the `silero-vad` pip package (pulls in torch, ~hundreds of MB) or onnxruntime + a model file.
- **Decision**: Use the `silero_vad_v6.onnx` asset already bundled inside faster-whisper (`faster_whisper/assets/silero_vad_v6.onnx`), run via onnxruntime (CPUExecutionProvider). No extra package, no torch.
- **Rationale**: Zero additional dependencies, keeps the image lean and CPU-only, and reuses an onnxruntime session we already ship for TTS.
- **Alternatives Considered**: `silero-vad` pip (torch dep, heavy), separate onnx download (redundant — already bundled).
- **Note**: Verified in-image 2026-09-15 that the asset exists and onnxruntime loads it (U3 resolved).

### D007 — kokoro int8 from GitHub releases; accept ~1× realtime, stream per sentence
- **Date**: 2026-09-15
- **Context**: kokoro-onnx 0.6.1 ships no model downloader; model files must be fetched separately. Benchmarked on this CPU: 61.8 ms/char ≈ 1.08× realtime; ORT thread sweep (intra 4–24) gave no gain.
- **Decision**: Download `kokoro-v1.0.int8.onnx` (114 MB) + `voices-v1.0.bin` (28 MB) from the kokoro-onnx GitHub release `model-files-v1.1` into `./models/kokoro/` on first start (idempotent, `app/models.py`); use int8 (CPU-friendly, small) and voice `af_heart`; keep the TTS wrapper streaming per sentence/chunk so first-audio latency is one short synthesis, not a whole reply.
- **Rationale**: int8 halves memory vs fp32 with no quality hit for 82M; 1× realtime is acceptable for a voice assistant when chunked (1–2 s sentences start after ~0.5–1 s).
- **Alternatives Considered**: fp32 onnx (slower, no quality gain on CPU), piper (faster, lower quality — D002), waiting for a faster engine (scope creep).

### D008 — Barge-in frees the mic immediately via a generation token
- **Date**: 2026-09-15
- **Context**: `test_barge_in_cancels_reply` timed out intermittently (pass alone, fail under LLM/CPU contention). Root cause: on `barge_in` the old worker set a shared `cancel` Event, but `reply_active` stayed `True` until that worker's `finally` ran. If the worker was mid an uninterruptible TTS synthesis (up to ~7 s when the CPU is shared), the next utterance's mic audio was silently dropped by the echo guard and `start_utterance` was rejected, so no `end` was ever emitted → test timeout. It only passed when the worker happened to be in the (fast) LLM-stream step at the moment of interruption.
- **Decision**: Barge-in now calls `session.release()`, which atomically sets `reply_active = False` and increments `generation`. Worker liveness is a generation check `session.alive(gen)` (current generation + not closed) rather than the shared `cancel` Event. The interrupted worker finishes its current (uninterruptible) step but then fails `alive(gen)` and can never send stale audio; its `finally` only resets state if it is still the current generation, so it can't clobber a newer reply.
- **Rationale**: Makes the interruption point deterministic regardless of which step the worker is in (LLM stream vs TTS synthesis); fresh speech is accepted immediately; the generation counter gives a race-free "am I still the active reply?" test that also handles a new utterance starting while an old worker is still draining.
- **Alternatives Considered**: keep the shared cancel Event (racy, the bug); make TTS synthesis interruptible (kokoro-onnx has no abort handle — a full re-synthesis would be needed, not worth it); a lock-guarded "draining" flag (more state than the counter, same outcome).
- **Verified (2026-09-15)**: barge-in test passes 3/3 alone (25–31 s) and the full suite is 18/18 in ~45 s (was 17/18 flaky at ~150 s).
