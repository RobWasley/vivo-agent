# TASKS: vivo

> Last updated: 2026-09-15 20:02

## Tasks

| ID | Priority | Deps | Description | Verification | Status |
|----|----------|------|-------------|--------------|--------|
| T001 | P0 | — | Scaffold project: Dockerfile (python:3.11-slim, CPU wheels for faster-whisper + kokoro-onnx + onnxruntime), docker-compose.yml (single service, models volume, env LLM_BASE_URL/LLM_MODEL/PERSONA), .gitignore, README. | `docker compose build` succeeds; `docker compose up -d` starts a container that serves a placeholder page on the fixed port. | done |
| T002 | P0 | T001 | Implement Silero VAD wrapper (`app/vad.py`): consume 16k mono PCM frames, emit speech-start/speech-end with endpointing (min speech, max silence). | Unit test: feed synthetic PCM (silence + tone + silence) → correct start/end events with expected timing. | done |
| T003 | P0 | T001 | Implement faster-whisper STT wrapper (`app/stt.py`): lazy-load small int8, transcribe a 16k PCM segment → text. Bench latency on host CPU. | Unit test: transcribe a generated TTS sample (or recorded clip) → expected words present. Record tok/s-equivalent (chars/s) in NOTES.md. | done |
| T004 | P0 | T001 | Implement kokoro-onnx TTS wrapper (`app/tts.py`): lazy-load 82M, synthesize text → 24k PCM, streaming per sentence. Bench latency. | Unit test: synthesize "Hello world" → non-empty PCM at 24kHz. Record ms/char in NOTES.md. | done |
| T005 | P0 | T001 | Implement stateless agent (`app/agent.py`): hand-rolled tool loop against llama.cpp v1 `/chat/completions`, `chat_template_kwargs={"enable_thinking": false}`, streaming, tool-call parsing + re-call. | Curl test against live server: request with enable_thinking=false returns no reasoning; agent loop completes a tool round-trip (get_time) and returns final text. | done |
| T006 | P0 | T005 | Implement tools (`app/tools.py`): get_time, weather (open-meteo, no key), read_file (sandboxed to mounted dir). | Unit tests: get_time returns current time; weather returns JSON for a lat/lon; read_file reads a file in the sandbox and rejects paths outside it. | done |
| T007 | P0 | T002,T003,T004,T005,T006 | Wire the pipeline in `app/main.py`: WS endpoint, VAD→STT→agent→TTS loop, sentence chunker, barge-in (cancel playback + abort LLM). | Integration test: scripted WS client sends synthetic speech → receives transcript + TTS audio; barge-in mid-playback cancels. | done |
| T008 | P1 | T007 | Build the UI (`static/`): wobbly audio-reactive dot (canvas), mic capture (getUserMedia → PCM 16k), audio playback (PCM 24k), transcript sidebar. | Browser test: open UI, speak → transcript appears, dot wobbles with audio, TTS plays. | done |
| T009 | P1 | T007 | Models on volume: download faster-whisper small + kokoro 82M + silero VAD into the mounted volume on first run; skip if present. | `docker compose down && up` → second start does not re-download (log line + no network). | done |
| T010 | P2 | T008 | Polish: persona env honoured, error surfacing in UI, latency logging. | Change PERSONA env → different spoken character; errors visible in sidebar. | done |

## Status Legend
- `pending` — not started
- `in-progress` — actively being worked
- `blocked` — cannot proceed (see Blockers)
- `done` — complete and verified (evidence recorded)
- `descoped` — removed from scope (record why)

## Blockers
- <none>

## Dependency Notes
- T001 done 2026-09-15 16:34: `docker compose build` OK; container serves placeholder UI on host port 8600 (in-container 8000), `/health` OK. All engine imports verified in-image (ctranslate2 4.8.2, onnxruntime 1.30.0 CPU, kokoro_onnx 0.6.1). faster-whisper bundles `silero_vad_v6.onnx` → no separate VAD package needed (see D006).
- T002–T006 done 2026-09-15 17:52: full suite `docker compose run --rm vivo python -m pytest tests -v` → **16 passed in 17.8s** (test_vad 5, test_tts 3, test_stt 1, test_agent 1 live LLM, test_tools 6).
  - T002: `app/vad.py` streaming Silero v6 (onnx, 512-sample windows + 64-sample context + LSTM state), endpointing with hysteresis (0.5/0.35), min_speech applied to the voiced region, max-speech split, `flush()`. Tests use the real TTS speech fixture (synthetic tones proved unreliable — see NOTES).
  - T003: `app/stt.py` faster-whisper small int8. Bench: 2.84 s audio in 0.61 s = **4.6× realtime** (NOTES.md).
  - T004: `app/tts.py` kokoro-onnx. Bench: 180 chars → 11.1 s, 61.8 ms/char ≈ **1.08× realtime** (NOTES.md; D007).
  - T005: `app/agent.py` verified against live `qwen3.8-27b` — tool round-trip (get_time) + no think tags, streaming deltas.
  - T006: `app/tools.py` all 6 tests pass incl. path-escape rejection.
- T007 done 2026-09-15 18:35: `app/pipeline.py` (Engines shared singletons, SentenceChunker, VoiceSession, threaded utterance worker with `run_coroutine_threadsafe` sends) + `app/main.py` `/ws` endpoint + lifespan `Engines`. Integration tests `tests/test_pipeline.py` (scripted `websockets` client vs live service): happy path (start→end→transcript≥5/7 words→agent_text→24k TTS binary→reply_done) and barge-in (mid-reply cancel → barge_ack, no further TTS, fresh utterance still works). Full suite **18/18 in 50.4s** incl. live LLM. Latency: first audio ≈ STT (~0.7 s steady) + LLM first sentence (~1.5–2 s) + TTS sentence (~1.6–3 s) after speech end (NOTES.md).
- T008 done 2026-09-15 19:12: `static/{index.html,style.css,app.js}` — canvas wobbly audio-reactive dot (state-coloured, eyes+blink, level-driven wobble/glow), mic capture (getUserMedia → ScriptProcessor → linear resample to 16 kHz int16 → WS binary, any device rate), playback (24 kHz int16 → chained BufferSources with lookahead scheduling + AnalyserNode level), transcript sidebar (you / toy streaming / tool / error), status pill (connecting/ready/listening/thinking/speaking), controls (start/stop mic, barge in, flush, clear). Verified headless via Playwright (`.playwright-mcp/ui_test*.js`): injected the TTS fixture through the page's own send path → WS open → `transcript` "The quick brown fox jumps over the lazy dog." → `agent_text` reply streamed → 24 kHz TTS decoded + played to completion → meter tracked input level (80%) → **zero console errors** (inline data-URI favicon removes the `/favicon.ico` 404). Mic *capture* itself not testable headless (no device); getUserMedia→PCM path exercised via the same resample/send functions.
- Critical path: T001 → {T002,T003,T004,T005} → T006 → T007 → T008.
- T002/T003/T004/T005 are parallelizable after T001 (independent wrappers).
- T006 depends on T005 (tool schemas live in the agent).
- T009 done 2026-09-15 19:36: kokoro via `ensure_models()` (lifespan), faster-whisper small lazily on first transcription (`download_root=/models`), silero VAD bundled in the faster-whisper package (D006, never downloaded). Verified: `docker compose down && up` → logs show `model present: /models/kokoro/…` ×2 and zero download/fetch lines; forced whisper load in the fresh container (1.15 s, transcribed the fixture); **all 600 MB of volume files md5- and mtime-identical** before/after. Also removed a stray `models/testwrite` debug leftover.
- T009 can run any time after T001 (volume wiring).
- T010 done 2026-09-15 20:02 (polish, final): (1) **persona env** — `PERSONA` in `.env`/compose → `config.PERSONA` → `Agent` system prompt; verified live both ways: pirate persona (`.env`) → "That's the classic tongue-twister, arr, but what be yer true quest, matey?", default after removing it → "…lazy dog! That's the rest of the famous pangram." (2) **error surfacing** — pipeline exceptions already sent as `error` WS events → sidebar `entry-error`; verified end-to-end: ephemeral container with dead `LLM_BASE_URL` (port 9) → browser sidebar shows `ConnectError: [Errno 111] Connection refused` with the transcript still displayed, zero console errors. Added: WS disconnect after a successful connect now also posts a sidebar error ("disconnected from server") — previously only the status pill changed. (3) **latency logging** — `app/pipeline.py` now logs per utterance: `utterance 11.78s (stt 1.78s, first text 5.64s, first audio 8.61s)` (speech-end → first LLM text / first TTS audio / total) alongside the existing per-stage `stt`/`tts` lines. Full suite re-run: 17/18 with one barge-in TimeoutError under heavy LLM contention (suite 151 s vs ~50 s norm; the host LLM serves this session too) — test passes in 34.7 s when run alone; change is logging-only.
