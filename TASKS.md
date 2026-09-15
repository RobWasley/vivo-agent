# TASKS: voice-toy

> Last updated: 2026-09-15 15:50

## Tasks

| ID | Priority | Deps | Description | Verification | Status |
|----|----------|------|-------------|--------------|--------|
| T001 | P0 | — | Scaffold project: Dockerfile (python:3.11-slim, CPU wheels for faster-whisper + kokoro-onnx + onnxruntime), docker-compose.yml (single service, models volume, env LLM_BASE_URL/LLM_MODEL/PERSONA), .gitignore, README. | `docker compose build` succeeds; `docker compose up -d` starts a container that serves a placeholder page on the fixed port. | pending |
| T002 | P0 | T001 | Implement Silero VAD wrapper (`app/vad.py`): consume 16k mono PCM frames, emit speech-start/speech-end with endpointing (min speech, max silence). | Unit test: feed synthetic PCM (silence + tone + silence) → correct start/end events with expected timing. | pending |
| T003 | P0 | T001 | Implement faster-whisper STT wrapper (`app/stt.py`): lazy-load small int8, transcribe a 16k PCM segment → text. Bench latency on host CPU. | Unit test: transcribe a generated TTS sample (or recorded clip) → expected words present. Record tok/s-equivalent (chars/s) in NOTES.md. | pending |
| T004 | P0 | T001 | Implement kokoro-onnx TTS wrapper (`app/tts.py`): lazy-load 82M, synthesize text → 24k PCM, streaming per sentence. Bench latency. | Unit test: synthesize "Hello world" → non-empty PCM at 24kHz. Record ms/char in NOTES.md. | pending |
| T005 | P0 | T001 | Implement stateless agent (`app/agent.py`): hand-rolled tool loop against llama.cpp v1 `/chat/completions`, `chat_template_kwargs={"enable_thinking": false}`, streaming, tool-call parsing + re-call. | Curl test against live server: request with enable_thinking=false returns no reasoning; agent loop completes a tool round-trip (get_time) and returns final text. | pending |
| T006 | P0 | T005 | Implement tools (`app/tools.py`): get_time, weather (open-meteo, no key), read_file (sandboxed to mounted dir). | Unit tests: get_time returns current time; weather returns JSON for a lat/lon; read_file reads a file in the sandbox and rejects paths outside it. | pending |
| T007 | P0 | T002,T003,T004,T005,T006 | Wire the pipeline in `app/main.py`: WS endpoint, VAD→STT→agent→TTS loop, sentence chunker, barge-in (cancel playback + abort LLM). | Integration test: scripted WS client sends synthetic speech → receives transcript + TTS audio; barge-in mid-playback cancels. | pending |
| T008 | P1 | T007 | Build the UI (`static/`): wobbly audio-reactive dot (canvas), mic capture (getUserMedia → PCM 16k), audio playback (PCM 24k), transcript sidebar. | Browser test: open UI, speak → transcript appears, dot wobbles with audio, TTS plays. | pending |
| T009 | P1 | T007 | Models on volume: download faster-whisper small + kokoro 82M + silero VAD into the mounted volume on first run; skip if present. | `docker compose down && up` → second start does not re-download (log line + no network). | pending |
| T010 | P2 | T008 | Polish: persona env honoured, error surfacing in UI, latency logging. | Change PERSONA env → different spoken character; errors visible in sidebar. | pending |

## Status Legend
- `pending` — not started
- `in-progress` — actively being worked
- `blocked` — cannot proceed (see Blockers)
- `done` — complete and verified (evidence recorded)
- `descoped` — removed from scope (record why)

## Blockers
- <none>

## Dependency Notes
- Critical path: T001 → {T002,T003,T004,T005} → T006 → T007 → T008.
- T002/T003/T004/T005 are parallelizable after T001 (independent wrappers).
- T006 depends on T005 (tool schemas live in the agent).
- T009 can run any time after T001 (volume wiring).
- T010 is polish, last.
