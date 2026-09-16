# NOTES: vivo

> Scratch space for environment discoveries and debugging. Not a plan.

## Environment Discoveries
- Project path: `/home/rob/projects/vivo-agent`
- Git repo initialised (local-only, no remote).
- Host: 24 cores, 46 GiB RAM (A3 satisfied). Docker 29.6.0, Compose v5.1.4.
- llama.cpp server: `llama-server` listening on `0.0.0.0:8080` (verified via `ss`). Host URL `http://localhost:8080/v1`; container URL `http://host.docker.internal:8080/v1` via `extra_hosts: host-gateway` (U1 resolved).
- Live models: `qwen3.8-27b` (loaded, Qwen3.8-27B-UD-Q4_K_XL.gguf, ~11 tok/s gen), `qwen3.6-35b-a3b` (unloaded), `unsloth/Qwen3.8-Flash-Next-GGUF:Q4_K_XL` (cache). Server flags include `--tools all` (tool calling supported) and template kwargs `preserve_thinking`, `reasoning_effort`.
- U2 resolved: `chat_template_kwargs={"enable_thinking": false}` on `qwen3.8-27b` returns content only, no `reasoning_content` (tested with curl 2026-09-15).
- U3 resolved: faster-whisper 1.2.1 bundles `faster_whisper/assets/silero_vad_v6.onnx` → run VAD via onnxruntime, no torch / no separate package (D006).
- kokoro-onnx 0.6.1 imports as `kokoro_onnx` (not `kokoro`); deps: espeakng-loader, numpy, onnxruntime, phonemizer. ctranslate2 4.8.2, onnxruntime 1.30.0 (CPUExecutionProvider only).
- UI host port: 8600 (avoids 8080 llama, 3000/8888/8889/3552/8091 in use by other containers).
- Pithagoras reference: `thecodacus/pithagoras` (main) — voice pipeline design reference (Silero V5, speculative STT, sentence-chunked TTS ≤600 chars, barge-in cancels queues + upstream). Its voice stack is GPU-locked (Breeze-TTS-2 / audio.cpp CUDA), so this is a fresh CPU build.

## Benchmarks (2026-09-15, in-container, CPU)
- STT (faster-whisper small int8, cpu_threads=8): 2.84 s of speech transcribed in **0.61 s** (best of 3; warm 1.76 s) → **4.6× realtime**; exact transcript "The quick brown fox jumps over the lazy dog." Load 1.06 s after first download.
- TTS (kokoro-onnx int8, ORT defaults): 180 chars → 12.08 s of 24 kHz audio in **11.1 s** (best of 3) = **61.8 ms/char ≈ 1.08× realtime**. Load 0.47 s.
- TTS thread tuning sweep (intra 4/8/12/16/24, inter 1): no difference (11.05–11.49 s) → the model is kernel-bound, not thread-bound. Implication: stream TTS per sentence (first chunk after ~0.5–1 s) — see D007.
- Pipeline live latencies (T007 integration, from server logs): STT steady-state **0.7–0.75 s** for a 2.8 s utterance (2.15 s first-in-process incl. model load); LLM first sentence out in ~1.5–2 s; first TTS sentence **3.3 s** (24 chars; ~1.5 s of that is first-call ORT warmup), later sentences ~63 ms/char. End-to-end speech-end → first-audio ≈ **4–7 s** (persona's one-or-two-sentence replies dominate via the LLM).

## Benchmarks — LLM/TTS decoupling before/after (2026-09-16, repeats 2, `benchmarks/`)
Baseline = instrumented old serial pipeline (`baseline-pure-20260916-060238.json`); after = queue pipeline (`after-queue-20260916-061624.json` + `after-queue-barge-20260916-063027.json`). All seconds, end of speech → event. LLM first token varies 0.9–5.9 s run-to-run (shared host model), so read ranges, not single values.

| metric (server unless noted) | baseline | after | note |
|---|---|---|---|
| long: `llm_blocked_on_tts` | 8.09–8.41 | **0.000** | LLM no longer waits on TTS |
| long: `llm_gen_total` (wall) | 9.74–10.30 | **1.11–1.20** | pure generation |
| long: first_audio | 11.29–11.65 | **7.31–7.94** | chunker 180→90 + overlap |
| long: audio frames | 1 (8.2 s) | 2 | |
| long: total | 11.29–11.65 | 11.83–13.57 | TTS-bound now (9.3 s audio @ ~1.2×) |
| reply: first_audio | 3.85–7.78 | 3.61–4.99 | LLM-variance-dominated (single sentence) |
| barge: first_audio | 10.92–13.40 | **7.39** (13.60 run w/ slow LLM) | |
| barge: server `cancelled` | False (raced — reply done 11 ms pre-barge) | **True** | `barge_in`→`stale_stop` ≈ 1 in-flight synth |
| barge: stale audio (client) | 0 | **0** | `barge_to_ack` 0.0–0.004 s |
| res: TTS CPU peak | 918–1709 % | 1619–1858 % | ~16–18 cores, unchanged |
| res: container mem max | 1002–1008 MB | 902–928 MB | |
| res: LLM VRAM | 14910 MB | 14910 MB | host model, unchanged |

Caveats: resource sampler effective interval ~0.5–1 s (docker stats + nvidia-smi subprocesses), so sub-second LLM windows (after: ~1.2 s) often yield `cpu: None`; GPU util samples mostly land on TTS-dominated stretches. Conversation state persists (`/data/conversation.json`, ~25 turns) — not reset, prefill impact negligible.

## Debugging
- **Synthetic tones are unreliable Silero v6 triggers**: 220 Hz sine 0.5 amp peaks at prob 0.507 (exactly 1 window ≥ 0.5); 440 Hz, sawtooth sweep, pink noise → all ~0. Earlier probe that "sine220 detected" was an artifact: one lucky onset window + segment auto-close at end-of-audio. VAD tests therefore use the real TTS speech fixture (`tests/fixtures/stt_sample.wav`).
- **Silero onsets early**: on the fixture it fires ~0.3–0.4 s before acoustic energy starts (low-freq rumble in TTS output), and its reported start timestamps are not 512-sample aligned (context offset). Test assertions use generous margins.
- **faster-whisper VAD API**: `fv.get_vad_model()(audio)` requires `len(audio) % 512 == 0` (pad it); `get_speech_timestamps` pads internally.
- **Streaming vs batch VAD scores are bit-identical** (verified by direct comparison, max diff 0.0) — per-window onnx calls with carried h/c state + 64-sample context prefix exactly reproduce faster-whisper's batched scoring.
- **Volume ownership**: docker auto-created `./models`/`./data` as root:root → appuser (uid 1000) got EACCES at startup. Fixed with `sudo chown -R 1000:1000 models data`.
- **pytest in image**: `COPY tests/` added to Dockerfile (tests run in-container, share the image's deps); main container can lag a rebuild — use `docker compose run --rm` for fresh runs.
- **kokoro-onnx 0.6.1**: pure onnxruntime (no ctranslate2 in the hot path despite the dependency); `create_session()` takes no thread params → session tuning only via monkeypatch; showed no gain.
- **Headless UI test (no mic device)**: Playwright Chromium has no capture device, so `getUserMedia` can't be exercised. Instead inject the TTS fixture through the page's own `resampleTo16k`→`f32To16`→`ws.send` (only the getUserMedia call is skipped). Headless WebAudio *does* run: an `AudioContext` reaches `state:"running"` and scheduled `BufferSource`s play to `onended` (24 kHz TTS decoded + consumed fully). Static files are baked into the image (`COPY static/`), so UI edits need `docker compose build && up -d --force-recreate`. Inline data-URI `<link rel=icon>` removes the browser's `/favicon.ico` 404.
- **Barge-in verified live**: after the client received the first TTS chunk, `barge_in` → `barge_ack`, zero further TTS synthesis logged for the cancelled reply (cancel checked before each sentence synth and in the LLM stream loop, which breaks and closes the httpx stream), then a fresh utterance processed normally. The worker's `finally` still emits `reply_done` for a cancelled reply — clients should treat `barge_ack` as "stop playing".
- **Bench harness pitfalls (T013)**: (1) container logs go to **stderr** (python logging) — `docker logs` stdout is empty; read both. (2) `docker logs --since` parses naive timestamps as **UTC** — pass `datetime.fromtimestamp(x, tz=timezone.utc).isoformat()`. (3) Binary WS frames must `continue` before `json.loads` (a single fallthrough crashes the client 100% of the time — "intermittent" was a misdiagnosis). (4) `Bench.report()` must not call `first_audio()` while holding its non-reentrant lock (self-deadlock → line never logged, threads leaked). (5) Barge mode: the server logs its report only after finishing + discarding the in-flight synthesis, **seconds after `barge_ack`** → poll the log (15 s) instead of reading once.
- **Sentinel-drain deadlock (T014)**: `release()` originally drained the TTS queue blindly. If the producer had already finished (queue = `[sentence, _TTS_DONE]`), the drain ate the sentinel; the worker then blocked forever in `q.get()` while the producer sat in `worker.join()` — two leaked threads, no bench line, no cleanup, and the next reply on that session was fine only because it gets a fresh queue. Fix: drain preserves `_TTS_DONE` (put it back and stop). Regression test: `test_barge_after_generation_complete_does_not_deadlock` (threads must return to the pre-utterance count).

## Useful References
- Pithagoras voice pipeline (design reference): Silero V5 VAD, rolling speculative transcription, sentence-chunked streaming TTS (≤600 chars), independent synthesis/playback queues, barge-in cancels queues + upstream request, lazy model loading via leases.
- faster-whisper: https://github.com/SYSTRAN/faster-whisper (CTranslate2, int8, small model)
- kokoro-onnx: https://github.com/thewh1teagle/kokoro-onnx (82M, CPU)
- Silero VAD: https://github.com/snakers4/silero-vad
- open-meteo (no API key): https://open-meteo.com/
- llama.cpp server: `chat_template_kwargs` for per-request template vars (e.g. `enable_thinking`); `--reasoning-format` controls thought extraction.
