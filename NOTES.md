# NOTES: voice-toy

> Scratch space for environment discoveries and debugging. Not a plan.

## Environment Discoveries
- Project path: `/home/rob/projects/voice-toy`
- Git repo initialised (local-only, no remote).
- User's llama.cpp server: existing, v1 OpenAI-compatible endpoint. URL/port not yet confirmed (U1).
- Pithagoras reference: `thecodacus/pithagoras` (main) — voice pipeline design reference (Silero V5, speculative STT, sentence-chunked TTS ≤600 chars, barge-in cancels queues + upstream). Its voice stack is GPU-locked (Breeze-TTS-2 / audio.cpp CUDA), so this is a fresh CPU build.

## Debugging
- <none yet>

## Useful References
- Pithagoras voice pipeline (design reference): Silero V5 VAD, rolling speculative transcription, sentence-chunked streaming TTS (≤600 chars), independent synthesis/playback queues, barge-in cancels queues + upstream request, lazy model loading via leases.
- faster-whisper: https://github.com/SYSTRAN/faster-whisper (CTranslate2, int8, small model)
- kokoro-onnx: https://github.com/thewh1teagle/kokoro-onnx (82M, CPU)
- Silero VAD: https://github.com/snakers4/silero-vad
- open-meteo (no API key): https://open-meteo.com/
- llama.cpp server: `chat_template_kwargs` for per-request template vars (e.g. `enable_thinking`); `--reasoning-format` controls thought extraction.
