# vivo

A CPU-only, single-container voice assistant: talk to it hands-free through a browser, it listens (Silero VAD), transcribes (faster-whisper small int8), answers via a stateless agent pointed at your llama.cpp v1 endpoint, and speaks back (kokoro-onnx 82M).

See `SPEC.md` for the full spec, `TASKS.md` for progress, `DECISIONS.md` for design rationale.

## Quickstart

```sh
docker compose up -d --build
```

UI: http://localhost:8600
Health: http://localhost:8600/health

## Configuration (env)

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | llama.cpp OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3.8-27b` | Model name to request |
| `PERSONA` | vivo | System prompt persona |

Models download to `./models` (mounted volume) on first run and survive rebuilds.
`read_file` tool is sandboxed to `./data`.

## Voice pipeline

The browser talks to a WebSocket at `ws://<host>:8600/ws`. Per utterance the
server runs: mic PCM (16 kHz) → Silero VAD endpointing → faster-whisper STT →
agent (llama.cpp, streaming, tools) → sentence chunker → kokoro TTS.

- Client sends mic audio as **binary int16 mono 16 kHz** frames (any size).
- Server streams back JSON events (`start`, `end`, `transcript`,
  `agent_text` deltas, `tool`, `reply_done`) plus **binary int16 mono 24 kHz**
  TTS audio, one chunk per sentence.
- `{"type":"barge_in"}` interrupts the active reply (stops speaking, aborts
  the LLM). While a reply is playing the mic is ignored (echo guard).

Full protocol in `app/pipeline.py`. The UI (`static/`) is wired in T008.

## Microphone & HTTPS (secure context)

The mic uses the browser `getUserMedia` API, which is only available in a
**secure context**: `https://…` or `http://localhost`. The container itself
serves **plain HTTP** on `:8600`, so opening it by LAN IP (e.g.
`http://192.168.0.200:8600`) disables the mic — the UI detects this and shows
a hint instead of a cryptic error. To use the mic from another device:

- **Reverse proxy with a real certificate (recommended)** — e.g. Nginx Proxy
  Manager. Forward to `http://192.168.0.200:8600` and:
  - enable **Websockets** (the app is a long-lived WS at `/ws`), and
  - raise the upstream timeout, e.g. add `proxy_read_timeout 300s;` (the WS
    stays open between utterances; background tabs can throttle the 15 s
    keep-alive ping).
- **`http://localhost:8600`** on the machine running the container.
- **Chrome/Edge only:** `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
  → add the HTTP origin → relaunch.

## Tests

```sh
# engine unit tests (run in the app image; share its deps + models)
docker compose run --rm -w /app vivo python -m pytest tests -v -p no:cacheprovider

# add the live WS pipeline integration test (talks to the running service)
docker compose run --rm -w /app -e WS_URL=ws://vivo:8000/ws \
  vivo python -m pytest tests -v -p no:cacheprovider
```

The pipeline + agent tests call the live llama.cpp server, so the server must
be up and reachable from the containers.
