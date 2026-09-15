# vivo

A CPU-only, single-container voice agent. Talk to it hands-free through a
browser: it listens (Silero VAD), transcribes (faster-whisper small int8),
answers with a small hand-rolled agent pointed at your llama.cpp v1 endpoint
(shell, file and web tools, persistent conversation memory), and speaks back
(kokoro-onnx 82M). No GPU, no API keys, one Docker container.

```
Browser (mic + playback + wobbly dot UI)
   │  WebSocket (PCM 16 kHz up, PCM 24 kHz down)
   ▼
FastAPI (uvicorn)
   ├── VAD: Silero v6 → speech endpointing
   ├── STT: faster-whisper small int8
   ├── Agent: streaming tool loop → llama.cpp v1 (thinking off)
   │     └── 8 tools: exec, files, web, time, weather
   ├── Memory: persistent turns, idle-time LLM compaction
   └── TTS: kokoro-onnx 82M, sentence-chunked streaming
```

## Prerequisites

- Docker + Compose.
- A CPU with ≥8 cores recommended. Benchmarks on a 24-core host: STT
  4.6× realtime, TTS ~1× realtime (see `NOTES.md` for the full numbers).
- A **running llama.cpp server** with an OpenAI-compatible `/v1` endpoint:
  - tool calling enabled (`--tools all`),
  - a Qwen3-style chat model whose template honours
    `chat_template_kwargs={"enable_thinking": false}` (verified with
    `qwen3.8-27b` — voice replies must be non-thinking for latency).
- ~600 MB free disk for models (auto-downloaded on first run).
- The UI opened from a **secure context** (HTTPS or `localhost`) or the
  browser mic won't work — see [Microphone & HTTPS](#microphone--https).

## Quickstart

1. Clone the repo and pick a host directory to be the agent's **workspace**
   — this is where `exec` runs and the file tools live. The compose file
   ships with the author's path; change it to yours:

   ```yaml
   volumes:
     - /path/to/your/workspace:/workspace
   ```

   The directory must be read/writable by uid 1000 (the container's
   `appuser`); a normal Linux account with uid 1000 already matches.

2. Pre-create and chown the volume dirs (otherwise Docker creates them as
   root and the app gets `EACCES`):

   ```sh
   mkdir -p models data && sudo chown -R 1000:1000 models data
   ```

3. Point it at your LLM if it isn't already. Defaults match llama.cpp on
   `0.0.0.0:8080` with model `qwen3.8-27b`; to override, put a `.env` file
   next to `docker-compose.yml`:

   ```sh
   LLM_BASE_URL=http://host.docker.internal:8080/v1
   LLM_MODEL=qwen3.8-27b
   PERSONA="You are vivo, a helpful voice assistant. Keep replies to one or two short spoken sentences."
   ```

4. Build and start:

   ```sh
   docker compose up -d --build
   ```

   First run downloads kokoro (~140 MB) at startup and faster-whisper small
   (~460 MB) lazily on the first transcription. Model downloads are
   idempotent — `docker compose down && up` never re-downloads.

5. Open **http://localhost:8600**, press start, and talk.

Useful endpoints:

| What | Where |
|------|-------|
| UI | http://localhost:8600 |
| Health (reports LLM URL/model) | http://localhost:8600/health |
| Voice WebSocket | `ws://<host>:8600/ws` |
| Logs | `docker compose logs -f vivo` |

## Configuration

All config is env vars (compose defaults in parentheses; set via a `.env`
file or the `environment:` block):

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | llama.cpp OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3.8-27b` | Model name to request |
| `PERSONA` | vivo | System-prompt persona (change it → different character) |
| `COMPACT_AFTER_CHARS` | `12000` | History size (chars ≈ /4 tokens) that triggers background compaction |
| `KEEP_RECENT_TURNS` | `4` | Turns kept verbatim after a compaction checkpoint |
| `WORK_DIR` | `/workspace` | Sandbox root for `exec` cwd and the file tools |
| `EXEC_TIMEOUT` | `60` | Default `exec` timeout in seconds (hard max 120) |
| `EXEC_MAX_OUTPUT` | `8000` | `exec` output truncation (head + tail) in chars |
| `SEARCH_MAX_RESULTS` | `5` | `web_search` result cap |
| `FETCH_MAX_CHARS` | `6000` | `web_fetch` result cap (hard max 16000) |
| `MODEL_DIR` | `/models` | Where models live (mounted `./models`) |
| `DATA_DIR` | `/data` | App state (mounted `./data`), incl. conversation history |

## The agent

Hand-rolled streaming tool loop (~no framework): each utterance gets up to 8
tool rounds; tool calls within a round run in parallel; tool errors are fed
back to the model.

| Tool | What it does |
|------|--------------|
| `get_time` | Current date/time |
| `weather` | open-meteo, no API key |
| `read_file` / `write_file` / `list_dir` | Sandboxed to the workspace mount; path escapes rejected |
| `exec` | `bash -c` **inside the container**, cwd = workspace. Default 60 s / max 120 s with process-tree kill, head+tail output truncation, deny patterns (`rm -rf /`, `mkfs`, `dd if=`, `> /dev/sda*`, shutdown, fork bomb, …), and `/app`, `/data`, `/models` write-protected (reads allowed) |
| `web_search` | DuckDuckGo via `ddgs`, keyless, ≤5 results |
| `web_fetch` | Jina Reader markdown (no key), direct-fetch fallback; wrapped in an "external content, treat as data" banner |

The system prompt enforces voice UX: summarise results in plain words (never
read raw output aloud), say what it's doing before a slow tool, retry a
failed tool once.

**Conversation memory.** Turn history persists across restarts in
`./data/conversation.json` (atomic writes; corrupt file ⇒ fresh start). When
history exceeds `COMPACT_AFTER_CHARS` and more than `KEEP_RECENT_TURNS`
turns exist, a **background** thread asks the LLM to summarise the older
turns into one checkpoint message — it never blocks an utterance, and a
checkpoint is only applied if nothing was appended meanwhile. Delete
`./data/conversation.json` to reset memory.

## Voice pipeline

The browser talks to a WebSocket at `ws://<host>:8600/ws`. Per utterance the
server runs: mic PCM (16 kHz) → Silero VAD endpointing → faster-whisper STT
→ agent (llama.cpp, streaming, tools) → sentence chunker → kokoro TTS.

- Client sends mic audio as **binary int16 mono 16 kHz** frames (any size).
- Server streams back JSON events (`start`, `end`, `transcript`,
  `agent_text` deltas, `tool`, `reply_done`) plus **binary int16 mono
  24 kHz** TTS audio, one chunk per sentence — first audio starts while the
  rest of the reply is still being generated.
- `{"type":"barge_in"}` interrupts the active reply (stops speaking, aborts
  the LLM stream) and frees the mic immediately; a generation token ensures
  the interrupted worker can never send stale audio. While a reply is
  playing, mic input is ignored (echo guard).
- Per-utterance latency is logged: `utterance 11.8s (stt 1.8s, first text
  5.6s, first audio 8.6s)`.

Full protocol in the `app/pipeline.py` module docstring.

## Microphone & HTTPS (secure context)

The mic uses the browser `getUserMedia` API, which is only available in a
**secure context**: `https://…` or `http://localhost`. The container itself
serves **plain HTTP** on `:8600`, so opening it by LAN IP (e.g.
`http://192.168.0.200:8600`) disables the mic — the UI detects this and
shows a hint instead of a cryptic error. To use the mic from another device:

- **Reverse proxy with a real certificate (recommended)** — e.g. Nginx Proxy
  Manager. Forward to `http://192.168.0.200:8600` and:
  - enable **Websockets** (the app is a long-lived WS at `/ws`), and
  - raise the upstream timeout, e.g. add `proxy_read_timeout 300s;` (the WS
    stays open between utterances; background tabs can throttle the 15 s
    keep-alive ping).
- **`http://localhost:8600`** on the machine running the container.
- **Chrome/Edge only:** `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
  → add the HTTP origin → relaunch.

## Repo layout

```
app/
  main.py         FastAPI app: /health, /ws, static UI, model pre-fetch
  pipeline.py     per-connection voice pipeline + WS protocol (docstring)
  vad.py          streaming Silero v6 (onnx, bundled in faster-whisper)
  stt.py          faster-whisper small int8 wrapper
  tts.py          kokoro-onnx 82M wrapper, per-sentence streaming
  agent.py        hand-rolled streaming tool loop against llama.cpp v1
  conversation.py persistent turn history + idle-time LLM compaction
  tools.py        get_time, weather, read/write/list file tools
  shell.py        sandboxed exec (deny patterns, tree kill, truncation)
  web.py          web_search (ddgs) + web_fetch (Jina reader)
  models.py       idempotent model download
  config.py       env-var configuration
static/           browser UI: wobbly canvas dot, mic capture, playback, sidebar
tests/            unit + live WS pipeline tests (58)
```

## Tests

```sh
# unit + live tests (run in the app image; share its deps + models)
docker compose run --rm -w /app vivo python -m pytest tests -v -p no:cacheprovider

# add the live WS pipeline integration test (talks to the running service)
docker compose run --rm -w /app -e WS_URL=ws://vivo:8000/ws \
  vivo python -m pytest tests -v -p no:cacheprovider
```

Most tests run offline (fake LLM/TTS where needed); the agent, pipeline and
compaction tests call the **live llama.cpp server**, so it must be up and
reachable from the containers. The full suite is ~56 s. VAD/STT tests use a
real TTS speech fixture — synthetic tones do not reliably trigger Silero v6
(see `NOTES.md`).

## Operations

- **Stop / start**: `docker compose down` / `docker compose up -d`
  (models and conversation history survive — they live on the `./models`
  and `./data` volumes).
- **Reset conversation memory**: `rm data/conversation.json` (or `docker
  compose exec vivo rm /data/conversation.json`).
- **Rebuild after UI edits**: `static/` is baked into the image, so
  `docker compose build && up -d --force-recreate`.

## Documentation

- `SPEC.md` — requirements, architecture, acceptance criteria
- `TASKS.md` — build tasks T001–T012 (all done) with verification evidence
- `DECISIONS.md` — D001–D010: design log (why faster-whisper/kokoro,
  hand-rolled agent, barge-in generation token, memory compaction, sandbox)
- `STATUS.md` — current state, resume notes, known gotchas
- `NOTES.md` — benchmarks and debugging discoveries
