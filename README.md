# vivo

A CPU-only, single-container voice agent. Talk to it hands-free through a
browser: it listens (Silero VAD), transcribes (faster-whisper small int8),
answers with a small hand-rolled agent pointed at your llama.cpp v1 endpoint
(shell, file and web tools, named conversation sessions), and speaks back in a
cloned voice (LuxTTS voice cloning). No GPU, no API keys, one Docker container.

```
Browser (mic + playback + wobbly dot UI)
   │  WebSocket (PCM 16 kHz up, PCM 48 kHz down)
   ▼
FastAPI (uvicorn)
   ├── VAD: Silero v6 → speech endpointing
   ├── STT: faster-whisper small int8
    ├── Agent: streaming tool loop → llama.cpp v1 (thinking on)
    │     ├── 8 tools: exec, files, web, time, weather
    │     └── spoken fillers ("one moment…") while reasoning
    ├── Memory: named conversation sessions, idle-time LLM compaction
   └── TTS: LuxTTS voice cloning (48 kHz), sentence-chunked streaming
```

## Prerequisites

- Docker + Compose.
- A CPU with ≥8 cores recommended. Benchmarks on a 24-core host: STT
  4.6× realtime, TTS ~0.2× realtime (see `NOTES.md` for the full numbers).
- A **running llama.cpp server** with an OpenAI-compatible `/v1` endpoint:
  - tool calling enabled (`--tools all`),
  - a Qwen3-style chat model whose template honours
    `chat_template_kwargs={"enable_thinking": …}` (verified with
    `qwen3.8-27b`). The agent runs with **thinking on**: reasoning tokens
    stream first (surfaced as `reasoning`/`reasoning_content`, never spoken
    or stored) and vivo speaks short filler phrases during the silence — see
    [Thinking & fillers](#thinking--fillers).
- ~2.5 GB free disk for models (auto-downloaded on first run; the LuxTTS
  voice model is the bulk of it).
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
   `0.0.0.0:8080` with model `qwen3.8-27b`. All tunables (LLM, persona,
   voice, VAD, …) live in **`vivo.toml`** — or just in the web UI: the
   **settings** button opens a pane for every setting (tooltips, sliders,
   voice dropdown); saving writes `vivo.toml` and applies what it can
   without a restart. Env vars still win over the file, so a `.env` next to
   `docker-compose.yml` (e.g. `LLM_BASE_URL=…`) remains the escape hatch;
   see [Configuration](#configuration).

4. Build and start:

   ```sh
   docker compose up -d --build
   ```

    First run downloads the LuxTTS voice model (~1.2 GB) and whisper-tiny
    (~150 MB) into the HF cache at startup, and faster-whisper small
    (~460 MB) lazily on the first transcription. Model downloads are
    idempotent — `docker compose down && up` never re-downloads.

5. Open **http://localhost:8600**, press start, and talk.

Useful endpoints:

| What | Where |
|------|-------|
| UI | http://localhost:8600 |
| Health (reports LLM URL/model) | http://localhost:8600/health |
| Voice WebSocket | `ws://<host>:8600/ws[?session=<id>]` |
| Conversation sessions (list / create / activate / delete) | `http://localhost:8600/api/sessions` |
| Logs | `docker compose logs -f vivo` |

## Configuration

All tunables live in **`vivo.toml`** at the project root. Every key keeps its
historical env-var name and can be overridden by one: **env var > file >
built-in default**. Set overrides in a `.env` file next to
`docker-compose.yml` (or the `environment:` block); unset/empty env vars fall
through to the file.

### Settings pane (UI)

The **settings** button in the UI footer opens a modal rendered entirely from
the server's schema (`GET /api/config`), so it always matches what the code
understands. Controls are typed: sliders for numerics (with a live value
readout), dropdowns for enums, checkboxes, and textareas for the filler
phrases and prompts. Each row carries a `(?)` tooltip and an **apply** badge —
`live` (takes effect immediately, even mid-reply), `next session` (next
utterance/connection), or `restart` (needs a container restart).

The **voice** row is special: voices are reference clips, so it pairs the
dropdown with **upload** (pick an audio file), **record** (capture 3–30 s from
the mic), **preview** (play the selected clip), and **delete**. An upload is
decoded to 16-bit mono WAV in the browser, stored under `data/voices/`, made
the active voice, and cloned in the background so the first reply in the new
voice is already warm.

**Save** POSTs the whole snapshot to `POST /api/config`. The server validates
it against the schema (a `400` lists every problem), rewrites `vivo.toml` on
the host (the file is mounted read-write), reloads the config, hot-applies the
live-applicable values to the running engines, and re-broadcasts the
`[barge_in]` `config` frame to every open WebSocket so connected browsers
pick up new barge timings without a reload. Values that need a restart (STT
model/compute/threads) are written but take effect on the next
`docker compose up`.

Hand-editing the file still works — it's the same source of truth — but the
pane is the low-friction path.

```toml
# vivo.toml — every key, shown at its default (env-var name)
[llm]
base_url = "http://host.docker.internal:8080/v1"   # LLM_BASE_URL
model = "qwen3.8-27b"                               # LLM_MODEL
thinking = true                                      # LLM_THINKING
max_tokens = 600                                     # LLM_MAX_TOKENS

[persona]
blurb = "You are vivo, a helpful voice assistant. …"  # PERSONA
system_prompt = """…voice-UX rules, appended to the blurb…"""  # file only

[filler]
first_after_s = 2.0                                  # THINK_FILLER_FIRST_AFTER
interval_s = 8.0                                     # THINK_FILLER_INTERVAL
phrases = ["Let me think about that.", …]            # THINK_FILLER_PHRASES (comma-separated)

[voice]
tts_voice = "default"           # TTS_VOICE — reference clip name in <data>/voices
tts_speed = 1.0                 # TTS_SPEED
sentence_pause_s = 0.25         # TTS_SENTENCE_PAUSE — silence appended after each sentence
sentence_max_chars = 90         # SENTENCE_MAX_CHARS — hard split without punctuation
clause_max_chars = 40           # CLAUSE_MAX_CHARS — clause split for early TTS start (0 = off)
tts_queue_size = 2              # TTS_QUEUE_SIZE — LLM→TTS sentence queue bound

[stt]
model = "small"                 # STT_MODEL (tiny/base/small/medium/large-v3)
compute_type = "int8"           # STT_COMPUTE_TYPE
cpu_threads = 8                 # STT_CPU_THREADS
language = "en"                 # STT_LANGUAGE
beam_size = 1                   # STT_BEAM_SIZE

[vad]
threshold = 0.5                 # VAD_THRESHOLD — Silero speech probability
min_speech_ms = 200             # VAD_MIN_SPEECH_MS
min_silence_ms = 400            # VAD_MIN_SILENCE_MS
reopen_ms = 600                 # VAD_REOPEN_MS — breath-tolerant endpoint (0 = classic)
speech_pad_ms = 200             # VAD_SPEECH_PAD_MS
max_speech_s = 30.0             # VAD_MAX_SPEECH_S

[barge_in]                      # pushed to the browser over the WS handshake
level_threshold = 0.25          # BARGE_LEVEL_THRESHOLD
sustain_ms = 250                # BARGE_SUSTAIN_MS
cooldown_ms = 700               # BARGE_COOLDOWN_MS

[memory]
compact_after_chars = 12000     # COMPACT_AFTER_CHARS (≈ /4 tokens)
keep_recent_turns = 4           # KEEP_RECENT_TURNS

[agent]
max_tool_rounds = 8             # MAX_TOOL_ROUNDS
exec_timeout_s = 60             # EXEC_TIMEOUT (hard max below)
exec_max_timeout_s = 120        # EXEC_MAX_TIMEOUT
exec_max_output_chars = 8000    # EXEC_MAX_OUTPUT
search_max_results = 5          # SEARCH_MAX_RESULTS
fetch_max_chars = 6000          # FETCH_MAX_CHARS (hard max 16000)
```

Deployment paths stay env-only in `docker-compose.yml` (container-internal):
`MODEL_DIR=/models`, `DATA_DIR=/data`, `WORK_DIR=/workspace` (the sandbox
root for `exec` cwd and the file tools), `HF_HOME=/models/hf` (the Hugging
Face cache the LuxTTS + whisper-tiny weights land in), plus the port mapping.
`TTS_CPU_THREADS` (default 8) is the one deployment-only tuning knob: the
Omp threads LuxTTS uses for CPU inference.

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

**Conversation memory — named sessions.** Every conversation is a
*session*: its turn history persists in `./data/sessions/<id>.json` (atomic
writes; corrupt file ⇒ fresh start), indexed by `./data/sessions.json` (one
session is *active*; ids are `YYYYMMDD-HHMMSS` creation timestamps). In the
UI, the footer **dropdown** switches sessions and **new** starts a fresh
one; each browser tab remembers its session (`localStorage`) and connects
with `/ws?session=<id>`, while a plain connection follows the active one.
The same operations exist as REST: `GET /api/sessions`, `POST /api/sessions`,
`POST /api/sessions/{id}/activate`, `DELETE /api/sessions/{id}` (deleting
the last session creates a fresh active one). Within a session, when history
exceeds `COMPACT_AFTER_CHARS` and more than `KEEP_RECENT_TURNS` turns exist,
a **background** thread asks the LLM to summarise the older turns into one
checkpoint message — it never blocks an utterance, and a checkpoint is only
applied if nothing was appended meanwhile. A pre-sessions
`./data/conversation.json` is migrated automatically on first start.

## Voice pipeline

The browser talks to a WebSocket at `ws://<host>:8600/ws`. Per utterance the
server runs: mic PCM (16 kHz) → Silero VAD endpointing → faster-whisper STT
→ agent (llama.cpp, streaming, tools) → sentence chunker → LuxTTS TTS.

**Voice cloning.** The TTS engine is LuxTTS (zipvoice), which speaks in a
cloned voice: each named voice is a 3–30 s reference clip stored as
`data/voices/<name>.wav`. The clip is transcribed (whisper-tiny) and
feature-extracted once, on first use or upload (~1 s), and every synthesis
after that is a flow-matching generate at ~0.2× realtime on CPU, outputting
48 kHz PCM. A `default` clip ships in the repo and is seeded into
`data/voices/` on first run; add your own from the settings pane (upload or
mic record).

- Client sends mic audio as **binary int16 mono 16 kHz** frames (any size).
- Server streams back JSON events (`session` [first frame — the conversation
  this connection is bound to], `config`, `start`, `end`, `transcript`,
   `agent_text` deltas, `tool`, `reply_done`) plus **binary int16 mono
   48 kHz** TTS audio, one chunk per sentence — first audio starts while the
  rest of the reply is still being generated. Connect with `/ws?session=<id>`
  to bind a specific conversation; `/ws` follows the active one.
- `{"type":"barge_in"}` interrupts the active reply (stops speaking, aborts
  the LLM stream) and frees the mic immediately; a generation token ensures
  the interrupted worker can never send stale audio. `{"type":"session"}`
  starts a new conversation and `{"type":"session","id":…}` switches to an
  existing one (both also set it active, for other tabs); each is answered
  with a `session` frame (unknown id → `error`). A switch affects the next
  utterance only — an in-flight reply keeps the conversation it started in. While a reply is
  playing, mic input is ignored (echo guard) — except for **auto barge-in**:
  the browser watches the mic level and sends `barge_in` on its own when your
  voice (after the browser's echo cancellation) stays above a threshold long
  enough. The timings are not hardcoded: the server pushes its
  `[barge_in]` settings to the browser in a `config` message on connect.
- Per-utterance latency is logged: `utterance 11.8s (stt 1.8s, first text
  5.6s, first audio 8.6s)`.

Full protocol in the `app/pipeline.py` module docstring.

## Thinking & fillers

The agent runs the LLM in **thinking mode** (`LLM_THINKING=1`, the default):
reasoning tokens stream before the spoken answer. They are surfaced as
`ReasoningDelta` items — **never spoken, never sent to the client, never
stored in history** — and the UI's "Thinking…" pill already covers display.

While no speakable text has arrived for `THINK_FILLER_FIRST_AFTER` seconds
(prefill, reasoning, or a tool round), vivo speaks a short filler phrase
("One moment.", "Working on it.", …), and repeats every
`THINK_FILLER_INTERVAL` seconds while the silence goes on — a long reasoning
stretch keeps getting fresh filler audio. Filler enqueue never blocks (a full
TTS queue means there's already audio to play) and barge-in stops it at once.
Filler audio is excluded from the `first_audio` bench metric. Set
`THINK_FILLER_FIRST_AFTER` very high (or `LLM_THINKING=0`) for a quiet
assistant.

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
   tts.py          LuxTTS voice-cloning wrapper, per-sentence streaming (48 kHz)
   agent.py        hand-rolled streaming tool loop against llama.cpp v1
   conversation.py named conversation sessions + idle-time LLM compaction
   tools.py        get_time, weather, read/write/list file tools
   shell.py        sandboxed exec (deny patterns, tree kill, truncation)
   web.py          web_search (ddgs) + web_fetch (Jina reader)
   models.py       model pre-fetch (HF repos) + default-voice seed
  config.py       vivo.toml + env-var loader (env > file > default)
  config_schema.py  typed schema: pane, validation, file comments, write-back
static/           browser UI: wobbly canvas dot, mic capture, playback, sidebar, settings pane
vivo.toml         central configuration (rewritten by the settings pane, env-overridable)
tests/            unit + live WS pipeline tests (131)
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
reachable from the containers. The full suite is ~83 s (up to ~105 s under
LLM contention). VAD/STT tests use a
real TTS speech fixture — synthetic tones do not reliably trigger Silero v6
(see `NOTES.md`).

## Operations

- **Stop / start**: `docker compose down` / `docker compose up -d`
  (models and conversation history survive — they live on the `./models`
  and `./data` volumes).
- **Reset a conversation**: `DELETE /api/sessions/<id>` removes its file and
  index entry (deleting the last session creates a fresh active one). Wipe
  everything: `rm -r data/sessions data/sessions.json` — a fresh active
  session is created on next start.
- **Rebuild after UI edits**: `static/` is baked into the image, so
  `docker compose build && up -d --force-recreate`.

## Documentation

- `SPEC.md` — requirements, architecture, acceptance criteria
- `TASKS.md` — build tasks T001–T022 (all done) with verification evidence
- `DECISIONS.md` — D001–D018: design log (why faster-whisper/LuxTTS,
  hand-rolled agent, barge-in generation token, memory compaction, sandbox,
  thinking mode + spoken fillers, named conversation sessions, LuxTTS voice
  cloning)
- `STATUS.md` — current state, resume notes, known gotchas
- `NOTES.md` — benchmarks and debugging discoveries
