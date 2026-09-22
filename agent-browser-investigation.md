# Agent Browser Investigation

Investigation into giving Vivo (voice agent) the ability to drive a real browser,
with the live view streamed back to the user so they can watch the interaction in
real time.

## Background

- Rob wants to see what the agent is doing in a browser, not just hear descriptions.
- The user's browser already receives a WebSocket stream for the agent's voice,
  so a second viewport stream could plausibly ride the same connection path.

## The tool: agent-browser (Vercel Labs)

- Repo: **github.com/vercel-labs/agent-browser** (an earlier `vercel/agent-browser`
  URL was wrong — it 404s), Apache-2.0, v0.38.1 (2026-09-16), docs at agent-browser.dev.
- **Pure Rust**: native Rust CLI + persistent Rust daemon driving Chrome directly
  over CDP. No Playwright, no Node.js at runtime (Node 24+ is only needed to build
  from source or run the npm postinstall).
- Install paths: npm `agent-browser` (bundles platform binaries; postinstall needs
  Node 24+), Homebrew, Cargo — and **GitHub releases: a standalone
  `agent-browser-linux-x64` binary (~18 MB)**. The release binary is the right path
  for Vivo's Node-less container; it was checksum-verified here (sha256
  `5100149a…205ea1`) and smoke-tested headless.
- `agent-browser install` downloads Chrome for Testing (Chrome 153); `--with-deps`
  also installs the Linux system libraries via the package manager (needs root →
  do it at Docker build time).
- Headless in a container: Chrome needs `--args "--no-sandbox"` (verified: launch
  fails without it, works with it). Set once in `~/.agent-browser/config.json`,
  not per command.

## Architecture: client–daemon

- The CLI parses commands and talks to a daemon that starts automatically on the
  first command and **persists between CLI invocations** (local socket).
  Consecutive `exec` tool calls from Vivo therefore join the *same* browser session —
  tabs, cookies, element refs, and auth state all survive between calls.
- Idle behaviour: after 1 h without commands the daemon saves restore state (if
  `--restore` is configured), closes the browser, and exits. Tunable via
  `--idle-timeout` / `AGENT_BROWSER_IDLE_TIMEOUT_MS` (`0` disables).
- Sessions: `--session <id>` isolates browser instances (own browser, cookies,
  history, auth). `--restore` auto-saves/restores cookies + localStorage
  (AES-256-GCM at rest with `AGENT_BROWSER_ENCRYPTION_KEY`).
- Agent workflow: `open <url>` → `snapshot -i` (accessibility tree with `@eN` refs,
  interactive elements only) → `click @e2` / `fill @e3 "…"` → re-snapshot after page
  changes. `batch` runs multiple commands in one CLI invocation (args or JSON on
  stdin) — matters for the voice loop, since each exec round-trip is an LLM turn.
  `screenshot --if-changed` skips unchanged images to save tokens.
- Also notable: `read [url]` (HTTP fetch preferring markdown + `llms.txt` lookup,
  no Chrome launch), `record start x.webm` (video with animated cursor, needs
  ffmpeg), experimental WebMCP page tools, `doctor` diagnostics, and
  `agent-browser skills get core` (version-matched workflow instructions).

## View streaming (the "watch" feature)

- **Every session automatically starts a WebSocket stream server** on an
  OS-assigned port. Pin it with `AGENT_BROWSER_STREAM_PORT=9248`;
  `stream status` shows port/state; `stream enable/disable` toggle at runtime.
- Wire protocol (JSON over WS):
  - server → client: `{"type":"frame","seq":N,"data":"<base64 jpeg>","metadata":
    {deviceWidth, deviceHeight, scrollOffsetX/Y, timestamp, …}}` plus
    `{"type":"url","url":…}` on navigation;
  - client → server: `input_mouse`, `input_keyboard`, `input_touch` events
    (two-way "pair browsing" is built in), and `config` with per-client `maxFps`
    (1–120) and `pacing: "ack"` (one-frame-at-a-time acking for slow clients).
    Both settings can also be declared in the WS URL (`?pacing=ack&maxFps=10`).
  - Delivery is **latest-first**: the server picks the newest frame at send time;
    stale frames are skipped, never queued. Input events dispatch on a dedicated
    task, so a click stays responsive even while a frame is mid-write.
- Frame cost (JPEG): 1280×720 q80 ≈ 54 KB/frame; q20 ≈ 25 KB; 640×360 q20 ≈ 9 KB.
  Tunables: `AGENT_BROWSER_STREAM_QUALITY` (default 80),
  `AGENT_BROWSER_STREAM_MAX_WIDTH/MAX_HEIGHT`.
- **Built-in dashboard** (`agent-browser dashboard start`, port 4848): ready-made
  web viewer — live viewport, command activity feed, console output, session
  creation. Token-guarded access URLs for non-loopback origins
  (`--allowed-origins`); session stream traffic is proxied internally, so session
  ports never need exposing. The zero-code option for a watch window (considered,
  not chosen — see below).

## MCP mode

- `agent-browser mcp` runs an MCP stdio server (protocol 2025-11-25) with tool
  profiles (`core` default; `network`, `state`, `debug`, `tabs`, `react`,
  `mobile`, `all`).
- Vivo is **not** an MCP host (custom tool registry in app/tools.py), so MCP mode
  is not a direct plug-in. The integration point is the **CLI via the existing
  `exec` tool**. (MCP mode would matter for MCP-capable assistants that want to
  share the same browser.)

## Security (relevant: LLM-driven browser in a container)

- `--allowed-domains` (wildcards) — blocks navigation, sub-resource requests,
  WebSocket/EventSource, `sendBeacon`, and WebRTC to non-allowed origins.
- `--content-boundaries` — delimits page output as untrusted (same pattern Vivo
  already uses: `UNTRUSTED_BANNER` in app/web.py).
- `--confirm-actions` / `--action-policy` — gate risky action categories (eval,
  download, …). `--max-output` caps context flooding.
- Auth vault: credentials stored encrypted; the LLM never sees passwords.

## Fit with Vivo (from the code)

- `app/shell.py run_shell()`: bash -c, cwd=/workspace, 60 s default / 120 s max
  timeout, output truncation, deny list. agent-browser commands pass the deny list
  unmodified; the daemon (a background process) persists across exec calls.
- `app/tools.py`: the `exec` tool is the integration point; optionally point its
  description at the browser skill.
- `app/pipeline.py` + `app/main.py`: single `/ws` voice protocol (typed JSON frames
  + binary PCM). The viewport rides a **separate `/ws/browser` endpoint** with its
  own tiny protocol — cleaner than overloading the voice channel.
- `static/` UI: a side-panel pattern already exists (transcript aside); the
  browser panel mirrors it.
- Docker: python:3.11-slim, non-root `appuser`, no Node, /app /data /models
  read-only, WORK_DIR=/workspace, host port 8600. `websockets>=13` is already in
  requirements.txt — the relay client needs no new dependencies.

## Chosen architecture (decided with Rob)

1. **Embedded side panel** in the Vivo UI (not a separate dashboard tab).
2. **Watch-only first** — no input events from the panel; the stream protocol
   supports adding mouse/keyboard later without rework.
3. **Ephemeral browser state** — no `--restore`; fresh browser per container start.

Flow: Vivo drives agent-browser via `exec` → daemon's stream server on
`127.0.0.1:9248` (container-internal, never exposed) → new FastAPI `/ws/browser`
endpoint relays `frame`/`url` frames to the UI panel (and only a `maxFps` config
back) → the panel renders JPEGs into an `<img>`.

## Implementation plan

1. **Dockerfile**: create `appuser` first; fetch the pinned release binary
   (v0.38.1, sha256-verified) → `/usr/local/bin/agent-browser`; as root
   `agent-browser install --with-deps`, move the browser tree to
   `/home/appuser/.agent-browser`, chown; as appuser write `config.json` with
   `{"args": ["--no-sandbox"]}`. Image grows ~200 MB.
2. **docker-compose.yml**: `AGENT_BROWSER_STREAM_PORT=9248` (+ optional
   quality/size env vars). No new published ports, no new volumes.
3. **app/browser_stream.py** (~100 lines): `/ws/browser` handler — client WS to
   `127.0.0.1:9248` on connect; forward `frame`/`url` out; accept only
   `browser_fps` in; `browser_status: offline` + ~3 s polling while no stream
   server is up; close upstream on disconnect. Endpoint registered in app/main.py.
4. **static/**: sidenav toggle + browser panel (aside) with URL header and
   rAF-gated `<img>` updates, "not browsing yet" state; connect on open,
   disconnect on close. ~0.3–0.5 MB/s at ~10 fps on LAN.
5. **data/skills/browser/SKILL.md**: core-workflow skill (open → snapshot -i →
   act by ref → re-snapshot; use `batch`; `screenshot --if-changed`; note that
   the user is watching).
6. **Cleanup/verification**: `docker compose build` + up; pytest; live check —
   ask Vivo to open a page and watch the panel.

## Bottom line

Feasible and low-risk. agent-browser's CLI+daemon model is a natural fit for
Vivo's existing `exec` tool, and its built-in WebSocket screencast provides the
live view with no custom capture code — the work is a small in-process relay plus
a UI panel. The container build adds ~200 MB and one pinned binary; no new
runtime dependencies.
