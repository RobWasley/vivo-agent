# vivo-agent

Local voice AI assistant ("vivo"): FastAPI + WebSocket backend with a local
STT/LLM/TTS pipeline, plus a single-page web console (vanilla JS, no build
step).

## Layout

- `app/` — Python backend. `main.py` (FastAPI routes), `pipeline.py` (live
  voice pipeline, system tasks, reminder firing), `reminders.py` (reminder /
  task store + background scheduler), `tools.py` (LLM tools), `config.py` +
  `config_schema.py` (settings).
- `static/` — web console: `index.html`, `app.js` (one file, no modules),
  `style.css`. No framework, no build step.
- `tests/` — plain pytest suite.
- `vivo.toml` — central config; every key can be overridden by an env var
  (see `docker-compose.yml`).

## Environment: Docker only

The host Python does **not** have the app's dependencies (fastapi,
onnxruntime, faster-whisper, torch, ...). Do not run the app or the test
suite on the host — it will fail on imports.

Everything runs in the `vivo` container (`docker compose up -d`, image
`vivo-agent`):

- Host port `8600` → container port `8000` (web console + API).
- App code is **baked into the image** at `/app`; it is *not* a volume
  mount, so host edits don't reach a running container until you copy or
  rebuild.
- Volume mounts: `./models` → `/models`, `./data` → `/data`,
  `./vivo.toml` → `/app/vivo.toml`, `/home/rob/projects` → `/workspace`.

## Running tests

Always inside the container. Fast iteration without a rebuild:

```sh
docker cp app/. vivo:/app/app        # same for static/, tests/  (the "/." copies
docker cp static/. vivo:/app/static  # the *contents*; without it docker cp
docker cp tests/. vivo:/app/tests    # nests a sub-dir, e.g. /app/app/app)
docker exec vivo python -m pytest tests/test_reminders.py -q   # or tests/
```

Note: `vivo.toml` is a bind mount and is always live in the container; there
is nothing to copy. After a code copy the server must be restarted
(`docker restart vivo`) to pick it up — uvicorn runs without `--reload`.

Full cycle (bakes the current host tree into the image):

```sh
docker compose build && docker compose up -d
```

Notes:

- Some tests need live state: `test_pipeline.py` connects to a running
  server; STT/vad/wake tests load local models from `./models`.
- Reminder/task model lives in `app/reminders.py`: `repeat` is
  `once | interval | daily | weekly`; `interval` is a free-running span
  (`every`, minutes, 1–1440) that re-arms from the last fire.

## Style

No linter or typechecker is configured; match the existing code. Python:
`from __future__ import annotations`, module logger
`log = logging.getLogger("vivo.<module>")`. JS: plain script, element cache
in `el`, dialogs mirrored from the "skills-dialog" pattern.
