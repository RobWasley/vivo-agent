import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, config_schema, models, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("vivo")


class ConfigPayload(BaseModel):
    values: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(models.ensure_models, config.MODEL_DIR, config.DATA_DIR)
    app.state.engines = pipeline.Engines()
    log.info("vivo ready (persona: %s)", config.PERSONA[:60])
    yield
    app.state.engines.close()


app = FastAPI(title="vivo", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_base_url": config.LLM_BASE_URL,
        "llm_model": config.LLM_MODEL,
        "model_dir": config.MODEL_DIR,
    }


@app.get("/api/config")
async def get_config(request: Request):
    """Effective settings + the schema the settings pane renders from (T019)."""
    engines = request.app.state.engines
    voices = await asyncio.to_thread(engines.tts.voices)
    return {
        "values": config.effective_dict(),
        "schema": config_schema.SCHEMA,
        "voices": voices,
        "path": config.CONFIG_PATH,
    }


@app.post("/api/config")
async def post_config(body: ConfigPayload, request: Request):
    """Validate, persist to vivo.toml, hot-apply, and broadcast to open sessions."""
    engines = request.app.state.engines
    voices = tuple(await asyncio.to_thread(engines.tts.voices))
    errors = config_schema.validate(body.values, voices)
    if errors:
        raise HTTPException(status_code=400, detail=errors)
    data = config.effective_dict()
    for section, keys in config_schema.coerce(body.values).items():
        data[section].update(keys)
    config.write_config(data)
    config.refresh()
    pipeline.apply_config(engines)
    pipeline.broadcast_config()
    log.info("settings saved via UI")
    return {"ok": True, "values": config.effective_dict()}


@app.post("/api/voice")
async def upload_voice(request: Request, file: UploadFile, name: str = Form("voice")):
    """Store a reference clip as a new voice, make it active, and start
    cloning it in the background (T022). Returns the new voice's name."""
    engines = request.app.state.engines
    data = await file.read()
    try:
        slug, dur = await asyncio.to_thread(engines.tts.save_voice, data, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    cfg = config.effective_dict()
    cfg["voice"]["tts_voice"] = slug
    config.write_config(cfg)
    config.refresh()
    engines.tts.set_voice(slug)
    # Pre-encode off the request path; a daemon thread so a slow first-time
    # model load can never hold the process open at shutdown.
    threading.Thread(target=engines.tts.prime, args=(slug,), daemon=True).start()
    pipeline.broadcast_config()
    log.info("voice %r uploaded (%.1fs) and made active", slug, dur)
    return {"ok": True, "voice": slug, "duration_s": round(dur, 2)}


@app.get("/api/voices/{name}")
async def voice_clip(name: str, request: Request):
    """The raw reference clip (preview in the settings pane)."""
    path = request.app.state.engines.tts.voice_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown voice: {name}")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.delete("/api/voices/{name}")
async def delete_voice(name: str, request: Request):
    """Delete a reference clip. The active voice is protected (switch first)."""
    engines = request.app.state.engines
    if name == engines.tts.voice:
        raise HTTPException(status_code=409, detail="cannot delete the active voice")
    try:
        await asyncio.to_thread(engines.tts.delete_voice, name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown voice: {name}")
    return {"ok": True}


@app.get("/api/sessions")
async def list_sessions(request: Request):
    """Named conversation sessions (T021): the active id + all sessions, most
    recently used first."""
    store = request.app.state.engines.sessions
    return {"active": store.active_id, "sessions": store.list_sessions()}


@app.post("/api/sessions")
async def create_session(request: Request):
    """Create a new session and make it the active one."""
    store = request.app.state.engines.sessions
    session_id = store.create()
    return {"active": store.active_id, "session_id": session_id}


@app.post("/api/sessions/{session_id}/activate")
async def activate_session(session_id: str, request: Request):
    """Make an existing session the active one."""
    store = request.app.state.engines.sessions
    try:
        store.set_active(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"active": store.active_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a session. If it was active, the most recently used remaining
    session becomes active (or a fresh one is created)."""
    store = request.app.state.engines.sessions
    try:
        store.delete(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"active": store.active_id}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await pipeline.serve_session(ws, ws.app.state.engines)


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True),
    name="static",
)
