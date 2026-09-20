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
from app.skills import SkillStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("vivo")


class ConfigPayload(BaseModel):
    values: dict


class SessionRenamePayload(BaseModel):
    name: str


class SkillPayload(BaseModel):
    name: str
    description: str
    instructions: str


class MemoryPayload(BaseModel):
    text: str
    core: bool = False


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
    engines = app.state.engines
    return {
        "status": "ok",
        "llm_base_url": config.LLM_BASE_URL,
        "llm_model": config.LLM_MODEL,
        "model_dir": config.MODEL_DIR,
        "tts_warm_state": engines.tts.warm_state,
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
    pipeline.broadcast_wake(engines.wake)
    log.info("settings saved via UI")
    return {"ok": True, "values": config.effective_dict()}


@app.post("/api/dream")
async def trigger_dream(request: Request):
    """Run an immediate dream pass for the local memory store."""
    engines = request.app.state.engines
    summary = await asyncio.to_thread(engines.dream_scheduler.trigger)
    return {"ok": True, "summary": summary}


@app.get("/api/memory")
async def get_memory(request: Request):
    facts = request.app.state.engines.memory.list()
    return {"facts": facts, "core_count": sum(item["core"] for item in facts)}


@app.post("/api/memory")
async def add_memory(body: MemoryPayload, request: Request):
    try:
        fact = request.app.state.engines.memory.add(body.text, core=body.core)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "fact": fact}


@app.put("/api/memory/{fact_id}")
async def update_memory(fact_id: str, body: MemoryPayload, request: Request):
    store = request.app.state.engines.memory
    try:
        fact = store.update(fact_id, body.text)
        if fact["core"] != body.core:
            fact = store.set_core(fact_id, body.core)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown memory fact") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "fact": fact}


@app.delete("/api/memory/{fact_id}")
async def delete_memory(fact_id: str, request: Request):
    try:
        request.app.state.engines.memory.delete(fact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown memory fact") from exc
    return {"ok": True}


def _skills() -> SkillStore:
    return SkillStore(Path(config.DATA_DIR) / "skills")


@app.get("/api/skills")
async def list_skills():
    """List the compact metadata used to select task-specific skills."""
    return {"skills": _skills().list()}


@app.get("/api/skills/{name}")
async def get_skill(name: str):
    try:
        return _skills().details(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/skills/{name}")
async def save_skill(name: str, body: SkillPayload):
    """Create or update a skill. The URL and payload names must agree."""
    if name.strip().lower() != body.name.strip().lower():
        raise HTTPException(status_code=400, detail="skill name does not match the URL")
    try:
        action = _skills().save(body.name, body.description, body.instructions)
        return {"ok": True, "action": action, "name": body.name.strip().lower()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/skills/{name}")
async def delete_skill(name: str):
    try:
        _skills().delete(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


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


@app.get("/api/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str, request: Request):
    """Stored transcript for one session: compacted summary + remaining turns.

    Turns are ordered oldest -> newest and each turn is returned as
    {"user": ..., "assistant": ...} so the UI can repopulate on reconnect.
    """
    store = request.app.state.engines.sessions
    try:
        conv = store.conversation_for(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    with conv.lock:
        turns = [
            {"user": user_text, "assistant": assistant_text}
            for user_text, assistant_text in conv.turns
        ]
        summary = conv.summary
    return {"id": session_id, "summary": summary, "turns": turns}


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


@app.post("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, body: SessionRenamePayload, request: Request):
    """Set or clear a display name for a session."""
    store = request.app.state.engines.sessions
    try:
        store.rename(session_id, body.name)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"ok": True, "session_id": session_id, "name": body.name.strip()}


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
