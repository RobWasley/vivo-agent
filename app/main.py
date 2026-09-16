import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, config_schema, models, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("vivo")


class ConfigPayload(BaseModel):
    values: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(models.ensure_models, config.MODEL_DIR)
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await pipeline.serve_session(ws, ws.app.state.engines)


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True),
    name="static",
)
