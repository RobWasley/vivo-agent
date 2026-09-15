import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from app import config, models, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("vivo")


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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await pipeline.serve_session(ws, ws.app.state.engines)


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True),
    name="static",
)
