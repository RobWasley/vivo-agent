FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first (the CUDA wheels from PyPI would bloat the image ~8 GB
# for a CPU deployment, SPEC: no GPU). LuxTTS's deps resolve around it.
# git is needed to install LinaCodec/LuxTTS from git (both are pure Python);
# removed again afterwards to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY app/ ./app/
COPY static/ ./static/
COPY voices/ ./voices/
COPY tests/ ./tests/
COPY vivo.toml ./

RUN useradd -m appuser \
    && mkdir -p /models /data \
    && chown -R appuser:appuser /models /data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
