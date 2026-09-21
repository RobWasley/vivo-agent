FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    AGENT_BROWSER_VERSION=0.38.1 \
    AGENT_BROWSER_SHA256=5100149a1903211c889de4e545bf36d90803740cea4f99aa22651649f9205ea1

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first (the CUDA wheels from PyPI would bloat the image ~8 GB
# for a CPU deployment, SPEC: no GPU). LuxTTS's deps resolve around it.
# git is needed to install LinaCodec/LuxTTS from git (both are pure Python);
# removed again afterwards to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git sudo \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
        && curl -fsSL -o /usr/local/bin/agent-browser \
            "https://github.com/vercel-labs/agent-browser/releases/download/v${AGENT_BROWSER_VERSION}/agent-browser-linux-x64" \
        && echo "${AGENT_BROWSER_SHA256}  /usr/local/bin/agent-browser" | sha256sum -c - \
        && chmod 0755 /usr/local/bin/agent-browser \
        && useradd -m appuser \
        && agent-browser install --with-deps \
        && mv /root/.agent-browser /home/appuser/.agent-browser \
        && chown -R appuser:appuser /home/appuser/.agent-browser \
        && apt-get purge -y curl git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY app/ ./app/
COPY static/ ./static/
COPY voices/ ./voices/
COPY tests/ ./tests/
COPY vivo.toml ./

RUN mkdir -p /models /data \
    && printf '%s\n' '{"browser":{"args":["--no-sandbox"]}}' > /home/appuser/.agent-browser/config.json \
    && chown appuser:appuser /home/appuser/.agent-browser/config.json \
    && chown -R appuser:appuser /models /data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
