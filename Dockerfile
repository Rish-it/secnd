FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package.json
RUN npm ci
COPY frontend frontend
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.9.26 AS uv
FROM python:3.12-slim AS runtime
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.data/huggingface \
    TOKENIZERS_PARALLELISM=false \
    SECND_DATA_DIR=/app/.data \
    SECND_FRONTEND_DIST=/app/frontend/dist \
    PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY backend backend
RUN SECND_DATA_DIR=/opt/secnd python -m backend.encoders
COPY docs/results/s3/card-*.json docs/results/s3/
COPY docs/results/s3/curves-*.json docs/results/s3/
COPY --from=frontend /build/frontend/dist frontend/dist
RUN useradd --create-home --uid 10001 secnd \
    && mkdir -p /app/.data \
    && chown -R secnd:secnd /app/.data
USER secnd
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"]
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
