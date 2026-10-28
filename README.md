# Secnd

A companion that learns the decisions you make over and over. Teach it with past
examples; the goal is to handle confident decisions and ask you about the rest.

This is the foundation for the Meta AI hackathon project.

## What works today

- FastAPI with persistent local SQLite storage and a database health check.
- React and TypeScript workspace with Teach, Companion, and Metrics views.
- CSV upload, validation, a five-row preview, and target-column selection.
- Locked Python and JavaScript dependencies, automated checks, and one Docker image.

Model training, calibrated predictions, the answer/retrain loop, and Muse Spark
integration are next. Companion and Metrics show empty states until those exist.
The example CSV is eight illustrative rows for trying the preview, not a benchmark.
CSV previews are not saved; target selection currently stays in the browser.

## Run locally

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/), Python
3.12 (uv can install it), and Node.js 24 (`nvm use` reads `.nvmrc`).

From the repository root:

```sh
uv sync --locked
npm ci
cp .env.example .env
npm run dev
```

Open **http://localhost:3000**. API docs: **http://127.0.0.1:8000/docs**.
`npm run dev` starts both processes and stops the other if one exits. To run them
in separate terminals, use `npm run dev:api` and `npm run dev:web`.

Vite proxies `/api` to FastAPI. Production serves the built frontend and API
from FastAPI on the same port.

## Check and build

```sh
npm run check
```

This runs Python lint/format checks, API tests, frontend lint, TypeScript checks,
and a production frontend build. GitHub Actions runs the same checks on pushes
and pull requests. `uv.lock` and `package-lock.json` are the dependency sources
of truth; use `uv lock` or `npm install` intentionally when changing dependencies.

## Run one container

```sh
docker compose up --build
```

Open **http://localhost:8000**. The `secnd-data` volume keeps SQLite and future
embedding/model caches across restarts. The container runs as an unprivileged
user. PyTorch uses CPU wheels on Linux; no GPU or CUDA is required.

## Configuration

Copy `.env.example` to `.env` for local overrides. Local defaults work without
an API key. Configuration is read by the backend; never put provider keys in
frontend environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECND_DATA_DIR` | `.data` | SQLite and future model/cache files |
| `SECND_FRONTEND_DIST` | `frontend/dist` | Built frontend served when present |
| `SECND_MUSE_API_KEY` | empty | Reserved for the Muse client |
| `SECND_MUSE_BASE_URL` | empty | Reserved; confirm the official endpoint before integrating |
| `SECND_MUSE_MODEL` | empty | Reserved; confirm the available model before integrating |

The bootstrap makes no provider calls and does not download embedding model
weights. The sentence-transformers and scikit-learn dependencies are installed
so the model pipeline can be implemented next.

## API available now

| Endpoint | Behavior |
| --- | --- |
| `GET /api/health` | Checks SQLite; returns 503 if storage is unavailable |
| `POST /api/preview` | Multipart `file`; UTF-8 CSV, at most 5 MiB and 100 unique columns; row count and five preview rows |

The decision/training endpoints in the plan are not implemented yet.

## Next milestone

Choose a licensed transaction dataset with 5,000+ rows, descriptions, amounts,
and 10–30 categories. Implement fixed seed/stream/calibration/test splits,
cached `all-MiniLM-L6-v2` CPU embeddings, and logistic regression. Then print
version 0 metrics before connecting the UI to training.

The evaluation must distinguish held-out calibration from the final test set.
Coverage at the precision target and the learning curve remain the core demo.
