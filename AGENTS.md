# Secnd

- Read `README.md` and, if present locally, the ignored `plan.md` before expanding the bootstrap.
- Python 3.12 + FastAPI + SQLite; Vite + React + TypeScript. Work from the repository root.
- Use `uv sync --locked` and `npm ci`. Keep both lockfiles updated when dependencies change.
- Run `npm run check` for API and frontend changes. Use `docker compose up --build`
  to verify production packaging when the Dockerfile changes.
- Model work runs on CPU. Keep calibration/test data out of training, report measured
  results only, and preserve fixed evaluation splits across retraining rounds.
- Keep secrets, uploaded data, model files, and caches out of Git.

## graphify

- **graphify** (`~/.Codex/skills/graphify/SKILL.md`) — any input to knowledge graph.
  Trigger: `/graphify`. When invoked, read and follow that skill before other work.
