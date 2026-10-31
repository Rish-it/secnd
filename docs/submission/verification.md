# Verification — 2026-10-31

All source changes are restricted to `frontend/`, `backend/demo_seed.py`,
`backend/tests/test_product_demo.py`, `README.md` and new `docs/submission/` files.
The worktree branch is `stage/c4-frontend`. Changes remain uncommitted.

- `npm run check`: passed. Ruff check and formatting, 344 backend tests, frontend
  lint and production build passed. Backend invocations used `UV_NO_SYNC=true`,
  a writable `/tmp/secnd-uv-cache`, and a temporary Python startup hook disabling
  dotenv loading. No `.env` or secret was inspected.
- After final demo/status and UI refinements, the four demo backend tests, Ruff,
  lint, build and the four Node evidence tests were rerun successfully.
- `npm run test:evidence --workspace @secnd/frontend`: four passing cases for
  personal evidence, public evidence precedence, missing personal proof and the
  separate calibration-history fallback.
- `git diff --check`: passed.
- `playwright test --config frontend/playwright.demo.config.ts --list`: discovers
  all six harness scenarios. Requests and assets are intercepted locally; this
  config requires a built frontend and does not require a listening web server.
- Browser execution was attempted. The first server-based attempt was refused
  when binding loopback. The socket-free attempt reached Chromium launch, which
  was refused by macOS sandbox Mach port permissions. All six cases stopped
  before their assertions ran. There are no successful browser assertions or
  new screenshots from this work.

Frontend lint exits successfully with five warnings: four React effect warnings
(including the retained Metrics component) and a ref-cleanup warning. They are
recorded here rather than described as zero-warning validation.

A TestClient smoke check returned successful health, pending/automatic queues,
autonomy, outcomes, activity, confirmation and undo responses. The demo tests also
prove that existing data is refused, live setup/generation/provider writes are
blocked, synthetic selection saves, and the stop state persists in its database.
These checks do not verify a real Meta integration or model quality.

Not run: live OAuth, live sends, paid generation, actual 400M serving, visual
browser review, video recording/upload, hosting, or Docker packaging. The
Dockerfile was unchanged. Existing installed packages were used; dependencies and
lockfiles were not changed. No Git write or commit was attempted.

Before submission, run:

```sh
npm run lint
npm run build
npm run test:evidence --workspace @secnd/frontend
./node_modules/.bin/playwright test --config frontend/playwright.demo.config.ts
npm run demo --workspace @secnd/frontend
```

Then verify desktop and 390px views in light and dark, focus order, screen-reader
announcements and touch targets; record the video and supply the URLs marked TBD.
The complete deliberate shortcuts are listed in `implementation-notes.md`.
