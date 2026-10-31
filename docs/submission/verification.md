# Verification — 2026-10-31

Run on the merged harness and frontend, after merging current `main`.

- `npm run check`: passed. Ruff check and formatting, 373 backend tests, frontend
  lint and production build.
- `npx playwright test frontend/tests/harness.spec.ts`: six passing scenarios in
  Chromium. They serve the built frontend and mock API responses through request
  interception, so they verify frontend behaviour, not a live integration.
- `npm run test:evidence --workspace @secnd/frontend`: personal evidence, public
  evidence precedence, missing personal proof and the separate calibration-history
  fallback.

The demo backend tests prove that existing data is refused, live setup, generation
and provider writes are blocked, synthetic selection saves, and the stop state
persists in its database. These checks do not verify a real Meta integration or
model quality.

Not run: live OAuth, live sends, paid generation, actual 400M serving, visual
review, video recording or upload, and hosting.

Before submission, run:

```sh
npm run check
npx playwright test frontend/tests/harness.spec.ts
npm run test:evidence --workspace @secnd/frontend
npm run demo --workspace @secnd/frontend
```

Then verify desktop and 390px views in light and dark, focus order, screen-reader
announcements and touch targets; record the video and supply the URLs marked TBD.
The deliberate shortcuts are listed in `implementation-notes.md`.
