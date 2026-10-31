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

## Live Meta run

On 2026-10-31 a real Facebook Login for Business sign-in ran against the app through a temporary
HTTPS tunnel. It connected, stored the token encrypted and listed the person's one ad account. The
real selection path saved that account to a workspace, and a live insights request succeeded and
returned zero rows, because the account has no campaigns and has spent nothing. So the path is
proven; Ren has not decided on real ad data. The account has no Pages, so no webhook or message
reached the app. Ad changes stayed off and no send was attempted.

Not run: Meta messaging, live sends, ad changes, proposals from real ad data, paid generation, actual
400M serving, visual review, video recording or upload, and fixed hosting.

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
