# Product demo integration notes

The frontend uses the existing session, business ownership, Meta connection,
account-selection, autonomy, answers/undo, agent and outcomes APIs. No production
backend behaviour, training, evaluation split or policy calculation was changed.
The only backend implementation added here is the isolated synthetic demo.

## Evidence contract

Decision `basis` takes priority over legacy fields. Supported structured bases:

```json
{"kind":"personal_reviews","confirmed":61,"reviewed":61,"lower_bound":0.952}
```

```json
{"kind":"audited_public_history","lower_bound":0.994}
```

These are illustrative contract examples, not evaluation claims. A human-readable
`text`, `display_text` or `summary` can also be supplied. Older responses use
`autonomy` for personal-review counts; an authorized, complete public-history
model policy can use its saved lower bound. Other calibration history is labelled
as calibration history. Missing proof is shown as unavailable.
The synthetic feed carries an explicit personal-review basis; public-history
wording is exercised by mocked browser scenarios, not invented certification for
the synthetic business.

`/serving` supplies the running `model_id`. Optional `model_card_url` or `card_url`
becomes a model-card link. The current production response does not include either
URL, so no link is invented. The demo returns a synthetic model identifier and
explicitly discloses that no Ren checkpoint is loaded.

## Deliberate shortcuts

- Synthetic setup simulates sign-in and Meta connection. Normal sign-in and OAuth
  remain on the existing API path. The first business name is entered as part of
  sign-in; the API requires a short identifier without spaces.
- The first-run completion flag is local to this browser and business. Existing
  connected businesses with selected accounts go straight to Inbox. `?setup=1`
  explicitly reopens setup. Account selections and bars save through the API.
- The temporary demo persists reviews, corrections, undo, settings and the stop
  switch in SQLite during the process lifetime. Proposals and their confidence,
  executed status and ad observations are synthetic. No model fits or infers.
  Restarting creates a fresh database rather than resetting an existing one.
- The demo seeds synthetic review evidence directly. Its statistical bounds and
  remaining-confirmation counts are calculated by the existing autonomy code.
  This does not certify the real model or authorize a live action.
- Detailed permission changes and observed level drops are stored on this browser;
  corrections and agent events come from the business API. Cross-device permission
  history needs richer backend audit events. The UI labels this distinction.
- The primary navigation focuses on Inbox, Autonomy, Activity, Connections, Past
  decisions and About. The previous advanced metrics screens and paid starting-
  example generator are no longer entry points. CSV import remains available;
  the isolated demo refuses imports and generation.
- Connections uses the existing picker and status endpoints. The old manual Ads
  execution/debug controls are outside this product flow. Background scheduling,
  live execution permissions and stop-reset behaviour remain backend concerns.
- Proof percentages are rounded down for “at least” wording. Confidence is a model
  score; the progress bar uses the API's measured lower bound, not that score.
- Browser scenarios serve the built frontend and mock API responses through request
  interception, so they verify frontend behaviour without live OAuth or provider
  traffic. Separate backend tests verify the disposable seed and persisted reviews.
- Dependencies use the existing installed packages; no new packages or lockfile
  changes were needed.
- Platform marks are monochrome inline SVG. Fonts use the local system sans serif
  and Georgia; there are no font downloads or new dependencies.

## Remaining integration and submission work

Record and upload the video; supply the final product and video URLs. Verify real
Facebook Login for Business with the app's roles and permissions. Select and run
the actual 400M model only with its required local files and supported serving
configuration; its latency for this build is TBD. Visually review desktop and
narrow layouts, contrast and focus navigation. No live or paid verification is
claimed by this branch.
