# Workspace API

The versioned API is under `/api/v1/workspaces/{ws}`. Requests need a session: register at
`/api/v1/auth/register`, create a workspace at `/api/v1/workspaces`, and send the session
cookie. [The README walkthrough](../../README.md#csv-to-a-live-decision) shows each step.
The [OpenAPI contract](openapi.json) defines every request and response.
`/api/health` stays public. `/api/preview` also needs a session.

**Import, predict, feed, queue and answers use WorkspaceService.** Import saves
historical labels and a fixed calibration holdout; predictions and human answers
are persisted. `X-Secnd-Mode` stays `fixture` until the workspace has a promoted
model, then becomes `live`. Before import, the example predict/feed/queue/answer
responses remain fixtures. A prepared workspace without a model has empty pages
and returns 409 on predict. Imports and accepted answers can be durable before a
model exists; the header describes model availability. Chat remains a fixture.

**Metrics default to the stored workspace:** the current calibration card, saved-version
history, training-label count and live `guarded` flag. Explicit `dataset=mtop`,
`banking77` or `ipsa` selects a published held-out test evaluation with
`X-Secnd-Mode: evaluation`. Before import the default is the MTOP evaluation.
Model-version reads and pointer changes use the workspace registry.

## Meta connectors

Signed WhatsApp, Messenger and Instagram text events use the existing workspace queue
and answers API. Receiving accounts require explicit workspace mappings. Connector
status lives at `GET /integrations/meta/status` under the workspace prefix.

Ads routes under `/integrations/ads` expose status, a saved auto-act preference,
read-only insight sync, guarded execution and a paginated action audit. The server
switch and workspace auto-act both default off. Human answers never bypass current
certification. Only explicitly allowlisted sandbox accounts are eligible.
See [configuration, action semantics and replay behavior](../integrations/meta.md).

## Import and predict

Install dependencies using the root README, then start an API process:

```sh
uv run uvicorn backend.api:app --host 127.0.0.1 --port 8016
```

In a second terminal, sign in and create a workspace, set
`API=http://127.0.0.1:8016/api/v1/workspaces/<id>`, and add `-b <cookie file>` to each request.
Use your own historical CSV with `text` or `description`, a separate target
column, and optional `item_id` (or `id`), `amount` and ISO `date`. IDs must be
unique and labels nonempty. At least two labels must remain in training after
the holdout. These three commands import it, predict, then read queued questions:

```sh
curl --fail-with-body -sS "$API/decisions:import" \
  -F 'file=@/path/to/labelled-decisions.csv;type=text/csv' -F target_column=category

curl --fail-with-body -sS "$API/predict" -H 'Content-Type: application/json' \
  -d '{"items":[{"item_id":"invoice-1","text":"Train fare","amount":25}]}'

curl --fail-with-body -sS "$API/queue"
```

| Method and path | Request | Response |
| --- | --- | --- |
| `POST /decisions:import` | Multipart `file`, `target_column` | `ImportResult` with row count and card |
| `POST /predict` | `{items: Item[]}`, 1–256 unique item IDs | `Decision[]` in input order |
| `GET /feed` | Optional `cursor`, `limit` (1–100) | `Page<Decision>`, automatic items |
| `GET /queue` | Optional `cursor`, `limit` (1–100) | `Page<Decision>`, uncertain items |
| `POST /answers` | `{item_id, label}`, required `Idempotency-Key` header | `AnswerResult` |
| `GET /metrics` | Optional `dataset=mtop\|banking77\|ipsa` | Workspace calibration card, version history, label count and guard; dataset selects published test curves |
| `GET /models` | None | `Page<ModelVersion>` |
| `POST /models/{v}:promote` | None | Select an existing registry version atomically |
| `POST /models/{v}:rollback` | None | `ModelVersion`; 404 for an unknown version |
| `POST /chat` | `{question}` | `ChatResult` |

`Item` contains `item_id`, `text`, and optional `amount` and ISO `date`.
`Decision` contains the predicted `label`, top-k `probabilities`, winning `prob`,
`auto`, nullable `threshold`, `model_version`, `reasons`, the input `item`, and
`spot_check`. Top-k probabilities keep their original values; omitted classes
can carry the remaining mass. `reasons` has `neighbours` and `merchant_memory`.
A null threshold means automatic decisions have not been certified.

Pages have `items` and a nullable `next_cursor`; null means there is no next page.
Unprepared fixture feed and queue each contain one example; the registry starts empty.
The fixture answer reports
`retrain: fixture`; the live contract also permits `retrain: queued` with the
current version and coverage. The fixture checks for an idempotency header but
does not persist or replay answers.

Versioned HTTP and request-validation errors use `{code, message}`. CSV uploads
are limited to 5 MiB, 100 unique columns and UTF-8 encoding. Workspace IDs use
1–64 letters, digits, underscores or hyphens, starting with a letter or digit.

## Start with label names

`POST /cold-start` accepts `{labels: ["Travel", "Food"], paid: true}` with
2–128 distinct names. It starts a background history job and returns progress;
`GET /cold-start` returns its saved status and editable examples. Alternatively,
`POST /cold-start:import` accepts a one-column UTF-8 `file` and `paid=true`.
The optional first row may be `label`, `labels` or `category`.

Writing new examples is paid, so it is off unless the server sets
`SECND_COLD_START_PAID=true`; label lists already in the cache still work without it.
Each workspace may spend at most $1, and all workspaces together at most
`SECND_COLD_START_TOTAL_USD` (default $3).

| Method and path | Behavior |
| --- | --- |
| `PATCH /cold-start/history/{example_id}` | Replace `{text, label}` in this workspace |
| `DELETE /cold-start/history/{example_id}` | Remove an invented example; resume preserves the deletion |
| `POST /predict` | CPU Ren reader suggestions over the current history, with human review |
| `POST /answers` | Persist a human correction into history; returns `retrain: history_updated` |

The Meta API provider is `meta-wonder`, using `MuseClient.examples(label, labels,
n=10)` and the existing `BudgetRun` and `docs/results/s5/spend-meta.csv` ledger.
The cumulative $1 workspace cap survives resume; the configured Meta budget and
shared provider cap also apply. Generation stops before an unaffordable request,
keeps completed examples and reports `stopped`. A large label list may receive
only partial history. Cache keys use the canonical label list, so completed
examples can be reused by another workspace without a new provider call.
An interrupted or failed attempted label is never automatically paid for again.

An existing workspace's label list cannot be replaced. Edits and deletions affect
only its invented history; another workspace's cached examples remain intact.
Human answers are durable and take precedence in the reader's history. Muse is
called during explicit generation only, never by prediction or answer routes.

These workspaces have zero certified coverage, a null threshold and unmeasured
accuracy/precision until independent calibration is supplied. Invented examples
do not certify accuracy. The reader checkpoint must already be installed as
described in the root README; missing files return 503 before a paid request.
The mode reports `X-Secnd-Mode: live` for real reader suggestions, while the
registry remains empty because no calibrated classifier version was fitted.

## Maintaining the contract

Add fields rather than renaming existing fields. The drift test compares the
running API's schema against the committed JSON. After an intentional change:

```sh
uv run python - <<'PY'
import json
from pathlib import Path
from backend.api import create_app
from backend.config import Settings
schema = create_app(Settings(_env_file=None)).openapi()
Path('docs/api/openapi.json').write_text(json.dumps(schema, indent=2, sort_keys=True) + '\n')
PY
uv run pytest -q backend/tests/test_api.py backend/tests/test_openapi.py
```

## Later integration points

Authentication belongs in a workspace router dependency.
Postgres replaces the SQLite store behind the same storage calls.
A worker queue replaces the per-workspace retraining dispatcher.
SDKs can be generated from the committed OpenAPI schema.
Chat remains a fixture; the other workspace routes persist real data.

## Published metrics and versions

```sh
curl --fail-with-body -sS "$API/metrics?dataset=mtop"
curl --fail-with-body -sS "$API/metrics?dataset=banking77"
curl --fail-with-body -sS "$API/metrics?dataset=ipsa"
curl --fail-with-body -sS "$API/models"
curl --fail-with-body -sS -X POST "$API/models/ren-1.0:promote"
curl --fail-with-body -sS -X POST "$API/models/ren-1.0:rollback"
```

The last two requests require that version to exist in this workspace. Missing
versions and missing evaluation files return 404. Invalid or incomplete cards
or curves return 503; the current pointer is unchanged when validation fails.
The card's version must match the selected registry version. Pointer changes
select an existing artifact; they do not retrain it or certify a new candidate.

Published evaluations come from `docs/results/s3/card-*.json` and matching
`curves-*.json`, included in the image. `SECND_EVALUATION_DIR` overrides their
directory. `MetricsResult.dataset` identifies the benchmark and `curves` holds
reliability bins (`confidence`, `accuracy`, `count`) and trade-off points
(`threshold`, `coverage`, `precision`). A missing threshold on IPSA means no
automatic answers are certified. Published evaluations have an empty `history`; workspace history contains one point
per saved promoted model, with its original card-file timestamp and calibration
coverage/precision. Refused candidates do not create a saved version or history
point. The API never substitutes a benchmark trade-off curve for workspace history.
The live `guarded` flag pauses automatic answers even if the saved card certified
coverage. The browser polls for background updates. Spot-check state is held in
memory and resets after a process restart.

Registry cards must validate as `EvalCard`, including measured `coverage`.
Cards written with only a threshold are unavailable until the training path
supplies a complete card. No coverage number is inferred from that threshold.

## Workspace storage and answer worker

SQLite schema version 2 migrates the bootstrap database at startup without
replacing existing data. Workspaces own their items, label history, prediction
history, idempotent answers and model-version metadata. Label sources are
`seed`, `answer`, `spot_check` and `muse_cold_start`; their weights are 1, 1, 1
and 0.3. Answer keys are unique within a workspace. Reusing a key with a
different item or label returns 409; replaying it returns the original result.

`backend.workspace.WorkspaceService.create(ws, labels, calib_frame, ...)`
persists a fixed calibration set separately from training labels. The service
rebuilds the Loop by replaying labels in arrival order, with their sources,
then retraining. Calibration item IDs are refused as answers. Every routed
Decision is saved with its version, confidence, threshold and spot-check flag.

For a prepared workspace, `POST /answers` commits the answer and label together
and returns `retrain: queued` before background fitting completes. A per-workspace
lock serialises model work; another workspace can train independently. Labels
survive a failed fit or process restart and replay on rebuild. One observed
class is retained without fitting until a second class arrives. The registry
remains the authority for the current model; the database mirrors its cards.

Imports reserve a fixed 20% for calibration (whole groups can change the row share): the newest rows for fully dated inputs,
otherwise whole merchant groups with seed 0. Calibration is stored separately
and never becomes training labels; workspace cards measure this fixed calibration
set, not a benchmark test set. Source benchmark splits remain unchanged. Importing
into an existing workspace returns 409 and preserves its labels and holdout.
A refused first candidate leaves zero coverage and no model.

Feed is newest first; queue is least confident first and includes spot-checks.
Pages expose the latest prediction per item, with its original item/evidence;
all prediction events remain in storage. Answered items leave the queue. Cursors
are scoped to workspace and route and remain stable when earlier questions are
answered. Invalid/cross-workspace cursors return 422.

Registry reads and pointer changes share the workspace retraining lock. The next
prediction loads a selected model head/temperature/threshold; the current workspace
label memory remains in use. Restart replays durable labels, restores the prior
model for the gate, and may fit a new candidate. This is not an immutable snapshot
of the historical merchant memory.

This service runs in one API process. Multiple processes require a durable
worker queue and coordinated model/cache writes. Prepared predictions return
409 when no model has passed the gate. Initial import fitting uses a worker
thread; subsequent answer fitting stays off the request path.

## Decision inbox

The app opens the selected workspace's inbox. `?workspace=<id>` selects a workspace.
Source badges distinguish WhatsApp,
Messenger, Instagram, Ads and workspace submissions.

Existing `/queue`, `/feed`, `/predict`, `/answers`, `/models` and rollback routes
serve the screen. Three small additions cover data those routes could not return:

- `GET /labels`: every allowed workspace label, including labels outside prediction top-k.
- `GET /overridden`: cursor-paginated decisions whose latest active human answer differs
  from the saved prediction. Corrected automatic answers leave `/feed`.
- `POST /answers:undo` with `{"idempotency_key":"<original answer key>"}`: retract the
  latest active answer in reverse order. Repeat undo is idempotent; replaying a retracted
  answer key returns 409. Other workspace keys return 404. The answer remains in the
  audit ledger, leaves training history, and its question or original automatic answer
  returns. The app retains its undo stack in browser session storage.

Saved predictions carry `source_app` and their model's `cp_lower_bound`. Evidence lists
nearest training-history examples; fixed calibration rows never enter that list.
Older predictions may have no saved bound or neighbours and show them as unavailable.
Undo restores the pre-answer model before rebuilding from active history, so a refused
candidate cannot leave the retracted answer in the active classifier. The gate remains
unchanged. Undo changes local labels; it does not recall a previously delivered message.
