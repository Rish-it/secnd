# Workspace API

The versioned API is under `/api/v1/workspaces/{ws}`; use `default` locally.
The [OpenAPI contract](openapi.json) defines every request and response.
`/api/health` and `/api/preview` retain their existing behavior.

**Import, predict, feed, queue, answers and chat still return contract fixtures**
with `X-Secnd-Mode: fixture`. Uploads are validated but not saved, predictions
are illustrative and answers do not train a model.

**Metrics are published evaluations**, with `X-Secnd-Mode: evaluation`: a real
held-out card and curves for MTOP (default), BANKING77 or IPSA. They are not
workspace learning results. Model-version reads and pointer changes use the
workspace registry and return `X-Secnd-Mode: live`; a new workspace has no versions.

## Try the contract

Install dependencies using the root README, then start an API process:

```sh
uv run uvicorn backend.api:app --host 127.0.0.1 --port 8016
```

In a second terminal, set `API=http://127.0.0.1:8016/api/v1/workspaces/default`.
These three commands validate a CSV, request an illustrative decision and read
the example evaluation card:

```sh
printf 'text,amount,category\nTrain fare,25,Travel\n' | curl --fail-with-body -sS "$API/decisions:import" \
  -F 'file=@-;filename=demo.csv;type=text/csv' -F target_column=category

curl --fail-with-body -sS "$API/predict" -H 'Content-Type: application/json' \
  -d '{"items":[{"item_id":"invoice-1","text":"Train fare","amount":25}]}'

curl --fail-with-body -sS "$API/metrics"
```

| Method and path | Request | Response |
| --- | --- | --- |
| `POST /decisions:import` | Multipart `file`, `target_column` | `ImportResult` with row count and card |
| `POST /predict` | `{items: Item[]}`, 1–256 unique item IDs | `Decision[]` in input order |
| `GET /feed` | Optional `cursor`, `limit` (1–100) | `Page<Decision>`, automatic items |
| `GET /queue` | Optional `cursor`, `limit` (1–100) | `Page<Decision>`, uncertain items |
| `POST /answers` | `{item_id, label}`, required `Idempotency-Key` header | `AnswerResult` |
| `GET /metrics` | Optional `dataset=mtop\|banking77\|ipsa` | Published card and reliability/coverage–precision curves; empty learning history |
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
The fixture feed and queue each contain one item; the registry list starts empty. The fixture answer reports
`retrain: fixture`; the live contract also permits `retrain: queued` with the
current version and coverage. The fixture checks for an idempotency header but
does not persist or replay answers.

Versioned HTTP and request-validation errors use `{code, message}`. CSV uploads
are limited to 5 MiB, 100 unique columns and UTF-8 encoding. Workspace IDs use
1–64 letters, digits, underscores or hyphens, starting with a letter or digit.

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
Live model and chat handlers replace the fixtures while preserving these types.

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
automatic answers are certified. An empty `history` means no workspace learning
curve has been measured; the API does not substitute the trade-off curve.

Registry cards must validate as `EvalCard`, including measured `coverage`.
Cards written with only a threshold are unavailable until the training path
supplies a complete card. No coverage number is inferred from that threshold.
