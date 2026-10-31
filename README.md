# Secnd · Ren for Meta apps

Ren is an independently released decision model built for Meta's platform.
Its encoder learns from Meta's open assistant data; each workspace learns its own
labels and corrections. Confident decisions are handled automatically. The rest
become questions for a person, whose answers train the next classifier version.

The open encoder is [Secnd-AI/ren](https://huggingface.co/Secnd-AI/ren/tree/v1.0.0):
release **ren-1.0.0**, tag **v1.0.0**, pinned by immutable revision and
[SHA-256 manifest](backend/encoder-ren-1.0.0.json). The app defaults to the released encoder and workspace classifier.
The 150M CPU reader and optional 400M GPU reader are explicitly configurable.
[Serving configuration and required local reader files](docs/serving.md).

On the fixed English MTOP split at 2,000 workspace labels, the released encoder
and calibrated head measured **42.42% automatic coverage at 99.93% test precision**.
The associated 2,000 re-splits passed **99.45%** of the time. These are recorded
benchmark results, separate from an uploaded workspace's calibration card.
[Card](docs/results/s3/card-mtop-pt-minilm+lr-b2000.json) ·
[re-splits](docs/results/s3/robustness-mtop-pt-minilm+lr-b2000.json) ·
[model and limits](docs/model.md).

The experimental 150M reader also uses Meta models as teachers for its history.
On fixed MTOP splits, **3,000 teacher-labelled history rows and zero human history
labels** measured **87.7% accuracy, 42.5% certified coverage and 99.7% accepted
precision**. Gold calibration labels are still required. Inventing ten messages
per label from label names alone measured **84.95% accuracy and 25.6% coverage**,
about equal to 200 human history labels. These are offline reader results;
Meta models supply history and Ren makes the decisions.
[Teacher protocol and results](docs/results/s2/teacher.md).

On typed-decisions, the fixed five-member ensemble measured **80.25% accuracy**.
The pre-registered label-fit point has Brier **0.098** and ECE **0.019**; a later
teacher-distribution fit has Brier **0.042** and ECE **0.169**, at the same accuracy.
Both operating points remain reported together. The reader can still accept
wrong answers when the correct intent is absent from its options.
[Ensemble protocol](docs/results/s2/typed-ensemble-plan.md) ·
[Missing-intent audit](docs/results/s3r/README.md).

## Sign in and connect Meta

Workspace routes now require a signed-in session and enforce workspace ownership.
Create an account and workspace, import the person's past decisions with Teach, then connect
Meta and select accounts in Connections. Autonomy controls permission per decision type;
Activity contains the background-work switch, global stop, audit trail and observed ad results.
Real sends and ad changes default off. Only sandbox ad accounts are supported.

[Login, exact OAuth settings, synthetic demo and execution switches](docs/integrations/harness.md).
Use `SECND_SESSION_COOKIE_SECURE=false` only for local HTTP development. Existing contract
fixtures require explicit `SECND_DEMO_MODE=true`; a seeded synthetic demo is also available.

## Run one container

The default command starts with the released encoder and workspace classifier:

```sh
docker compose up --build
```

To opt into a reader, install its checkpoint and pinned MiniLM retrieval cache,
then select it in [serving configuration](docs/serving.md). Existing registered
workspaces keep their own calibrated gates unless explicitly selected for a reader.
An explicitly configured reader fails startup if its files or device are unavailable.

Open **http://localhost:8000**. The build downloads and verifies the pinned public
encoder. Runtime can copy it into the persistent data volume and infer offline.
The `secnd-data` volume keeps labels, predictions, models and embedding caches.

Teach imports labelled CSVs, freezes a calibration holdout and fits a first model.
Companion shows automatic decisions, uncertain questions and random spot-checks.
Answers persist before background retraining. Metrics shows local version history,
rollback and `guarded` when automatic answers are paused. Published evaluations
and the recorded teacher handoff remain separate from workspace measurements.
Chat currently remains a preview.

Teach also offers **Start with only label names**: enter 2–128 labels or import
a one-column list. Muse Wonder on the Meta API invents up to ten starting
messages per label, with a **$1 cumulative workspace cap**, then Ren reads the
editable history for its suggestions. Saved examples are shared by label-list
hash; resuming keeps edits and deletions and does not buy completed examples
again. Failed or interrupted requests require review before another paid attempt.

This mode requires the configured reader checkpoint (default 150M; `encoder/` and
`scorer.pt`) under `SECND_DATA_DIR/encoders/<model_id>`, the pinned local MiniLM
retrieval cache, plus `SECND_META_API_KEY`. The released encoder above is a separate
artifact. The default reader and retrieval inference run on CPU with up to eight
nearest history examples.
Generation refuses to start if the reader cannot load. With only invented
history there is no independent workspace calibration, so every suggestion
requires human review. Confirmed answers join the history without a teacher call.
[Cold-start API and budget behavior](docs/api/README.md#start-with-label-names).

## CSV to a live decision

This generates 200 deliberately noisy illustrative rows, not a benchmark:

```sh
python3 - <<'PY' > /tmp/ren-demo.csv
import csv, sys
writer = csv.writer(sys.stdout)
writer.writerow(['item_id', 'text', 'amount', 'category'])
for i in range(200):
    label = 'Travel' if i % 2 else 'Food'
    text = 'flight ticket journey' if i % 2 else 'pizza dinner restaurant'
    tag = chr(97 + i // 26) + chr(97 + i % 26)
    if i % 17 == 0:
        label = 'Food' if label == 'Travel' else 'Travel'
    writer.writerow([f'past-{i}', f'{text} {tag} number {i}', i % 7, label])
PY
API=http://127.0.0.1:8000/api/v1/workspaces/default
curl --fail-with-body -sS "$API/decisions:import" \
  -F 'file=@/tmp/ren-demo.csv;type=text/csv' -F target_column=category
curl --fail-with-body -sS "$API/predict" -H 'Content-Type: application/json' \
  -d '{"items":[{"item_id":"next","text":"Train fare","amount":25}]}'
curl --fail-with-body -sS "$API/queue"
```

A small import can promote a classifier with zero certified coverage. In that
state every item goes to a person. Importing again into the same workspace returns
409 and preserves its labels and holdout; choose a new workspace for another task.

```sh
curl --fail-with-body -sS "$API/answers" -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-answer-1' -d '{"item_id":"next","label":"Travel"}'
curl --fail-with-body -sS "$API/metrics"
curl --fail-with-body -sS "$API/models"
```

[API contract and errors](docs/api/README.md).

## Meta messages and the learning loop

The [WhatsApp Business Cloud API and Messenger integration](docs/integrations/meta.md)
also supports Instagram messages, requires receiving-account workspace mappings,
verifies the GET handshake and raw-body HMAC-SHA256 signatures, deduplicates message
IDs across restarts, then routes supported text through the workspace's Loop.
Uncertain messages enter the same human queue. Replies are dry-run by default;
real WhatsApp sends require both a token and `SECND_WHATSAPP_SEND_ENABLED=true`.
Messenger and Instagram outbound replies remain previews. Sandbox Ads insights
become typed Ren proposals; account changes additionally require current certification,
workspace auto-act and the default-off Ads execution switch. Proposals and attempts
remain in the workspace action log. Compose forwards these settings from
`.env`; no secret enters frontend code or image layers.

The [cold-start protocol and curve](docs/results/s5/README.md#reproduce-the-cold-start-handoff)
uses Meta teacher labels at weight 0.3, followed by simulated human answers at weight
1. Fixed calibration and test rows never train the model. The teacher stays outside
the automatic decision path. The graph counts completed decisions, with pending
questions reported separately. It is an offline simulation, not customer behavior.

To watch a CPU simulation locally after installing dependencies:

```sh
uv run python -m backend.watch --dataset mtop --featurizer pt-minilm
```

The terminal shows coverage, held-out precision against the 99% target, accuracy,
label counts, promoted/refused candidates, reasons, guard state and recent answers.

## Develop and check

Python 3.12, [uv](https://docs.astral.sh/uv/) and Node.js 24:

```sh
uv sync --locked
npm ci
cp .env.example .env
npm run dev
npm run check
```

Development UI: **http://localhost:3000**. API docs: **http://127.0.0.1:8000/docs**.
Vite proxies `/api` to FastAPI. Production serves the built frontend and API from
one process. `uv.lock` and `package-lock.json` pin dependencies.

The app defaults to `SECND_SERVING_BACKEND=head`; reader checkpoints are optional.
Reader choices, workspace overrides and frozen safety policy are documented in
[serving configuration](docs/serving.md).

Configuration stays in the backend. `SECND_DATA_DIR` defaults to `.data` locally;
`SECND_FEATURIZER` defaults to `pt-minilm`, with a logged base-encoder fallback on
fetch/verification failure. Encoder repo/revision overrides must match the committed
manifest. `.env.example` lists messaging settings; teacher calls use Meta provider
settings locally and require an explicit paid flag with a $10 task cap across resumes.

## What the gate means

Temperature calibration and threshold selection use a fixed held-out calibration
set. Automatic coverage requires the one-sided Clopper–Pearson lower bound to pass
the selection level; insufficient evidence means zero coverage. A regression gate
can refuse a candidate. Bad random spot-checks can pause automatic answers through
`guarded`. A new promotion resets the live guard. A target is not a universal
precision guarantee on new traffic.

The service supports one API process. Answers and model artifacts persist;
spot-check counters reset on restart. Rollback restores the saved classifier head
and calibration while retaining current workspace label memory. English only.
Benchmark splits are the project's fixed group splits, not official-score comparisons.

## Data and weights

TOPv2 and MTOP are Meta open assistant datasets; encoder tuning removes normalized
MTOP text overlap before training on TOPv2. See the [training report](docs/results/s2/pretune-pt-minilm.json)
and the published [weight card](https://huggingface.co/Secnd-AI/ren/tree/v1.0.0).
Other evaluations use UK IPSA expenses (OGL v3.0), BANKING77 (CC-BY-4.0) and the
synthetic US bank set (MIT). Sources and fixed split manifests live in
[model documentation](docs/model.md) and [evaluation results](docs/results/).
Keep local data, secrets, model files and caches out of Git.
