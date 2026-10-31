# Secnd

**Your Meta business. With you.**

Sign in → Connect Meta → Pick accounts → Set your bar → Open your decision inbox.
Ren watches ads, WhatsApp, Messenger and Instagram, asks one tap at a time,
and earns permission to act separately for Ads, Inbox and Account.
You choose Watch, Ask or Act and a required bar of 90%, 95%, 98% or 99%.

Ren is our **400M decision model**, trained once and reading a person's past
choices as examples. Muse Spark Open supports the project's language work;
**Muse never decides**. The running model identifier is shown in About and the
footer, with a model-card link when supplied by the API. Deployments may select
other serving backends; the synthetic demo below runs scripted proposals, without
loading the 400M model. [Serving configuration](docs/serving.md).

Built for Meta's platform. Built with Muse Spark Open. Secnd is independently
developed and is not an official Meta product.

## Run the offline product demo

From the repository root, using Python 3.12 and Node.js 24:

```sh
uv sync --locked
npm ci
npm run demo --workspace @secnd/frontend
```

The first two commands install locked dependencies and may need network access if
packages are not cached. The demo command uses installed dependencies, builds the
frontend and starts a local server with `UV_NO_SYNC=true`; it makes no provider or
paid calls. If uv's normal cache is inaccessible, prefix the demo command with
`UV_CACHE_DIR=/tmp/secnd-uv-cache`. No `.env` file or model download is needed.

Open **http://127.0.0.1:8000/?setup=1** for the synthetic sign-in and Meta
walkthrough. Open **http://127.0.0.1:8000/** directly for the decision inbox.
Use `-- --port 8032` after the demo command to select another local port.
Restart the command for a fresh story; each run creates and removes its own
new temporary database. The seeder refuses any directory with existing contents
and any directory that is itself a symbolic link. It never opens an ordinary
business database. [Seeder](backend/demo_seed.py).

**All demo people, business names, messages, scores, reviews, actions and ad
outcomes are synthetic.** The fictional Fold & Field business contains:

- 20 pending decisions: five each from Ads, WhatsApp, Messenger and Instagram.
- Ads with 61 of 61 synthetic confirmations and a 95% chosen bar; three simulated
  acted decisions show that personal-review basis.
- Inbox with 40 confirmations, still short of its 95% bar.
- Account with 61 confirmations followed by a correction; its bound drops below
  95% and its effective level returns from Act to Ask. Account writes are not implemented.
- Three synthetic ad observations comparing spend, cost per result and click-through
  before and seven days after. They are illustrative observations, not causal results.

These counts and scores are **story props, not measured model performance**.
The running model identifier is `synthetic-scripted-demo`. Reviews and undo persist
in the disposable SQLite database; no model trains or infers in this demo.
Meta connection and account selection are simulated. Live sign-in, OAuth,
generation, imports and provider writes are blocked by the demo server.

In the inbox, **j/k** move, **y** confirms, **1–4** choose another action and **u**
undoes the last answer. Why shows nearest saved decisions. Autonomy shows exact
review counts, a conservative lower bound and the additional confirmations needed.
Activity includes the agent log, observations and a stop switch with one inline
confirmation. Stopping requires an operator reset before actions resume.

## Evidence, kept separate from your business

The fixed public-history evaluation for `ic-ettin-400m-s6000-near`, with eight
examples and the support floor, recorded **13.89% coverage**, **609 accepted
answers with zero errors** and a **99.40% lower confidence bound** over **4,386**
unchanged MTOP test rows. Source: `docs/results/s3r/floor-full-test.json`.
Calibration labels were audited by models, not humans. These results require the
full fixed public history and do not certify a new business or future traffic.
The separate missing-intent audit contains a wrong acceptance for this condition.
Source: `docs/results/s3r/README.md`.

Personal reviews are labelled “Measured on your own reviews”. Audited public
history is labelled separately. Saved classifier calibration is described as
calibration history; it is never presented as either of those evidence sources.

The independently released encoder remains available at
[Secnd-AI/ren](https://huggingface.co/Secnd-AI/ren/tree/v1.0.0), pinned by the
[SHA-256 manifest](backend/encoder-ren-1.0.0.json). Encoder, local classifier and
400M serving configurations are distinct artifacts.

[Three-minute video script](docs/submission/video-script.md) ·
[Indie developer submission](docs/submission/submission.md) ·
[Integration and demo limitations](docs/submission/implementation-notes.md).

## Connect a real business

Use `npm run dev` after installing the locked dependencies. Create a local
account and business, connect through Facebook Login for Business, select accounts
and choose a bar. Local HTTP requires `SECND_SESSION_COOKIE_SECURE=false`;
production requires HTTPS and secure cookies. Import past decisions before expecting
suggestions. Public app access, permissions and asset subscriptions require setup.
Real sends and ad changes default off; Ads supports sandbox accounts only.
[Full integration instructions](docs/integrations/harness.md).

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

Past decisions imports labelled CSVs, freezes a calibration holdout and fits a first
model. The inbox shows automatic decisions, uncertain questions and random spot-checks.
Answers persist before background retraining. Automatic answers pause through
`guarded`, and the About screen names the running model. Published evaluations remain
separate from workspace measurements.

Starting from label names alone is available through the API: Muse Wonder invents up
to ten starting messages per label under a **$1 cumulative workspace cap**, then Ren
reads the editable history for its suggestions. It needs the configured reader
checkpoint, the pinned MiniLM retrieval cache and `SECND_META_API_KEY`. With only
invented history there is no independent workspace calibration, so every suggestion
requires human review. [Cold-start API and budget behavior](docs/api/README.md#start-with-label-names).

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
ROOT=http://127.0.0.1:8000/api/v1
API=$ROOT/workspaces/ren-demo
curl --fail-with-body -sS -c /tmp/ren-demo-session "$ROOT/auth/register" -H 'Content-Type: application/json' \
  -d '{"username":"ren-demo","password":"choose-a-long-password"}'
curl --fail-with-body -sS -b /tmp/ren-demo-session "$ROOT/workspaces" -H 'Content-Type: application/json' \
  -d '{"workspace_id":"ren-demo"}'
curl --fail-with-body -sS -b /tmp/ren-demo-session "$API/decisions:import" \
  -F 'file=@/tmp/ren-demo.csv;type=text/csv' -F target_column=category
curl --fail-with-body -sS -b /tmp/ren-demo-session "$API/predict" -H 'Content-Type: application/json' \
  -d '{"items":[{"item_id":"next","text":"Train fare","amount":25}]}'
curl --fail-with-body -sS -b /tmp/ren-demo-session "$API/queue"
```

A small import can promote a classifier with zero certified coverage. In that
state every item goes to a person. Importing again into the same workspace returns
409 and preserves its labels and holdout; choose a new workspace for another task.

```sh
curl --fail-with-body -sS -b /tmp/ren-demo-session "$API/answers" -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-answer-1' -d '{"item_id":"next","label":"Travel"}'
curl --fail-with-body -sS -b /tmp/ren-demo-session "$API/metrics"
curl --fail-with-body -sS -b /tmp/ren-demo-session "$API/models"
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

## What the evidence means

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
