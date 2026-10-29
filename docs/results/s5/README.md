# Language-model integration

The decision classifier does not call a language model. Optional labeling,
explanations and chat use the configured Chat Completions endpoint.

## Measured comparison

Same 500 held-out IPSA claims, seed 0 and 62 allowed cost types. Raw
[comparison](comparison.json), [Muse result](muse-baseline.json),
[sample](muse-sample.json) and [trials](muse-predictions.jsonl):

| Model | Training labels | Accuracy | ECE-10 | p50 per item | Cost per 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Muse Spark, zero-shot | 0 | 37.0% | null | 1,577.04 ms | $2.3451 |
| Character TF-IDF + logistic regression | 200 | 36.2% | 0.1440 | 0.235 ms | unmeasured |
| Character TF-IDF + logistic regression | 5,000 | 45.2% | 0.2078 | 0.231 ms | unmeasured |

All 500 final outputs passed validation. Muse's cost includes conservative
retry estimates and all compact-protocol paid calls, including calls completed
while a failed batch was shutting down. It excludes the earlier distribution
experiments. The complete ledger, including those experiments and the
connection check, totals **$2.39298**. These are full-list token estimates;
cached-input discounts are excluded.

Muse NLL/ECE are null because confidence is self-reported. The text classifiers
are raw baselines; this table makes no calibrated precision or automation
coverage claim. CPU hosting cost and the fitted model comparison await measured
throughput. Latencies are local CPU prediction times versus remote HTTP times,
not equivalent hardware or service-level guarantees.

## Connection

Copy `.env.example` to `.env` and set `SECND_MUSE_API_KEY` locally. The default
endpoint is `https://inference.baseten.co/v1`; the configured model is
`meta-models/muse-spark-open-1026`. Check access with one small paid request:

```sh
uv run python -m backend.muse --smoke
```

The command outputs only HTTP status, the returned model ID and token counts.
It does not output the key, response text or reasoning text.

## Spending limits

`SECND_MUSE_BUDGET_USD` limits a run to $10 by default. Before starting, callers
provide a row count and measured input/output tokens per row; an estimate over
the limit is rejected before any HTTP request. The client also checks each
request against remaining funds, including reservations for concurrent calls.
The shared ledger caps total spend at $30.

Each run appends its token totals and full-list cost to [spend.csv](spend.csv),
including partial runs that fail. An exclusive file lock serializes paid runs
using that ledger on one host. Keep the same ledger across invocations. A lost
response is charged conservatively using the request's reserved token estimate
and marked `estimated`; actual provider billing may be lower. Restarting with
a different ledger does not carry forward earlier spend.

The [connection result](smoke.json) records one request. Its cost is computed
from the reported input and output tokens. Reasoning is a subset of output
tokens and is counted separately without adding it to the bill a second time.

## Provider limits

Checked October 29, 2026:

- [Chat Completions](https://docs.baseten.co/reference/inference-api/chat-completions)
  uses bearer authentication and returns usage metadata.
- [Feature support](https://docs.baseten.co/inference/model-apis/overview)
  states that Muse ignores `logprobs` and returns no log probabilities.
- [Structured outputs](https://docs.baseten.co/inference/structured-outputs)
  states that Muse accepts `json_schema` but does not enforce it. Responses must
  be parsed and validated in the client.
- [Reasoning](https://docs.baseten.co/inference/model-apis/reasoning) remains
  enabled; `low` reduces the requested effort. Reasoning tokens are included in
  completion tokens and must not be charged twice.
- [List pricing](https://www.baseten.co/library/muse-spark-open/) is $1.25 per
  million input tokens, $0.15 cached input and $4.25 output. Usage estimates use
  the full input price conservatively, without assuming a cache discount.

Generated confidence is a self-reported estimate, not a token likelihood or a
calibrated classifier probability. Baseline NLL and ECE stay null.

## Client calls

`backend.muse.MuseClient` is a synchronous context manager. Pass `Settings`, an
active `BudgetRun`, and optionally an HTTP transport for tests. The existing
HTTP and validation dependencies are sufficient; no additional SDK is needed.

| Method | Input | Result |
| --- | --- | --- |
| `zero_shot(items, labels)` | `MuseItem[]`, distinct allowed labels | `LabelResult[]` in input order |
| `cold_start(items, labels)` | Same | Same, including label and confidence |
| `explain(item, top3, neighbours)` | One item, up to three probabilities and three `LabelledItem` examples | Short explanation text |
| `chat(question, tools)` | Question and workspace-scoped read-only callbacks | `ChatReply` with answer and tool names used |

`MuseItem` contains `item_id`, `text`, optional `amount` and optional ISO `date`.
Empty descriptions are supported when the source only has numeric/date fields.
Label results include the selected label, confidence,
`confidence_source: self_reported`, successful-response usage and elapsed
milliseconds. Out-of-set labels, non-finite or out-of-range confidence, and
partial responses are rejected. Costs for retries
are included in the run ledger; per-result usage describes the successful call.

Each request has a five-second connection timeout and a 30-second HTTP timeout.
Transient network failures and HTTP 429/500/502/503/504 are retried with backoff,
at most three attempts. The budget is checked before every attempt. Unknown
usage from failed attempts is charged conservatively in the ledger. The smoke
command makes only one attempt.

Chat accepts only `count`, `mistakes` and `stats` callbacks. Each receives one
validated dictionary. `count` takes a `filter` with optional `label`, `auto`,
`model_version`, `start_date` and `end_date`. `mistakes` and `stats` take a
`period`: `today`, `week`, `month` or `all`. Callbacks must already be scoped to
the current workspace. No SQL or arbitrary function names are accepted.
Chat stops after four completion rounds; tool replies are capped at 32 KiB.

Explanations receive the classifier's original top-three scores and up to three
past labeled decisions. Scores keep their original mass; they are not
renormalized over the displayed classes. The prompt requires one or two short
sentences, the highest score as a percentage, uncertainty, and no invented
evidence. A snapshot checks the complete prompt with three classes and three
examples. Retrieval of real neighbours is still awaiting a fitted model.

Before using the client, measure tokens on a representative small sample. Then
provide those measurements and the planned row count to `BudgetRun`.

## Fixed-test comparison

```sh
uv run python -m backend.muse_baseline --limit 5
uv run python -m backend.muse_baseline
```

The first command checks a small sample; the second resumes the same 500 IPSA
test rows, sampled with seed 0. The model sees only the description, amount,
date and 62 allowed labels. No examples or held-out answers enter its prompt.
At most eight requests are in flight, with new calls spaced one second apart;
latency includes HTTP time and retries, excluding this admission wait. The first
84 compact trials preceded pacing; it was added after HTTP 429 responses.
Each completed trial is checkpointed. Invalid outputs count as incorrect and
are never retried for a better answer. Transport and budget failures stop the
run; completed checkpoints are reused on restart. Sample, input, protocol,
source-data and split hashes prevent mixing experiments.

The same rows are evaluated with the reference text classifier at 200 and
5,000 training labels, using only seed/stream rows. Its latency is measured one
item at a time, including text transformation. Its hosted cost is unmeasured.
The calibrated model comparison and learning-loop weighting need their fitted
artifacts before they can be measured.

Exploratory distribution-output trials are preserved separately. Their spend
counts against the experiment's $10 ceiling across restarts. Final inference
costs use the compact protocol's ledger entries; total experiment spend also
includes the exploratory calls. Cached input is charged at the full list
price throughout, so these are conservative token costs rather than invoices.

## Provider routing

`backend.providers.MuseProviders` tries Meta Model API (`muse-spark-1.3`),
Baseten (`meta-models/muse-spark-open-1026`), then Nebius Token Factory
(`meta-models/muse-spark-open`). Only timeouts, HTTP 429 and HTTP 5xx allow
failover after three bounded attempts. A valid answer, invalid model output,
authentication error or exhausted budget never triggers another provider.
Missing keys skip an unconfigured provider. These calls remain off the decision path.

Prices checked **2026-10-29**: all three providers list $1.25 per million input,
$0.15 per million cached input and $4.25 per million output tokens.
Sources: [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits),
[Baseten model page](https://www.baseten.co/library/muse-spark-open/),
[Nebius catalog](https://tokenfactory.nebius.com/model-catalog.md).
The guard conservatively charges cached input at the full input price; reasoning
is included in output. Each provider has its own run budget and locked ledger:
`spend-meta.csv` ($30 total), `spend.csv` (Baseten, $30 total), and
`spend-nebius.csv` ($25 total). Keys stay in backend `.env` settings.

For the fresh version comparison, run
`uv run python -m backend.provider_comparison --provider meta` and the same
command with `--provider baseten`. Each fixes the original 500 IPSA row IDs,
input, split and classification prompt; paid predictions never fail over between
models. Checkpoints and summaries are saved under `providers/<provider>/` and
charged to that provider's shared ledger. Each experiment is capped at $4 across
restarts, so the pair remains below $10. Invalid outputs count as incorrect.
