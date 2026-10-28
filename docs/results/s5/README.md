# Language-model integration

The decision classifier does not call a language model. Optional labeling,
explanations and chat use the configured Chat Completions endpoint.

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
request against remaining funds. The shared ledger caps total spend at $30.

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

Any probabilities requested in generated JSON are self-reported estimates,
not token likelihoods or calibrated classifier probabilities. Baseline NLL and
ECE remain unmeasured unless a suitable probability source becomes available.
No baseline accuracy, speed or coverage result is available here yet.
