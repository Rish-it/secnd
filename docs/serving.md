# Ren serving configuration

Ren is an independent decision model built for Meta's platform. The app defaults to
`ic-ettin-150m-s6000-near` on CPU with eight nearest history examples. The optional
`ic-ettin-400m-s6000-near` reader uses an explicitly configured CUDA device. There
is no quantization or automatic model/device fallback.

## Select a reader

Settings reads these values from `.env` or the process environment:

```dotenv
SECND_SERVING_BACKEND=reader
SECND_READER='{"model_id":"ic-ettin-150m-s6000-near","device":"cpu","examples":8}'
SECND_WORKSPACE_BACKENDS='{}'
SECND_WORKSPACE_READERS='{}'
```

For the GPU option, replace `SECND_READER` with:

```dotenv
SECND_READER='{"model_id":"ic-ettin-400m-s6000-near","device":"cuda","examples":8}'
```

A workspace can select its backend and override the reader independently:

```dotenv
SECND_WORKSPACE_BACKENDS='{"legacy":"head","messages":"reader"}'
SECND_WORKSPACE_READERS='{"messages":{"model_id":"ic-ettin-150m-s6000-near","device":"cpu","examples":8}}'
```

`head` selects the existing workspace encoder and calibrated classifier. Set
`SECND_SERVING_BACKEND=head` to select it globally. Configuration changes require
an API restart. An explicit reader `revision` must match the immutable checkpoint
fingerprint in [the manifest](../backend/reader-manifest.json); omitting it selects
that pinned fingerprint. Revisions cover all checkpoint files, including scorer,
encoder and tokenizer. Startup verifies every file hash before loading.

Install the chosen checkpoint under
`SECND_DATA_DIR/encoders/<model_id>/`, including `encoder/` and `scorer.pt`, and
preinstall the pinned MiniLM retrieval model in the local Hugging Face cache.
Checkpoints and caches stay outside Git. Reader loading is local only. Missing
files, a mismatched revision, missing retrieval files or an unavailable configured
CUDA/BF16 device stop startup with a clear error. A configured reader never becomes
the head or a different reader when loading fails.

The production image packages the frozen policy files, but does not redistribute
the experimental reader weights or the MiniLM retrieval cache. Supply those through
the data volume/cache mounts before using the default reader. The existing released
encoder is packaged for the explicitly selected head mode:

```sh
SECND_SERVING_BACKEND=head docker compose up --build
```

## Automatic answers and missing intents

Previously, `CPUReader.predict()` was called only by label-only onboarding. It
returned raw suggestions with `auto=False` and `threshold=None`: invented examples
provided no independent workspace calibration. There was no automatic reader
answer path. CSV workspaces used their calibrated classifier instead.

The API now dispatches reader workspaces through the configured reader. Each
prediction loads its model/example-count operating point from the committed
[150M results](results/s3r/attacks-floor.json) or
[400M results](results/s3r/attacks-floor-400m.json). The values are never fitted or
adjusted while serving. Temperature scales the probabilities before comparison
with the frozen confidence threshold.

Support is the **maximum MiniLM cosine similarity between the message and any
history row with the predicted intent**, including rows outside the retrieved
context. An automatic answer requires both confidence at or above its threshold
and support at or above its floor. Missing temperature, threshold or floor, no
history for the predicted intent, too few context examples, or capped input means
human confirmation. In particular, the 400M operating points with zero and 32
examples have null thresholds and floors and cannot answer automatically.

These values were measured with **full MTOP history only**. They do not certify a
smaller history, invented examples, a new label set or a different workspace. The
app therefore checks the complete audited history fingerprint and label set before
enabling reader automatic answers. An edit or new answer changes that history and
requires human confirmation again. The fingerprint was derived from the same
fixed split and checked against the split hash in the committed audit. Calibration
rows never enter reader retrieval history.

[Amendment 2](https://github.com/Rish-it/secnd/blob/6f192e0/docs/results/s3r/protocol.md#amendment-2-support-floor-for-missing-intents-registered-before-its-inference)
defines the floor. The added audit files are unchanged copies from commit
`6f192e0`. The support audit's observed precision on its small sample is separate
from the earlier confidence-only certification. The app displays both with their
scope; neither is a universal guarantee for future traffic or an independent CPU
precision certification.

Confirmations/corrections persist through the existing answers API. Ordinary
workspace answers continue through the existing background training mechanism and
join subsequent reader history. Label-only answers join its editable history.
Neither path asks a teacher for a decision. Outbound sends and account changes keep
their existing explicit switches.

## Inspect and verify

`GET /api/v1/workspaces/{workspace}/serving` exposes the selected model, immutable
revision, device, example count, frozen policy, scope status and separate published
evaluation. Metrics shows this information at desktop and mobile widths. Local
workspace cards retain null unmeasured values instead of borrowing benchmark
accuracy or coverage.

```sh
uv sync --locked
npm ci
npm run check
npm run test:reader
```

The browser fixture uses local stand-ins and marks itself as a fixture. Reader unit
and API tests replace model loading, retaining the actual policy, temperature,
retrieval, support and routing code. GPU configuration is tested without allocating
a GPU. CI uses mocked reader loading and explicitly selects `head` for its container
smoke test; no reader weights or live provider access are required.

[Running CPU HTTP measurements and screenshots](results/c3-serving/README.md).
