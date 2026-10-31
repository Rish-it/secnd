# Ren serving configuration

Ren is an independent decision model built for Meta's platform. Code and Compose
default to `head`, the existing encoder and calibrated workspace classifier. A
clean clone starts without experimental reader weights. A workspace with its own
registered model keeps that model's calibrated gate unless its configuration
explicitly selects a reader.

## Select a reader

Settings reads these values from `.env` or the process environment:

```dotenv
SECND_SERVING_BACKEND=head
SECND_READER='{"model_id":"ic-ettin-150m-s6000-near","device":"cpu","examples":8}'
SECND_WORKSPACE_BACKENDS='{}'
SECND_WORKSPACE_READERS='{}'
```

`SECND_WORKSPACE_BACKENDS` is the explicit per-workspace selection. An entry in
`SECND_WORKSPACE_READERS` also selects a reader, unless the backend map explicitly
says `head`. For a demo workspace:

```dotenv
SECND_WORKSPACE_BACKENDS='{"demo":"reader"}'
SECND_WORKSPACE_READERS='{"demo":{"model_id":"ic-ettin-150m-s6000-near","device":"cpu","examples":8}}'
```

The optional GPU selection is:

```dotenv
SECND_WORKSPACE_READERS='{"demo":{"model_id":"ic-ettin-400m-s6000-near","device":"cuda","examples":8}}'
```

`SECND_SERVING_BACKEND=auto` uses the reader until a workspace has both a registered
head and 1,000 history rows, then the head. At 1,000 labels the reader leads on MTOP and the
head on BANKING77; at full history the head leads both
([measurements](results/s2/wide-round.md#how-much-history-a-person-needs)). A per-workspace choice still wins.

A global `SECND_SERVING_BACKEND=reader` supplies a reader for workspaces without a
registered model. It preserves existing registered workspace models. An explicit
per-workspace reader selection can override a registered head. Label-only
onboarding remains an explicit reader flow, with human-reviewed suggestions.
Configuration changes require an API restart.

An explicit reader `revision` must match the checkpoint fingerprint in
[the manifest](../backend/reader-manifest.json); omitting it selects that pinned
fingerprint. Startup verifies encoder, scorer and tokenizer hashes. Install the
chosen checkpoint under `SECND_DATA_DIR/encoders/<model_id>/`, including `encoder/`
and `scorer.pt`, and preinstall the pinned MiniLM retrieval model in the local
Hugging Face cache. Checkpoints and caches stay outside Git.

The image packages the frozen policies, but does not redistribute experimental
reader weights or the MiniLM retrieval cache. Supply those through data/cache
mounts before opting into a reader. Explicit reader configuration eagerly loads
its model: missing files, a mismatched revision, missing retrieval files or an
unavailable configured CUDA/BF16 device stop startup. There is no quantization or
automatic model/device fallback. Default head startup needs none of these optional
reader files:

```sh
docker compose up --build
```

## Frozen policy and benchmark

Originally, `CPUReader.predict()` only served label-only onboarding, returning raw
suggestions with `auto=False, threshold=None` because invented examples provided
no independent calibration. Automatic reader answers were not wired.

Explicit reader serving now loads model/example-count policies from unchanged
copies of commit `ada0a0d`:

- [150M audited policy](results/s3r/attacks-floor-audited.json)
- [400M audited policy](results/s3r/attacks-floor-audited-400m.json)
- [Full test with support floor](results/s3r/floor-full-test.json)

**Calibration labels were changed by a model audit, not a human audit.** Twelve
MTOP calibration labels changed; history and test labels stayed unchanged. These
artifacts supersede the earlier serving policies. No temperature, threshold or
floor is fitted or adjusted during serving. All audited 400M example counts now
have complete policies; missing/null values in any future policy still fail closed.

Temperature scales probabilities before the frozen threshold comparison. Support
is the maximum MiniLM cosine similarity between the message and **every history
row carrying the predicted intent**, including rows outside the retrieved context.
Automatic eligibility requires both confidence and support to clear their frozen
cutoffs, the full audited history, enough context examples and an uncapped input.
No history for the predicted intent means no automatic answer.

The app's floored benchmark comes from all 4,386 unchanged MTOP test labels in
`floor-full-test.json`, rather than the 500-row attack sample or confidence-only
coverage. At eight examples, the 150M floored share is **10.85%**, observed precision
**100%**, with recorded 95% lower bound **99.23%**; the 400M values are **13.89%**,
**100%** and **99.40%**. These are the recorded GPU-derived evaluation conditions,
not independent CPU certification or measurements of an arbitrary workspace.
Confidence-only coverage figures do not describe the floored policy.

The [released reader](https://huggingface.co/Secnd-AI/ren/tree/v1.1.0),
`ic-ettin-400m-s6000-near-wide-fine-seed1`, has [its own gate](results/s3r/README.md#released-reader-amendment-6):
**21.04%**, 923 accepted with 1 error, lower bound **99.40%** at eight examples. It accepts nothing with no examples.
Install it with `hf download Secnd-AI/ren --revision v1.1.0 --include "reader/*"`, then move `reader/` to
`SECND_DATA_DIR/encoders/ic-ettin-400m-s6000-near-wide-fine-seed1/`.

Limits stay visible in the source results: the audited 150M eight-example policy
accepted two wrong missing-intent answers out of 500, exceeding the one-error target.
The 400M 32-example full-test precision lower bound is 98.86%. Frozen values are
served unchanged and these misses remain reported.

## Live checks and demo history

Reader automatic eligibility uses the existing `SPOT_CHECK_RATE`. Selected checks
have `auto=False, spot_check=True` and enter the human queue. Their answers persist
with the spot-check source and update the reader guard using the existing
one-sided 95% error-bound rule against the 99% target. The guard pauses subsequent
reader automatic answers. Counts are scoped to the workspace and reader policy,
retained across history changes, and reset on process restart like the head guard.
Replaying an answer cannot count the same check twice.

The frozen values cover **full MTOP history only**. They do not certify a smaller,
invented or different workspace history. The complete audited history fingerprint
and label set must match before reader automatic answers are enabled. Calibration
rows never enter retrieval history.

A person's answer changes that exact history and pauses reader automatic answers,
even when it confirms the suggestion. Restarting the same persisted workspace does
not restore its original history. For the demo, show automatic answers before
confirming, or use a disposable copy. The local HTTP rehearsal below creates a new
temporary workspace on every run; stopping and rerunning it restores a fresh demo
without editing a real database.

Ordinary workspace answers retain the existing background training path and join
reader history when that workspace explicitly uses a reader. Label-only answers
join its editable history. No teacher participates in a decision. Outbound sends
and account changes retain their explicit switches.

## Inspect and verify

`GET /api/v1/workspaces/{workspace}/serving` exposes the active backend, reader
identity, frozen policy, scope/guard status and published evidence, including
the full-test floored benchmark with its model-audit qualification. About shows the
running model. Workspace measurement fields stay separate.

```sh
uv sync --locked
npm ci
npm run check
```

Tests cover registered-head preservation, explicit reader selection, default
startup, frozen policy loading, support gating, spot-check routing and guard
feedback. Reader model loading uses stand-ins in CI.
Docker CI runs the default command without a backend override. No checkpoint or
GPU is needed for these tests.

[CPU HTTP rehearsal and screenshots](results/c3-serving/README.md).
