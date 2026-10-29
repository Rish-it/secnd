---
language:
  - en
pipeline_tag: text-classification
base_model: sentence-transformers/all-MiniLM-L6-v2
tags:
  - selective-classification
  - calibration
  - secnd
---

# Ren — secnd-ren-1 (DRAFT)

**Release status: RELEASE_TODO. No downloadable artifact is claimed by this draft.**
Version: **RELEASE_TODO (`ren-1.<n>`)**. Release license: **RELEASE_TODO**.

Ren learns a workspace's recurring decisions using a sentence encoder, numeric/date
features, merchant-memory features and a logistic-regression classifier head.
A temperature and automatic-answer threshold are fitted on held-out calibration
rows. Decisions below the threshold, and sampled automatic spot-checks, go to a
person. Muse is used for teaching/explanation work, not the automatic classifier.

## Intended use

English text classification over the ordered labels supplied with this release.
The demo's headline benchmark is Meta's MTOP English assistant intents. Expense
categorisation uses IPSA and can honestly certify no automatic answers. This is
not a general chat model. A new workspace needs its own labels, calibration and
release checks; benchmark precision is not a promise for arbitrary future inputs.

## Weights and runtime

The reviewed bundle contains `model.pkl` (trusted-source classifier/calibration
artifact), `card.json` (the matching EvalCard), `runtime.json` (ordered labels,
featurizer, version and merchant-memory snapshot) and this reviewed `README.md`.
The upload helper copies none of the source datasets or training caches and
never deserialises the pickle. Load pickles only from a trusted release.

The current reference encoder is
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, under Apache 2.0. It is
loaded separately on CPU; this head bundle does not include a fine-tuned encoder.
Update this section and packaging if the final chosen encoder differs.

Before release, verify runtime labels, merchant memory, encoder revision and
preprocessing match the exact saved classifier. The current registry's small
candidate artifact is not by itself a complete portable inference bundle.
Record the source commit, dependency lockfile, artifact hashes, dataset/split
hashes and seed. **RELEASE_TODO: provide a verified load/predict example and its
round-trip output from the final bundle.**

## Evaluation reference, not an assertion about unpublished weights

The published MTOP reference at 2,000 labels and seed 0 answers 31.68% of the fixed
test set automatically at 99.62% precision. The claim holds in 99.05% of 2,000
re-splits. [Measured card](https://github.com/Rish-it/secnd/blob/secnd-bootstrap/docs/results/s3/card-mtop-minilm%2Blr-b2000.json).
The current BANKING77 fixed test is below 99%; IPSA has zero automatic coverage
at that budget. Do not substitute one dataset's result for another's.

**RELEASE_TODO:** replace this reference with results tied to the exact artifact
being uploaded, and attach the full card. Include calibration lower bound,
threshold, coverage/precision, re-split pass rate, calibration error, latency,
artifact size and provenance. Missing measurements stay null.

## Training, data and limitations

Seed/stream labels train the model; calibration and test rows never do. Human
answers/spot-checks have weight 1; Muse cold-start labels have weight 0.3. Any
paraphrases must join only the selected training texts for that split.

MTOP comes from [Meta's published archive](https://dl.fbaipublicfiles.com/mtop/mtop.zip)
(CC BY-SA 4.0, `mtop/LICENSE.txt`); see
[Li et al., MTOP](https://arxiv.org/abs/2008.09335). BANKING77 is CC BY 4.0;
IPSA is OGL v3. Retain the notices applicable to the released artifact/data.
The release license has not been chosen in this draft.

Coverage can remain zero. Fixed-test precision can miss the target despite a
high re-split pass rate. Label noise, unseen categories and future distribution
changes need human review and monitoring. The 95% bound is statistical evidence,
not an unconditional guarantee.

## Release procedure

Follow the [release preparation notes](../results/final/model-release.md).
The [Hub model-card guide](https://huggingface.co/docs/huggingface_hub/guides/model-cards)
defines the README metadata. Review all RELEASE_TODO fields, set the release
license and remove DRAFT only after the model/bundle checks are complete.
