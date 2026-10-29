# Ren: a decision model built for Meta apps

## Current 150M reader: clean split and corrected certification

Ren's experimental reader is `ic-ettin-150m-s6000-near`, distinct from the released
encoder plus linear head documented below. #27 measured encoder tuning, #28
introduced the in-context reader, and #29 separated temperature fitting,
certification and test scoring. The earlier **25% / 65.7% certified figures are
withdrawn**: those evaluations do not support the precision claim.

The follow-up at commit `e2581ee` carves 4,000 calibration rows with seed 0 from
MTOP's official training split, leaving 11,667 history rows and the fixed 4,386
official test rows. Reader training already excluded MTOP training texts.
The calibration halves contain 2,000 rows each: one fits temperature and the
other certifies the unchanged rule-A gate. Test labels choose neither.

This supersedes #29's calibration from official eval: 1,032 texts overlapped
reader training, leaving only 1,203 clean rows and 602 certification rows.
That earlier clean card certified zero coverage. Changing the calibration source
increases evidence; it is not a gate-rule change. The new measured table is:

| History | Certified test coverage | Precision of accepted answers |
| --- | ---: | ---: |
| none | 23.32% | 100.00% |
| 200 human labels | 28.57% | 99.76% |
| 1,000 human labels | 28.27% | 99.84% |
| all human labels (11,667) | 36.89% | 99.94% |
| 200 teacher-labelled rows | 30.19% | 99.77% |

History conditions use eight nearest examples; “none” is zero-shot. Source:
[150M clean card](results/s2/incontext-ic-ettin-150m-s6000-near-clean.json).
The [200 teacher-labelled history result](results/s2/muse-teacher-mtop.json)
measured 30.19% coverage at 99.77% accepted precision. BANKING77 and the logistic
head still certify zero coverage. These are measured point estimates on the fixed
split, not guarantees under distribution shift. The split change follows earlier
experiments; an untouched external evaluation would provide stronger evidence.

[S3R's registered audit](results/s3r/README.md) tested 500 paired MTOP messages.
When the correct intent was removed from the options, the unchanged gate accepted
4, 9 and 9 wrong answers at 0, 8 and 32 examples: **0% accepted precision** in
that condition. Other registered attacks measured at least 99% on this sample,
with confidence intervals too wide to establish a 99% lower bound. This supports
an explicit option-set and distribution limit on the claim.

On a four-CPU Modal allocation, the 150M reader measured p50/p95 of **325/347 ms**
with no examples, **401/451 ms** with eight, and **610/677 ms** with 32; process
peak RSS was **2.42 GiB**, including the retrieval index and initialization.
All pass the requested one-second reader p95 budget. Timings cover single-request
CPU FP32 tokenization through softmax, excluding retrieval and API overhead;
[raw measurements and the 32M comparison](results/s3r/README.md#cpu-serving-single-requests-113-options)
record scope and limits.

## Why

Ren is an independently released decision model built for Meta's platform. It
starts with an encoder tuned on Meta's open assistant data, then learns a
workspace's own labels and corrections. The calibrated classifier decides when
to answer and when to ask a person. Optional Meta teacher models supply cold-start
labels; they stay outside the automatic decision path.

## Where

Secnd learns one organisation's past decisions and repeats them only where it can prove it is right.
The first decision it learns is which cost type an expense claim belongs to.

**Main dataset: UK MP business costs.** Every claim the Independent Parliamentary Standards Authority
published for 2022–23 to 2024–25: 283,682 claims, 62 cost types, with the claim date, the amount, the
free-text details and the MP office that filed it. Real decisions by a real organisation, under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Classes are imbalanced: the largest has 55,690 claims, the smallest a handful.

**Three more sets** check that results are not specific to one source:

- 68,000 synthetic US bank-statement descriptions in 17 spending categories (MIT).
- BANKING77: 13,083 customer-support questions in 77 intents (CC-BY-4.0).
- MTOP (English): 22,288 assistant requests in 113 intents, published by Meta (CC BY-SA 4.0).

`uv run python -m backend.data` downloads them, checks each file's SHA-256 and writes the split
manifests.

**How the data is split** ([manifests](results/s1/)):

- **Test** is the newest 15% of claims (2 Jan 2025 to 7 May 2026). Nothing after the training period is
  ever trained on.
- **Calibration** is the 15% of earlier claims just before test, so the confidence threshold is set on
  data that looks like what comes next.
- **Seed** is 200 labelled claims from the oldest data, the starting point for learning.
- Undated sets split by payee instead, so no payee appears in more than one part.

**What leaks and what doesn't** ([audit](results/s1/leakage-ipsa.json)):

- The claim's "short description" is picked from a list tied to the cost type and predicts it 97.8% of
  the time. It is never used as an input, and neither is the parent category.
- 75% of test claims come from a payee seen in training, but remembering a payee's last cost type is
  right only 64% of the time on those claims: the same payee is filed under different cost types.
  Every result is reported separately for payees seen before and payees never seen.

**Starting point** ([eval cards](results/s1/)): with 200 labelled claims, a character n-gram text
classifier gets 37.5% of test claims right (88.6% on known payees, 21.7% on new ones). With 5,000 it
gets 44.7%. None of the baselines can yet answer a useful share of claims at 99% precision.

The [card format](results/s1/README.md) follows the API response schema. Each card records both
the pinned source-data hash and the split hash; unmeasured values stay null.

## How

**The model** is a sentence encoder followed by one logistic-regression softmax over the
organisation's own labels, so probabilities over every class sum to 1 and there is no limit on the
number of labels. A payee's latest known label enters as an input the classifier learns to weigh,
not as an override. Only the linear head retrains when a new label arrives; the encoder stays fixed
between batch re-tunes.

### Meta-tuned encoder: release ren-1.0.0

The [open encoder weights](https://huggingface.co/Secnd-AI/ren/tree/v1.0.0)
are release **ren-1.0.0**, tag **v1.0.0**, immutable revision
`5560f25414195f1d1224a6ae7efd19c6b7646d3f`. The
[committed manifest](../backend/encoder-ren-1.0.0.json) pins every downloaded
file by SHA-256. Workspaces use `pt-minilm` by default, serve on CPU, and log a
fallback to the base MiniLM encoder when fetching or verifying the release fails.
The image includes the verified release for use with an offline data volume.

This release fine-tuned `all-MiniLM-L6-v2` with supervised contrastive learning
on **157,405 TOPv2 requests across 69 intents**, for **3,000 steps**.
**23,137 requests** whose normalized text overlaps MTOP were removed before
training ([training report](results/s2/pretune-pt-minilm.json)). TOPv2 supplies
the encoder's prior; a workspace's labels still train its own linear head.
This release does not claim training on every Meta dataset or language.

On the project's fixed English MTOP group split, with 2,000 workspace labels
and seed 0, the tuned encoder plus calibrated head measured **42.42% automatic
coverage at 99.93% test precision**, **88.09% accuracy**, ECE **0.0156**, and a
**99.37% calibration lower bound** ([card](results/s3/card-mtop-pt-minilm+lr-b2000.json)).
The corresponding 2,000 re-splits passed **99.45%** of the time
([robustness report](results/s3/robustness-mtop-pt-minilm+lr-b2000.json)). This is
the project's split, not an official MTOP test-score comparison. These results
are distinct from the base-encoder results below.

### Learning from answers

A workspace keeps a fixed calibration set, a sequence of training labels and a
current classifier version. Seed labels, human answers and random spot-check
answers have weight **1**. Optional teacher cold-start labels have weight **0.3**;
self-reported teacher confidence is not a calibrated probability and does not
change that fixed weight. Calibration and test items never join training.

Each human answer enters the durable label history before a background worker
calls `Loop.on_answer`. The head refits on cached frozen-encoder features,
recalibrates on the same holdout and goes through the release gate. Refusals
keep the current version. An all-human version can be promoted with a null
threshold; promotion alone does not certify any automatic traffic. The gate also
refuses a loss of more than one percentage point of certified calibration
coverage against the current unguarded model.

The loop asks a person to audit a random **5%** of otherwise automatic decisions.
When the one-sided **95%** lower bound on the spot-check error rate exceeds **1%**,
`guarded` pauses all automatic answers until a new version is promoted. Workspace
versions such as `ren-1.0` identify the local head and calibration; the pinned
encoder release `ren-1.0.0` is a separate artifact.

The [five committed MTOP learning curves](results/s4/) for `pt-minilm`, random
queries and 20 rounds end at mean automatic coverage **33.9%**, ranging from
**0.0% to 61.0%** across seeds. Spot-check counts vary, so final label counts vary.
These are offline simulations using the stream's true labels as human answers,
not observed customer behavior. They use fixed seed/calibration/test partitions;
coverage need not rise on every round. The curves do not include teacher
cold-start labels. Runtime spot-check state currently lives in memory and resets
on restart; the durable answer history survives.

**How it was chosen** ([bake-off](results/s2/)): 4 text featurizers (hashed character n-grams,
potion-base-8M static embeddings, all-MiniLM-L6-v2, and n-grams stacked with potion) times 3
classifiers (logistic regression, linear SGD, k-nearest neighbours), at 50, 100, 200, 500 and all
labels, 5 seeds each, scored on the fixed test split. The decision rule was written before the
table was read: keep models within 1 accuracy point and 0.05 nats NLL of the best at 100 and 500
labels, then take the lightest.

| Labels | IPSA accuracy | BANKING77 accuracy |
| --- | ---: | ---: |
| 100 | 34.1% ± 2.8 | 44.3% ± 3.5 |
| 500 | 44.6% ± 1.1 | 78.2% ± 2.0 |
| all | 66.3% (1 seed) | 91.2% (1 seed) |

MiniLM with logistic regression was the only model inside the rule at both budgets on IPSA, the
dataset the rule is judged on. On BANKING77 it is best from 200 labels up; at 50 and 100 labels MiniLM
with a linear SGD classifier is ahead (46.2% vs 44.3% at 100, with lower NLL). These numbers are before calibration; the coverage claim comes
from the calibration gate.

Larger encoders and encoders fine-tuned on the training labels (on a GPU, contrastive loss) are
being measured with the same grid and rule (`uv run python -m backend.bench --gpu`).

**Calibration and the 99% gate.** One temperature is fitted on the calibration split by NLL.
The automatic-answer threshold is found by walking down from the most confident answers and
stopping at the first threshold whose Clopper–Pearson 95% lower bound on precision falls below
99.3%. Stopping at the first failure keeps the 95% guarantee even though many thresholds are
looked at. The threshold is selected at 99.3% while the claim stays at 99%, because a model just
above 99% still measures below it on many finite test sets. If no threshold qualifies, nothing is
answered automatically. A new model version is promoted only if the bound holds, its predictions
have not collapsed to one class, and its accuracy on a fixed regression set drops by at most 1
point; any earlier version can be restored.

**Does the claim survive re-splitting?** ([runs](results/s3/)) The held-out pool (calibration and
test) is shuffled into two halves 2,000 times; temperature and threshold are refitted on one half
and precision is measured on the other. A split passes if the answers it makes automatically are
at least 99% correct, or if it makes none. With 2,000 training labels:

| Dataset | Held-out pool | Pass rate | Mean share answered automatically | Worst 1% precision |
| --- | ---: | ---: | ---: | ---: |
| MTOP | 6,184 | 99.1% | 27.4% | 99.0% |
| BANKING77 | 3,630 | 99.0% | 1.8% | 98.6% |
| IPSA | 78,722 | 98.4% | 0.0% | 98.2% |

**On the fixed split** ([eval cards](results/s3/), 2,000 labels, MiniLM + logistic regression):

| Dataset | Accuracy | ECE | Answered automatically | Precision of those answers | Calib lower bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTOP | 86.2% | 0.017 | 31.7% | 99.62% | 99.47% |
| BANKING77 | 88.4% | 0.011 | 36.9% | 99.03% | 99.50% |
| IPSA | 49.4% | 0.112 | 0.0% | – | – |

Each card has its reliability bins and coverage–precision curve next to it.

The 99.3% selection level was chosen on MTOP (pass rate 93.4% at 99.0%) and then applied
unchanged to BANKING77 and IPSA. Across re-splits of half the held-out data, the model answers a
large share only on MTOP; on the other two it mostly abstains at that size, which is the gate working.

## Numbers

On [500 fixed held-out IPSA claims](results/s5/comparison.json), zero-shot Muse
Spark gets 37.0% right. The reference text classifier gets 36.2% with 200
training labels and 45.2% with 5,000. These are the same test items; training
uses only the fixed seed and stream partitions.

| Model | Accuracy | ECE-10 | p50 latency | Cost per 1,000 |
| --- | ---: | ---: | ---: | ---: |
| Muse Spark, zero-shot | 37.0% | null | 1,577.04 ms | $2.3451 |
| Text classifier, 200 labels | 36.2% | 0.1440 | 0.235 ms | unmeasured |
| Text classifier, 5,000 labels | 45.2% | 0.2078 | 0.231 ms | unmeasured |

Muse's confidence is self-reported; its NLL and ECE remain null. Its price
includes retry estimates at the full input/output list rates, with no cached
input discount. Remote HTTP latency and local CPU prediction latency are
different operating conditions. The [protocol and raw results](results/s5/)
record the sample, hashes, token usage and exploratory spend separately.

A [fresh matched version comparison](results/s5/provider-comparison.json) on
2026-10-29 measured Muse Spark 1.3 at **39.4%** and Spark Open 1026 at **37.4%**,
each with 500 valid outputs. Their conservative costs were $2.465243 and
$2.0050175 per 1,000; HTTP p50 was 1,892 and 1,610 ms. Combined spend was
$2.23513025. The operational default is 1.3 because its observed accuracy was
higher. Selection used this test sample; the two-point gap is descriptive,
not a significance claim or an independent estimate of winner quality.

These raw baselines do not establish 99% precision or automation coverage.
Hosted CPU cost and the calibrated model comparison await their measured inputs.

## Limits

- At 2,000 labels the gate answers almost nothing on IPSA, and on BANKING77 it certifies in only a
  few re-splits of half the held-out data.
- English only.
