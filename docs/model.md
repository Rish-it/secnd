# The Secnd decision model

## Why

Filled in as each part ships.

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

**How it was chosen** ([bake-off](results/s2/)): 4 text featurizers (hashed character n-grams,
potion-base-8M static embeddings, all-MiniLM-L6-v2, and n-grams stacked with potion) times 3
classifiers (logistic regression, linear SGD, k-nearest neighbours), at 50, 100, 200, 500 and all
labels, 5 seeds each, scored on the fixed test split. The decision rule was written before the
table was read: keep models within 1 accuracy point and 0.05 nats NLL of the best at 100 and 500
labels, then take the lightest.

| Labels | IPSA accuracy | BANKING77 accuracy |
| --- | ---: | ---: |
| 100 | 34.1% ± 2.8 | 43.3% ± 1.7 |
| 500 | 44.6% ± 1.1 | 77.7% ± 1.3 |
| all | 66.3% (1 seed) | 91.2% (1 seed) |

MiniLM with logistic regression was the only model inside the rule at both budgets on IPSA and the
best at every budget on BANKING77. These numbers are before calibration; the coverage claim comes
from the calibration gate.

Larger encoders and encoders fine-tuned on the training labels (on a GPU, contrastive loss) are
being measured with the same grid and rule (`uv run python -m backend.bench --gpu`).

**Calibration and the 99% gate.** One temperature is fitted on the calibration split by NLL. The
automatic-answer threshold is the lowest confidence where the Clopper–Pearson 95% lower bound on
precision is at least 99%; that needs at least 299 correct answers above it. If no threshold
qualifies, nothing is answered automatically. A new model version is promoted only if that bound
holds, its predictions have not collapsed to one class, and its accuracy on a fixed regression set
drops by at most 1 point; any earlier version can be restored.

**Does the claim survive re-splitting?** ([run](results/s3/)) The held-out pool (calibration and
test) is shuffled into two halves 2,000 times; temperature and threshold are refitted on one half
and precision is measured on the other. On BANKING77 with 2,000 labels:

| Splits | Pass rate | Splits that answer nothing | Coverage when answering (median) | Worst 1% precision |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 91.4% | 60.4% | 45.0% | 98.4% |

The target is a 95% pass rate, so the gate is not yet tight enough at this size.

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

- The 99% gate passes 91.4% of re-splits on BANKING77 at 2,000 labels, below its 95% target.
- English only.
