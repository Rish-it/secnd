# Wider training pools for the 400M reader

The 400M reader was retrained on wider pools to test whether more kinds of decision
data improve it. Evaluation sets, shots, the 99% gate and the fixed test sets did not
change. No calibration or test text enters training. All rows below are one training run
with 6,000 steps, unless a seed is named.

## Which pools can ship

| Pool | Sets | Rows | Can ship | Why |
| --- | ---: | ---: | --- | --- |
| [base](incontext-ic-ettin-400m-s6000-near.json) | 7 | 256,409 | No | ATIS has no licence |
| [wide](incontext-ic-ettin-400m-s6000-near-wide.json) | 14 | 441,426 | No | ATIS, and a FEVER mirror with no licence |
| [wide-muse](incontext-ic-ettin-400m-s6000-near-wide-muse.json) | 17 | 442,882 | No | As above, plus 350 written rows traced to real posts |
| [clean](incontext-ic-ettin-400m-s6000-near-wide-clean.json) | 15 | 407,727 | Yes | Those sets and rows removed |
| [fine](incontext-ic-ettin-400m-s6000-near-wide-fine.json) | 38 | 422,906 | Yes | Clean, plus 23 written fine-grained taxonomies |

**Ship candidate: fine, [seed 1](incontext-ic-ettin-400m-s6000-near-wide-fine-seed1.json).** It certified
624 calibration rows at 8 nearest against seed 0's 564, and 920 at 32 against 0. Test sets chose nothing.

The fine pool adds 15,179 customer messages across 1,380 intents in 23 industries, written by
Muse Spark Open from intent names. Banking was never written. Every intent in a set is a close
neighbour of others, which is the shape of BANKING77.

## Test accuracy

| Pool | MTOP none | MTOP 8 nearest | MTOP 32 nearest | BANKING77 none | BANKING77 8 nearest | BANKING77 32 nearest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 77.61% | 93.66% | 94.28% | 63.96% | 88.96% | 85.36% |
| wide | 77.70% | 92.36% | 92.41% | 63.86% | 88.73% | 86.82% |
| wide-muse | 82.35% | 94.21% | 94.12% | 63.12% | 88.70% | 85.84% |
| clean, seed 0 | 77.29% | 90.26% | 90.63% | 63.12% | 88.38% | 85.91% |
| clean, [seed 1](incontext-ic-ettin-400m-s6000-near-wide-clean-seed1.json) | 77.29% | 91.79% | 91.45% | 64.61% | 88.57% | 85.58% |
| fine, seed 0 | 78.64% | 92.93% | 93.23% | 68.90% | 90.62% | 87.44% |
| **fine, seed 1** | **75.99%** | **92.38%** | **93.59%** | **68.99%** | **89.90%** | **86.23%** |

Fine is the best pool that can ship. It leads every other pool on BANKING77 on both seeds. On MTOP,
the base pool stays ahead by up to 1.3 points, and it cannot ship. Seed 1 gives up 2.65 points of
no-example MTOP accuracy against seed 0.

## Certified coverage on MTOP, on raw labels

| Pool | None | 8 nearest | Accepted precision at 8 | 32 nearest |
| --- | ---: | ---: | ---: | ---: |
| base | 0.00% | 25.08% | 100.00% | 0.00% |
| wide | 25.44% | 21.36% | 99.89% | 49.50% |
| wide-muse | 0.00% | 64.75% | 99.72% | 61.33% |
| clean, seed 0 | 0.00% | 32.22% | 99.93% | 33.11% |
| clean, seed 1 | 0.00% | 51.62% | 99.87% | 50.93% |
| fine, seed 0 | 25.44% | 28.59% | 99.92% | 0.00% |
| fine, seed 1 | 0.00% | 33.22% | 99.93% | 47.51% |

**One run cannot establish a coverage gain.** The two clean seeds differ by 19 points on the same
data, because a single wrong answer stops the threshold walk. Coverage on BANKING77 is zero in
every row at the 99% bar.

These numbers use raw labels. The 13.89% result in [S3R](../s3r/README.md) uses model-audited
calibration labels and a support floor, so the two are not directly comparable.

## How much history a person needs

[Fine, seed 1](incontext-ic-ettin-400m-s6000-near-wide-fine-seed1-clean.json), test accuracy, with
certified MTOP coverage in brackets (accepted precision 99.93% or better, raw labels):

| History | MTOP reader, 8 nearest | MTOP head | BANKING77 reader, 8 nearest | BANKING77 head |
| --- | ---: | ---: | ---: | ---: |
| 200 labels | 82.51% (23.78%) | 69.54% | 75.39% | 54.16% |
| 1,000 labels | 88.12% (32.76%) | 80.46% | 82.60% | 85.32% |
| all labels | 92.45% (33.49%) | 93.00% | 89.87% | 92.50% |

[Fine, seed 0](incontext-ic-ettin-400m-s6000-near-wide-fine-clean.json) is more accurate with little
history: 85.73% and 77.14% at 200 labels, 90.04% and 84.19% at 1,000. It certifies nothing at 200.

[Clean, seed 0](incontext-ic-ettin-400m-s6000-near-wide-clean-clean.json), for comparison:

| History | MTOP reader, 8 nearest | MTOP head | BANKING77 reader, 8 nearest | BANKING77 head |
| --- | ---: | ---: | ---: | ---: |
| 200 labels | 83.22% | 69.54% | 71.14% | 54.16% |
| 1,000 labels | 86.21% | 80.46% | 79.45% | 85.32% |
| all labels | 90.26% | 93.00% | 88.38% | 92.50% |

With few labels the reader wins. With thousands, a logistic head on the same retrieval vectors
wins. Serving should choose by history size rather than always reading.

## BANKING77 with a model audit

The [registered second audit](../s3r/protocol.md) cost $0.65. Three models from different labs voted on
9 confident disagreements and 100 controls in calibration. It overturned 7 disagreements and 2
controls, exactly at the 2% limit, so it counts. Coverage at the 99% bar stays **zero**: the first wrong
answer moved from rank 200 to rank 254, and the gate needs 427. There is no third attempt.
[Result and every changed label](../s3r/banking77-audit-2.json). Test labels are unchanged, and this
is a model audit, not a human audit.

Seed 1, 8 nearest, full history, selected at each bar plus 0.3 points as at 99%. Lower bars were
measured after the 99% result, so they are descriptive:

| Bar | Raw: test coverage | Raw: test precision | Audited: test coverage | Audited: test precision |
| ---: | ---: | ---: | ---: | ---: |
| 99% | 0% | – | 0% | – |
| 98% | 12.56% | 99.74% | 17.31% | 99.81% |
| 95% | 74.16% | 96.58% | 77.24% | 96.09% |
| 90% | 96.27% | 91.57% | 96.72% | 91.44% |

## BANKING77 with no examples

The reader alone measures 68.90% and 68.99% on the test set with the fine pool's two seeds. Separately, heads trained on
100 messages per label written by Muse Spark Open reached 70.07% on calibration, and rereading the
top shortlist did not improve it: the correct label is in the top 10 about 96% of the time, but
reader confidence cannot separate near-identical options. Secnd has no lead in this setting.

## Typed decisions

Fixed 2,000-row test, ±1.7 points at 95%. Validation picks the epoch.

| Reader used to start | Seed | Validation | Test | ECE |
| --- | ---: | ---: | ---: | ---: |
| base | 0 | 80.83% | 79.50% | 0.0299 |
| base, rerun | 0 | 80.67% | 79.95% | 0.0177 |
| wide | 0 | 81.00% | 80.15% | 0.0167 |
| wide | 1 | 81.83% | 79.40% | 0.0174 |
| wide | 2 | 82.50% | 80.90% | 0.0152 |
| wide | 3 | 80.50% | 78.95% | 0.0250 |
| wide-muse | 0 | 82.17% | 79.55% | 0.0371 |
| clean | 0 | 80.67% | 79.75% | 0.0202 |
| fine, seed 0 | 0 | 81.83% | 78.80% | 0.0147 |
| **fine, seed 1** | **1** | **82.17%** | **79.70%** | **0.0231** |
| fine, seed 1 | 2 | 80.50% | 79.15% | 0.0224 |

The same seed moved 0.45 points on rerun. Differences of this size are noise. Ship by validation
across seeds, never by test.

## Limits

The fixed test sets were read after every round, so comparisons between pools are descriptive.
The fine pool has two seeds. Written rows come from one model and carry its style. A release needs
more seeds, audited-floor refits and a model card tied to the exact weights.
