# S3R: red team of the 99% claim

The unchanged gate fails when the correct intent is absent: it accepted five
of 500 messages, and all five answers were wrong. Its one-sided 95% precision
upper bound is 45.07%, below 99%. The claim cannot extend to arbitrary traffic
whose correct answer may be missing from the supplied options.

The 150M reader used exactly 200 history rows and eight nearest examples.
Calibration followed the supplied fixed train-derived split; test labels scored
outcomes only. Temperature and threshold were frozen before attacks. See the
[preregistered protocol](protocol.md) and [raw results](red-team.json).

| Condition | Accepted / tested | Accepted precision | Coverage | Coverage change |
| --- | ---: | ---: | ---: | ---: |
| Original messages | 147 / 500 | 99.32% | 29.40% | Reference |
| Semantic paraphrases | 11 / 50 | 100.00% | 22.00% | -4.00 pp |
| Typos | 91 / 500 | 100.00% | 18.20% | -11.20 pp |
| Shuffled options | 141 / 500 | 99.29% | 28.20% | -1.20 pp |
| Near duplicates | 135 / 500 | 100.00% | 27.00% | -2.40 pp |
| Exact repeats | 147 / 500 | 99.32% | 29.40% | +0.00 pp |
| Gold option removed | 5 / 500 | 0.00% | 1.00% | -28.40 pp |
| Long inputs | 0 / 500 | Unmeasurable | 0.00% | -29.40 pp |
| Noisy inputs | 0 / 500 | Unmeasurable | 0.00% | -29.40 pp |

Paraphrases use a separately matched 50-row baseline (13 accepted, 26% coverage,
100% accepted precision); all other attacks use the same 500 originals. These
author-written rewrites are a small semantic sample, not broad paraphrase coverage.
Typos left 18 messages unchanged; their outcomes remain included.

All 500 long inputs exceeded the reader token limit; the request can be truncated.
Noisy inputs had no reader truncations. Both conditions accepted zero answers:
precision is unmeasurable, and coverage fell by 29.40 percentage points. Exact
repeats produced no prediction or acceptance flips. Shuffling options changed 36
predictions and 26 acceptance decisions despite identical message texts.

The finite accepted samples cannot establish 99% precision: even the original
messages have a two-sided 95% exact interval of 96.27%–99.98%. Paired variants
are not independent evidence; intervals are descriptive without multiplicity
adjustment. No thresholds were tuned to these results. This study supplies no
CPU serving latency evidence.

Modal metered spend: **$0.04096949**, from app-filtered hourly
billing including GPU, CPU, memory and network egress. Paid Muse spend: **$0**.
Hashes, resource line items, runtime and row-level decisions are in the JSON.

Validation: `npm run check` passed (214 tests, frontend lint and production build).
No shared reader, gate-study, escalation, typed-model, or S2 result files changed.
Ren is independently built for Meta’s platform on Meta data.
