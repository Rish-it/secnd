# S3R: preregistered red team of the 99% precision claim

Register this protocol in the pull request before running inference. No training,
gate changes, attack-dependent calibration, or paid teacher calls are permitted.

## Frozen baseline

- Reader: `ic-ettin-150m-s6000-near`.
- Split contract: commit `e2581eecca945c5abbf5e5158c71eb51eac49067`.
  Shuffle official English MTOP train with pandas `random_state=0`; first 4,000
  rows are calibration, remaining 11,667 are history candidates. Official test
  (4,386 rows) is used only for scoring. Labels retain the sorted MTOP order.
- History: first 200 candidates in a NumPy `default_rng(0)` permutation.
  Eight nearest MiniLM examples, from this history only. No test variants enter
  history. Changed messages are embedded again before retrieval.
- Calibration: original 4,000 calibration messages, inference batch size 32,
  CUDA BF16 autocast as in the supplied evaluation. A seed-0 NumPy permutation
  partitions calibration into 2,000 temperature-fit and 2,000 certification rows.
  Use the existing `fit_temperature`, `scale`, and `choose_threshold` unchanged.
  Freeze full-precision temperature and threshold before any attack scoring.
- Primary sample: first 500 official test rows in a NumPy
  `default_rng(20261030)` permutation. Frozen semantic paraphrases cover its first
  50 rows; score these against the same 50 original messages separately.

## Attacks

1. **Paraphrase:** the frozen 50 rewrites in `paraphrases.json`, copied unchanged
   from the prior preregistration, SHA-256
   `acf30e35d78c2b742674f994ef6065cecdad539db39762403587005232d7424e`.
   These author-written rewrites were prepared without reader predictions. Their
   limited size and possible interpretation differences must accompany results.
2. **Typo:** using a per-row seed `20261030 + source_row`, choose one alphabetic
   word with at least five characters; transpose two distinct adjacent internal
   letters. Preserve rows with no eligible word and report unchanged counts.
3. **Option order:** per-row seeded permutation of all 113 options. Map predicted
   slots back to original label IDs before scoring.
4. **Near duplicate:** lowercase, collapse whitespace, and append a seeded choice
   of punctuation (`.`, `!`, `?`). Keep each variant paired with its original;
   never count it as additional independent evidence. Also report exact repeat
   prediction/acceptance flips as a determinism check.
5. **Missing intent:** remove the gold option and exclude history examples bearing
   that label. Every accepted answer is an error. Report false-accept rate.
6. **Long input:** prefix 1,024 seeded words drawn from `log`, `record`, `status`,
   `metadata`, `trace`, `entry`, `value`, `note`, then `Request: ` and the original.
   This deliberately probes truncation of the request. Report actual token
   truncation using the untruncated rendering length.
7. **Noisy input:** surround the original request with 100 seeded noise words on
   each side, using the same vocabulary. This tests retrieval and context noise.

Each attack uses an independent RNG reset to the per-row seed. Seeds, texts,
option orders and sample are fixed before outcomes; do not edit or exclude failed
cases. Semantic paraphrases have their own matched 50-row baseline. Other attacks
and exact repeat use the same 500-row baseline.

## Reporting and interpretation

Save `red-team.json` and a short write-up. For every condition report sample size,
accepted count, accepted errors, accepted precision, difference from 0.99, coverage,
coverage change in percentage points against its matched baseline, accuracy,
prediction/acceptance flips, unchanged messages, and truncations. Precision is null
when nothing is accepted; that establishes no precision claim. Include two-sided
95% exact binomial intervals and a one-sided 95% upper bound: a bound below 0.99
is evidence against the target for that condition. Intervals are descriptive,
without multiplicity adjustment; do not pool paired variants across conditions.

Record split/history/checkpoint/runner/gate/protocol hashes, runtime versions,
GPU identity, timings, and Modal resource usage/spend. Distinguish metered spend
from estimates; unknown billing must remain null. Report all conditions, including
failures. Passing this finite suite does not establish a universal 99% guarantee.

Use Modal's existing benchmark volume and A10G pattern; no model training. Paid
Muse spend is zero because this protocol makes no Muse calls. Serving remains CPU.
