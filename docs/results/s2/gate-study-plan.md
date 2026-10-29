# Gate study plan (written before any clean-split result was seen)

## Question
A more accurate model can certify less. The 150M reader reaches 93% on MTOP yet sometimes
certifies 0%. Each zero must be explained as one of three causes:

- too few calibration answers;
- weak confidence ranking;
- real residual errors.

Only then should a different gate rule be considered.

## Data
- Saved probabilities from the in-context readers (32M, 150M) and the logistic head, on the clean
  split: history / calib / test.
  - MTOP: official train / eval / test. Eval texts that any training set contains are dropped.
  - BANKING77: 3,000 training rows fixed with seed 0 / official test.
- Calib is cut in half, seeded:
  - one half fits the temperature, and is the pilot for rule B and the grid for rule C;
  - the other half certifies.
- Test labels never choose anything.

## Fixed settings
- The claim is 99% precision with 95% confidence (one-sided Clopper–Pearson).
- Selection target 0.993, alpha 0.05.
- None of these change in this study.

## Rules compared
All three rules use calib only.

- **A, current rule:**
  - walk down every distinct confidence on the certify half, stopping at the first cohort whose
    bound is below 0.993;
  - cohorts smaller than 427 are not tested.
- **B, pilot-fixed:**
  - on the pilot half, take the lowest threshold whose accepted answers are at least 99.7%
    correct (point estimate);
  - test that one threshold once on the certify half; pass or answer nothing.
- **C, fixed grid:**
  - nine thresholds set from the pilot half, at the confidences that accept 10%, 20% … 90% of
    pilot rows;
  - on the certify half, skip grid points with fewer than 427 accepted rows (a count, not a label);
  - then walk down the remaining points with the same stop-at-first-failure rule.

## Measures
- **Per model, set and history budget:**
  - test coverage and test precision, with the median over 200 random pilot/certify halvings;
  - the share of halvings that certify anything;
  - the share whose accepted test answers fall below 99% precision. Test precision is itself a
    sample, so this over-states real failures.
- **Headroom (diagnosis only, never a rule):** the largest test cohort, by confidence, whose
  precision is at least 99.3%. Scaled to the certify-half size:
  - headroom of at least 427 rows means the ranking is good enough and a zero is a sample-size
    problem;
  - headroom of none means real residual errors at that confidence;
  - in between means weak ranking.
- **Synthetic validity check:**
  - known correctness curves: monotone; with a dip of confident errors; and with heavy ties;
  - true precision of each accepted set is computed exactly;
  - 2,000 trials per curve and rule;
  - reports how often a rule accepts a set whose true precision is below 99%.

## Decision, fixed now
A replaces the default only if:

- B or C has higher median certified coverage on at least two of the three history budgets in
  both sets;
- the same rule keeps the synthetic unsafe rate at or below 5% on every curve.

Otherwise A stays, and the study reports which cause explains each zero.

## Result (added after the run; nothing above was changed)

Inputs:
- `incontext-ic-ettin-{150m,32m}-s6000-near-clean.json` and `gate-study.json`.
- Clean calib sizes:
  - MTOP: 1,203 rows (2,235 eval rows minus 1,032 whose text a training set contains), so the
    certify half is 602.
  - BANKING77: 2,998 rows, so the certify half is 1,499.

Findings:
- **Nothing certifies at 99%.** Median certified coverage is 0% for every rule, model, set and
  history budget (52 configurations).
  - The only certifications in any halving are BANKING77 with full history:
    - the logistic head certifies in 15% of halvings under A and 10% under C;
    - the 8-neighbour vote certifies in 1%.
  - MTOP accuracy matches the earlier runs: the 150M reader reaches 85.4% at 200 labels and
    93.4% with full history. BANKING77 now has 3,000 fewer history rows, so its full-history
    figures dip slightly (logistic head 92.5%, was 92.9%).
- **No rule is unsafe.** The share of halvings whose accepted test answers fall below 99% is at
  most 0.5%. On the synthetic curves the unsafe rate is at most 1.25%, under A.
- **Causes of the zeros (150M):**
  - 18 weak ranking;
  - 4 sample size;
  - 4 residual errors, all from the 8-neighbour vote.
  - MTOP's 602-row certify half can only pass a cohort of at least 427 answers (71%) with zero
    errors. The best test headroom is 78%, for the 32-example reader with full history.
- **Decision (pre-registered):** B and C never beat A, so A stays.
- **Earlier results:** the certified shares reported before this split were measured with the
  temperature and threshold fitted on about 2,200 test rows. They are withdrawn as
  certification evidence.
