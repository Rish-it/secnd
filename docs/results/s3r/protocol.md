# S3R preregistration

Registered 2026-10-30 before attack inference. Baseline: merge c543ca4 (#29).
Checkpoint: ic-ettin-150m-s6000-near. No training or threshold tuning.

Use the official English MTOP history and test, and the 1,203 clean calibration
rows obtained by the existing evaluate() training-overlap filter. Seed 0 splits
calibration into temperature-fit and certification halves. Recompute the existing
choose_threshold rule only on calibration, freeze temperature and threshold, and
apply them unchanged to every attack. Evaluate 0, 8-nearest and 32-nearest with
full history. Save split, checkpoint and protocol hashes with results.

Primary sample: first 500 test rows in a seed-20261030 permutation, chosen before
inference. Each variant retains the source row ID; variants are paired, not
independent evidence. Report baseline alongside each attack:

- Paraphrase: prefix each message with “Please help me with this request: ”.
  This tests a narrow, meaning-preserving wrapper, not broad semantic paraphrases.
- Typo: transpose the first two internal letters of the first alphabetic word
  of length at least five; unchanged rows are counted and reported separately.
- Option order: independent seed-20261030 permutation per source row; map output
  slots back to label IDs before scoring.
- Duplicates: exact repeat and a lowercase/whitespace-normalized variant. Report
  prediction and acceptance flips per original, never inflate sample size by
  treating repeated texts as independent observations. Retrieve from history only.
- Missing intent: remove the gold intent from options and remove history examples
  of that intent. Every accepted answer is an error; report false-accept rate.

Re-encode changed texts for retrieval. Do not add test variants to history.
Record accepted count, errors, coverage, accepted precision (null if none), exact
binomial 95% interval, and difference from 99%. Intervals over unique source rows
are descriptive; do not pool correlated attacks. Zero acceptance establishes no
precision claim. Report every condition, including failures and truncation counts.

CPU latency: same checkpoint, FP32 eval/inference mode, batch size one, CPU only,
113 options, 0/8/32 examples, one process, four torch threads. Twenty warmups then
100 sequential distinct test requests per condition. Measure tokenization,
packing, forward and softmax together; precomputed nearest examples are excluded
and that exclusion must accompany results. Record p50/p95, raw durations, token
lengths, process peak RSS, model-load time, CPU/OS/torch versions. No GPU latency
may substitute for CPU measurements. No shipping decision without a latency SLA.

The committed #29 card currently certifies zero rows in all these conditions.
The handoff's 23–37% figure has no supporting artifact in this checkout. If a
newer gate is supplied, register its identity and any protocol amendment before
running it; do not silently replace this baseline.

## Amendment 1 (before inference)

The owner supplied pushed commit e2581ee on stage/s2f-mtop-calib. Use that
held_out() implementation instead: seed-0 4,000-row MTOP calibration from official
train, leaving 11,667 history rows; certification has 2,000 rows. All reader
training excluded these MTOP train texts. The earlier 602-row result is superseded.
Keep rule A unchanged, fit temperature and threshold only on the new calibration
split. Full-history 0/8/32 conditions remain the registered primary audit.
