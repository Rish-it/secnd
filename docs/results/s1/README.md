# Baseline evaluation cards

Run `uv run python -m backend.eval --dataset ipsa --model tfidf_lr --budget 200`
to write a card. The filename and `model_version` identify the dataset, model,
label budget and random seed. Cards use the flat `EvalCard` response shape:

- Quality metrics are top-level fields. `classwise_ece` maps every label to its
  one-versus-rest calibration error using ten equal-width probability bins.
- `seen_merchant`, `unseen_merchant` and `worst_classes` entries are slice objects
  with `count` and `accuracy`, rather than row counts or bare scores.
- `predict_ms` contains `p50` and `p99`; `peak_rss_bytes` reports memory in bytes
  on both macOS and Linux. Unmeasured performance and cost fields stay null.
- `data_hash` is the SHA-256 of the JSON source-filename-to-pinned-digest map,
  with sorted keys and compact separators. This covers every source file,
  including all three years of expense claims. `split_hash` separately
  identifies row assignments in the fixed split manifest.
- `cp_lower_bound` is null when no calibration predictions meet the cutoff.
  A measured bound of zero remains `0.0`; it is different from no observations.

These are raw baseline measurements at `threshold: 0.99`, before calibration
and the release gate. Positive baseline coverage does not certify 99% precision;
read the measured test precision and calibration bound alongside it. The
robustness pass rate and release decision remain unmeasured.

Quality, slices and hashes are deterministic for the same inputs and seed.
Latency and process peak memory are measured anew and can vary between runs.
