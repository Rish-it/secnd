# Reproduction checks

CI builds and starts the production image, checks database health and the
frontend, and sends the example CSV through the preview endpoint. Cleanup runs
on success and failure. The job is limited to ten minutes.

`uv run python -m backend.ci_eval` reproduces the committed IPSA and BANKING77 merchant
memory results with 200 training rows and seed 0, saved in
`eval-ipsa-merchant_memory-b200.json` and `eval-banking77-merchant_memory-b200.json`. Every field of that committed result must
match, including all quality metrics, per-class calibration, merchant slices,
threshold, bound, data/split hashes and seed. Its definition excludes measured
prediction latency, peak RSS, the revision SHA and the five-worst-class display. The original check omitted
that display because of architecture-dependent tie ordering; the seed quota
and worst-class ties now use stable label ordering. The BANKING77 check uses
the stable seed split. These cheap raw baselines check reproducibility, not
the calibrated 99% precision claim. The result retains every class's
calibration error; no numeric tolerance is applied. Changes to fields or schema
fail the check. The check neither overwrites
the reference nor downloads encoder weights. Dataset downloads are pinned and
hash-checked by the loader. CI limits the evaluation command to two minutes and
the entire job to ten minutes, using CPU only.

Container quickstart check:

```sh
curl --fail http://localhost:8000/api/health
curl --fail -F file=@examples/decisions.csv http://localhost:8000/api/preview
```

The eight-row file is illustrative and is not evaluation data.
