# Reproduction checks

CI builds and starts the production image, checks database health and the
frontend, and sends the example CSV through the preview endpoint. Cleanup runs
on success and failure. The job is limited to ten minutes.

`uv run python -m backend.ci_eval` reproduces the committed BANKING77 merchant
memory card with 200 training rows and seed 0. Every deterministic field must
match, including all quality metrics, per-class calibration, slices, threshold,
bound, data/split hashes, seed and schema. The only excluded fields are measured
prediction latency, peak RSS and the revision SHA. The check neither overwrites
the reference nor downloads encoder weights. Dataset downloads are pinned and
hash-checked by the loader. CI limits the evaluation command to two minutes and
the entire job to ten minutes, using CPU only.

Container quickstart check:

```sh
curl --fail http://localhost:8000/api/health
curl --fail -F file=@examples/decisions.csv http://localhost:8000/api/preview
```

The eight-row file is illustrative and is not evaluation data.
