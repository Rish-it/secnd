# Configurable reader serving

The app defaults to `head`, preserving each workspace's own calibrated model.
Reader selection is explicit. Automatic reader answers require the complete
MTOP history fingerprint, the frozen confidence and support cutoffs, and an
inactive spot-check guard. [Configuration and demo precautions](../../serving.md).

## Audited policies and benchmark

Policies are unchanged copies from `ada0a0d`:
[150M](../s3r/attacks-floor-audited.json),
[400M](../s3r/attacks-floor-audited-400m.json).
**Calibration labels were changed by a model audit, not a human audit.**
History and test labels did not change. Checksums pin the exact frozen files.

The floored benchmark displayed by the app is
[floor-full-test.json](../s3r/floor-full-test.json), covering all 4,386 unchanged
MTOP test labels at full history. At eight examples, floored shares are 10.85%
(150M) and 13.89% (400M); observed accepted precision is 100% in both, with the
recorded 95% lower bounds 99.23% and 99.40%. Confidence-only shares are not shown
as floored coverage. The 500-row missing-intent audit is a separate measurement.
The 150M eight-example missing-intent condition has two wrong accepts; the 400M
32-example full-test lower bound is 98.86%. These misses remain in the frozen files.
No benchmark certifies arbitrary or smaller histories or future traffic.

## Running API CPU measurement

[Audited-policy HTTP durations and environment](http-cpu-audited.json):
**162.5 ms p50 / 231.4 ms p95**, Apple M4 Pro,
PyTorch 2.15.0, CPU FP32, four threads, eight examples, 113 options and
11,667 history rows. Twenty warmups precede 100 distinct sequential HTTP decisions
through Uvicorn. Timing includes SQL history reads, scope hashing, MiniLM retrieval,
tokenization, reader inference, frozen policy and spot-check/guard checks, SQLite
persistence and JSON response. The file identifies the audited policy and checksum.

Model startup and initial history indexing are excluded from warm p50/p95.
This is one host, without a concurrent-load, throughput or new quality claim.
GPU performance is not measured here. Applying the frozen GPU-derived policy on
CPU does not create an independent CPU precision certification.

```sh
uv run python -m backend.tests.http_reader --artifacts DATA_DIRECTORY \
  --output docs/results/c3-serving/http-cpu-audited.json
```

Use already-installed weights, pinned MiniLM cache and the original MTOP archive.
The runner checks the original split and unchanged history fingerprint, explicitly
selects the reader for its workspace, reads weights through a symlink and creates
a temporary database. `--hold --port 8033` keeps it available for browser inspection.
For the demo, show automatic answers before confirming, or use this disposable
workspace: a person's answer changes the exact history and pauses reader automatic
answers. Stop and rerun for a fresh copy; do not edit a real database to restore it.

## Browser evidence

- [Live desktop](live-desktop-audited.png) / [live 390px](live-mobile-audited-390.png):
  real installed CPU reader, full history, audited policy, full-test floored benchmark.
  [Assertions](browser-live-audited.json): no horizontal overflow or page errors.
- [Fixture desktop](desktop-audited.png) / [fixture 390px](mobile-audited-390.png):
  synthetic smaller history requires confirmation. The fixture is visibly marked.
- These screens were later removed from the product, along with their browser test.
  The backend tests still cover the audited policy without a checkpoint or GPU.

Earlier `http-cpu.json`, `browser-live.json`, `checks.json` and screenshot files
are retained as historical evidence of the pre-audit implementation. They do not
describe the revised serving policy or current UI.

## Validation before integration rebase

[Checks](checks-audited.json): **249 backend tests**, including **31 reader tests**,
passed with Python checks, frontend lint and build. Two existing React effect
warnings remain. **One browser scenario** passed. Default `docker compose up --build`
passed on a separate loopback port with no backend override, followed by health,
frontend and CSV checks. CI no longer overrides the backend; tests use the real
head default. GPU config/model loading tests use stand-ins.

The rebase and final integrated rerun wait for #37 and #38 to merge. Neither PR nor
#39 has been merged by this work. Frozen values and protected model, training,
evaluation and gate code are unchanged. No GPU jobs, live sends, account changes or
paid calls were made.
