# Configurable reader serving

The app defaults to the 150M CPU reader. Automatic reader answers require the
full audited MTOP history, a complete frozen policy, enough confidence and enough
MiniLM support for the predicted intent. Other histories require a person.
[Configuration, previous behavior and safety scope](../../serving.md).

## Running API CPU measurement

[Raw HTTP durations and environment](http-cpu.json) record **172.1 ms p50**
and **252.0 ms p95** on **Apple M4 Pro**, PyTorch 2.15.0, CPU FP32,
four threads, eight examples, 113 options and the complete 11,667-row audited
history. Twenty warmups precede 100 distinct sequential test requests through a
running Uvicorn process over loopback HTTP. The caller includes SQL history reads,
scope hashing, query embedding/retrieval, tokenization, reader inference, frozen
temperature/threshold/support checks, prediction persistence and JSON response.

The first warmup builds the history embedding index. Startup and cold history
indexing are excluded from the warm p50/p95 and are separate from steady-state
requests. This is one host and a sequential measurement; it establishes no
concurrent-load, throughput, universal latency or new quality claim. GPU performance
was not measured here. The CPU run uses the frozen GPU-derived policy without
refitting; it does not create an independent CPU precision certification.

Reproduce with the already-installed checkpoint, pinned MiniLM cache and original
MTOP archive, all kept outside Git:

```sh
uv run python -m backend.tests.http_reader --artifacts DATA_DIRECTORY \
  --output docs/results/c3-serving/http-cpu.json
```

The runner verifies the fixed split hash and audited history fingerprint, uses a
temporary database and reads the original checkpoint through a symlink. It makes
no downloads, teacher calls, sends or account changes. `--hold --port 8033` leaves
the app available for browser inspection until interrupted. No history or
calibration/test labels are modified.

## Browser evidence

- [Live desktop](live-desktop.png) and [live 390px](live-mobile-390.png): installed
  CPU reader, full audited history, active frozen checks. [Assertions](browser-live.json)
  record no horizontal overflow and no page errors.
- [Fixture desktop](desktop.png) and [fixture 390px](mobile-390.png): the smaller
  synthetic history requires confirmation. The fixture banner distinguishes
  stand-in inference from the separately displayed committed benchmark results.
- `npm run test:reader` reproduces the fixture browser scenario, including the
  400M zero-example null-policy display. It needs no checkpoint or GPU.

Published accuracy and confidence-only certified share/precision come from the
existing [150M card](../s2/incontext-ic-ettin-150m-s6000-near-clean.json) and the
unchanged [400M card](../s2/incontext-ic-ettin-400m-s6000-near-clean.json). The separate
support-audit sample comes from unchanged copies of
[150M](../s3r/attacks-floor.json) and [400M](../s3r/attacks-floor-400m.json) results
from commit `6f192e0`. The earlier certified share excludes the new support floor;
the UI labels those measurements separately. Small audit samples do not establish
a universal 99% precision bound or certify smaller workspace histories.

## Validation

[Checks](checks.json): `npm run check` passed with **243 backend tests**, including
**25 reader tests**, Python lint/format checks, frontend lint and production build.
Two existing React effect warnings remain. **One browser scenario** passed.
`docker compose up --build` passed with the explicitly selected head backend on a
separate loopback port, followed by health/frontend/CSV checks. Both packaged
reader policies verified offline, and a container without the configured reader
failed startup with the expected clear error. Reader weights/cache remain an
operator-supplied prerequisite for reader mode.

The 400M model/device selection uses a stand-in in tests. No GPU jobs, live sends,
ad changes or paid calls were made. Protected model/training/evaluation/gate code
and existing committed results were unchanged.
