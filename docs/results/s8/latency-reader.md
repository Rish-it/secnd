# CPU reader latency gate

The 150M reader fails the one-second p95 bar with 32 examples. It passes with
zero or eight examples. The 32M reader passes all three measured conditions.
These are warm, sequential batch-one measurements on four reserved CPU cores;
they exclude HTTP, request queueing and cold model loading.

Both readers received the same 113 MTOP options, fixed 200-row history and
100 distinct measured requests per condition after 20 warmups. Each reader ran
in its own CPU-only Modal process with FP32 weights and four PyTorch threads.
MiniLM query embedding and nearest-history search were measured separately
inside each request. No embedding/search is needed in the zero-example case.

[Raw durations, checkpoints, runtime and billing](latency-reader.json).

| Reader | Examples | Reader p50 / p95 (ms) | Request p50 / p95 (ms) | Requests/s | Peak RSS (MiB) | ≤1s p95 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 150M | 0 | 571.92 / 644.04 | 571.92 / 644.04 | 1.73 | 1755.12 | Pass |
| 150M | 8 | 698.63 / 752.82 | 709.05 / 764.05 | 1.41 | 1756.72 | Pass |
| 150M | 32 | 1113.72 / 1335.27 | 1123.72 / 1345.69 | 0.87 | 1776.41 | Fail |
| 32M | 0 | 95.12 / 104.72 | 95.12 / 104.72 | 10.49 | 1322.05 | Pass |
| 32M | 8 | 123.08 / 139.08 | 134.58 / 153.17 | 7.48 | 1322.05 | Pass |
| 32M | 32 | 189.98 / 221.53 | 201.58 / 232.89 | 4.92 | 1330.82 | Pass |

| Reader | Examples | MiniLM embed + search p50 / p95 (ms) |
| --- | ---: | ---: |
| 150M | 8 | 10.76 / 12.68 |
| 150M | 32 | 11.00 / 12.88 |
| 32M | 8 | 11.25 / 14.53 |
| 32M | 32 | 11.06 / 13.23 |

Request timings include retrieval, tokenization, packing, reader forward and
softmax. Throughput is measured sequential requests divided by their elapsed
request time; it is not a concurrent-load benchmark. Separate embedding and
search durations are also retained in the JSON.

Peak RSS is process-wide and includes Python, the reader, MiniLM, the index and
setup allocations. It is cumulative across the three conditions for each model.
The sandbox reports the CPU model as `unknown`; host equivalence between the
two allocations is unverified. Treat these as measurements of these allocations,
not hardware-independent latency guarantees. No accuracy tradeoff for the smaller
reader is inferred from timing.

Modal metered spend: **$0.03113792**, including CPU, memory and
network egress. Muse spend: **$0**. No GPU or paid model API calls were used.

Validation: `npm run check` passed (203 tests, frontend lint and production build).
All 600 measured requests have raw durations; request/component timing consistency
and runner hashes were verified. No shared reader or S2 result files changed.
