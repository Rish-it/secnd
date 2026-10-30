# S3R: accepted precision under attack and CPU serving

Removing the correct intent from the option list breaks a universal 99% accepted-precision claim. The unchanged gate accepted wrong answers in every missing-intent condition. The other registered conditions measured at least 99% accepted precision on this sample, with too few accepted answers for their confidence intervals to establish a 99% lower bound.

## Frozen method

Checkpoint: `ic-ettin-150m-s6000-near`, using the 4,000-row calibration split supplied at `e2581ee`. Calibration has 2,000 temperature-fit and 2,000 certification rows; full history has 11,667 rows. The fixed test has 4,386 rows. Rule A is unchanged (selection level 99.3%, one-sided calibration alpha 0.05). No attack outcome chooses the temperature or threshold.

The [preregistration](protocol.md) preceded inference: initial protocol `d80a890`, split amendment `7bc367e`, numerical-precision clarification `761ce97`, natural rewrites `74e69ed`, and 32M CPU comparison `85ae0e3`. The authorship clarification changes no inputs. The primary sample is the first 500 rows of a seed-20261030 test permutation. All three shot counts use full human-labelled history; the 200/1,000-label and teacher-history conditions are outside this attack audit.

| Examples | Frozen temperature | Frozen threshold |
| ---: | ---: | ---: |
| 0 | 1.512632518 | 0.983959615 |
| 8 | 1.250791509 | 0.995631099 |
| 32 | 1.194290755 | 0.997082770 |

Attacks run on a GPU with the original CUDA BF16 autocast; serving measurements below use CPU FP32. The same temperature and threshold apply to every variant within a shot condition. Nearest examples are re-retrieved for changed messages, from history only. Missing-intent requests have 112 options and exclude history examples carrying the missing label. Every such accepted answer is incorrect.

## Primary attacks: 500 paired sources per row

Intervals are descriptive two-sided exact binomial 95% intervals over unique source rows within each condition. Variants are paired; they are not pooled as independent evidence. “Difference” is accepted precision minus the 99% target, in percentage points.

| Examples | Attack | Accepted | Errors | Accepted precision (95% interval) | Difference |
| ---: | --- | ---: | ---: | --- | ---: |
| 0 | Original | 115/500 | 0 | 100.00% (96.84%–100.00%) | +1.00 |
| 0 | Polite prefix | 29/500 | 0 | 100.00% (88.06%–100.00%) | +1.00 |
| 0 | Transposed letters | 46/500 | 0 | 100.00% (92.29%–100.00%) | +1.00 |
| 0 | Shuffled options | 115/500 | 1 | 99.13% (95.25%–99.98%) | +0.13 |
| 0 | Exact repeat | 115/500 | 0 | 100.00% (96.84%–100.00%) | +1.00 |
| 0 | Lowercase / whitespace | 114/500 | 0 | 100.00% (96.82%–100.00%) | +1.00 |
| 0 | Correct intent absent | 4/500 | 4 | 0.00% (0.00%–60.24%) | -99.00 |
| 8 | Original | 189/500 | 1 | 99.47% (97.09%–99.99%) | +0.47 |
| 8 | Polite prefix | 84/500 | 0 | 100.00% (95.70%–100.00%) | +1.00 |
| 8 | Transposed letters | 83/500 | 0 | 100.00% (95.65%–100.00%) | +1.00 |
| 8 | Shuffled options | 193/500 | 1 | 99.48% (97.15%–99.99%) | +0.48 |
| 8 | Exact repeat | 189/500 | 1 | 99.47% (97.09%–99.99%) | +0.47 |
| 8 | Lowercase / whitespace | 190/500 | 1 | 99.47% (97.10%–99.99%) | +0.47 |
| 8 | Correct intent absent | 9/500 | 9 | 0.00% (0.00%–33.63%) | -99.00 |
| 32 | Original | 167/500 | 0 | 100.00% (97.82%–100.00%) | +1.00 |
| 32 | Polite prefix | 80/500 | 0 | 100.00% (95.49%–100.00%) | +1.00 |
| 32 | Transposed letters | 81/500 | 0 | 100.00% (95.55%–100.00%) | +1.00 |
| 32 | Shuffled options | 182/500 | 0 | 100.00% (97.99%–100.00%) | +1.00 |
| 32 | Exact repeat | 167/500 | 0 | 100.00% (97.82%–100.00%) | +1.00 |
| 32 | Lowercase / whitespace | 168/500 | 0 | 100.00% (97.83%–100.00%) | +1.00 |
| 32 | Correct intent absent | 9/500 | 9 | 0.00% (0.00%–33.63%) | -99.00 |

Missing-intent false-accept rates are 0 examples: 0.80%, 8 examples: 1.80%, 32 examples: 1.80%. This directly limits the claim to an in-distribution option set containing the answer; it does not invalidate the separately measured clean-split point estimates.

## Stability and duplicate handling

| Examples | Variant | Prediction flips / 500 | Acceptance flips / 500 |
| ---: | --- | ---: | ---: |
| 0 | Shuffled options | 24 | 30 |
| 0 | Exact repeat | 0 | 0 |
| 0 | Lowercase / whitespace | 6 | 11 |
| 8 | Shuffled options | 16 | 34 |
| 8 | Exact repeat | 0 | 0 |
| 8 | Lowercase / whitespace | 2 | 9 |
| 32 | Shuffled options | 15 | 41 |
| 32 | Exact repeat | 0 | 0 |
| 32 | Lowercase / whitespace | 4 | 7 |

Repeats are scored against their original source IDs. Repeated predictions never inflate precision sample sizes. Lowercase/whitespace normalization is a narrow near-duplicate test, not a general semantic-duplicate benchmark.

| Examples | Text transform | Unchanged texts / 500 | Accepted changed texts | Errors on accepted changed texts |
| ---: | --- | ---: | ---: | ---: |
| 0 | Polite prefix | 0 | 29 | 0 |
| 0 | Transposed letters | 26 | 44 | 0 |
| 0 | Lowercase / whitespace | 120 | 89 | 0 |
| 8 | Polite prefix | 0 | 84 | 0 |
| 8 | Transposed letters | 26 | 80 | 0 |
| 8 | Lowercase / whitespace | 120 | 155 | 1 |
| 32 | Polite prefix | 0 | 80 | 0 |
| 32 | Transposed letters | 26 | 77 | 0 |
| 32 | Lowercase / whitespace | 120 | 134 | 0 |

Sequences at the 1,536-token cap: **0** across all primary variants (an upper bound on actual truncations).

The [split audit](split-audit.json) found 500 unique exact and normalized source texts in the primary sample, with no exact history matches. Three match history after lowercase/whitespace normalization; seven match under the stronger training normalizer. The full test contains 4,384 distinct exact texts among 4,386 rows, seven exact history matches and 69 matches under `backend.finetune._norm`. These overlaps are retained to preserve the supplied fixed split. “Clean” here means reader-training text exclusion and separated calibration roles; it does not mean a new globally deduplicated history/calibration/test split.

## Natural paraphrases: separate 50-source supplement

The [frozen rewrites](paraphrases.json) were written by an AI coding assistant from the first 50 primary source texts without inspecting per-source predictions or gold labels. The supplement was registered after the primary run had begun and aggregate zero-shot counts were visible, before supplement inference. This is a small authored sample, without independent annotation or broad paraphrase coverage.

| Examples | Variant | Accepted / 50 | Errors | Accepted precision (95% interval) | Prediction flips | Acceptance flips |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | Original | 12 | 0 | 100.00% (73.54%–100.00%) | — | — |
| 0 | Natural rewrite | 11 | 0 | 100.00% (71.51%–100.00%) | 5 | 3 |
| 8 | Original | 17 | 0 | 100.00% (80.49%–100.00%) | — | — |
| 8 | Natural rewrite | 13 | 0 | 100.00% (75.29%–100.00%) | 7 | 6 |
| 32 | Original | 14 | 0 | 100.00% (76.84%–100.00%) | — | — |
| 32 | Natural rewrite | 12 | 0 | 100.00% (73.54%–100.00%) | 8 | 2 |

No accepted errors were found in this supplement. Coverage falls under rewriting; its small accepted counts cannot establish a 99% precision guarantee.

## CPU serving: single requests, 113 options

Both models use a Modal `cpu=4`, 16 GiB allocation, four torch threads, FP32 and batch size one. Twenty warmups precede 100 timed sequential requests per shot count. The same 120 source IDs are used for both models (the first 20 are warmups). Timings include rendering/tokenization, packing, forward and softmax. They exclude retrieval, HTTP, queueing, temperature scaling and gate application. The processes are separate cloud allocations, without a controlled physical host; the guest reports the CPU model as `unknown`.

| Reader | Examples | p50 ms | p95 ms | Process peak RSS GiB | Token range | Meets 1 s p95 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 150M | 0 | 325.3 | 347.3 | 2.418 | 429–451 | yes |
| 150M | 8 | 400.5 | 450.9 | 2.418 | 492–596 | yes |
| 150M | 32 | 610.2 | 676.8 | 2.418 | 689–980 | yes |
| 32M | 0 | 92.7 | 102.6 | 1.780 | 429–451 | yes |
| 32M | 8 | 116.8 | 128.2 | 1.780 | 492–596 | yes |
| 32M | 32 | 179.1 | 206.2 | 1.780 | 689–980 | yes |

The 150M reader passes the requested one-second p95 budget at all three context sizes under these measured conditions. A [separate measurement](../s8/latency-reader.md) on another allocation, with retrieval included, measured 1,346 ms p95 at 32 examples, so the 32-example result depends on the host; with up to eight examples both measurements pass. The 32M alternative is faster; these measurements alone do not compare quality at a chosen workspace operating point.

Peak RSS is the cumulative Linux process high-water mark, including model loading, checkpoint hashing, the retrieval embedder/index and temporary initialization buffers. It is neither incremental reader memory nor a full API service memory budget. Model/tokenizer load times were 3.30 s (150M) and 3.63 s (32M). PyTorch is `2.15.0+cpu`. Raw timings, token lengths, environment, split hash and checkpoint file hashes are in the cards. This is a warm sequential benchmark, not a throughput or concurrent-load test and not an end-to-end one-second API guarantee.

## Reproduce and inspect

From the repository root with Modal access to `secnd-bench`:

```sh
uv sync --locked
uvx modal run backend/modal_reader_audit.py
uvx modal run backend/modal_reader_audit.py --natural
uvx modal run backend/modal_reader_audit.py --latency
uvx modal run backend/modal_reader_audit.py --latency --model-name ic-ettin-32m-s6000-near
```

[Primary raw results](attacks.json) · [Natural paraphrase raw results](paraphrase-results.json) · [150M CPU card](cpu.json) · [32M CPU card](cpu-32m.json). These artifacts store predictions and source IDs, without source messages, weights or secrets.

The first completed runs hit a local Modal deserialization failure on the `TorchVersion` metadata type. Their synchronous return buffers were unavailable for recovery, so the unchanged experiments were repeated to save complete artifacts. The repeated primary and natural counts match the initial console results. Repetitions are not treated as independent evidence. The runner now returns plain JSON and saves reports on the volume before RPC return. A separate app-creation rate limit rejected one retry before inference; it contributed no measurements. No teacher calls or spend-ledger changes were made.

Validation: source IDs and correctness were checked against the official MTOP
labels; accepted/error totals, exact-repeat identity, shared split/checkpoint
hashes, unchanged temperatures/thresholds and latency quantiles were checked
against the raw cards. `npm run check` passed with 202 tests, frontend lint and
build; lint reports two existing React effect warnings.

## Support floor for missing intents (Amendment 2)

Registered before inference in [protocol.md](protocol.md). The floor is fixed per shot count on
the 2,000 calibration certification rows with the gold intent removed: the lowest 0.01 step at
which at most 4 of 2,000 of those rows are still accepted. Temperature and threshold are unchanged.
Same checkpoint, same 500 test rows. Results: [attacks-floor.json](attacks-floor.json).

| Examples | Floor | Missing intent: wrong accepts / 500 | Clean accepted / 500 (errors) | Calibration clean accepts kept |
| ---: | ---: | --- | --- | --- |
| 0 | 0.84 | 4 → **0** | 115 (0) → 61 (0) | 232 / 451 |
| 8 | 0.80 | 9 → **1** | 189 (1) → 110 (0) | 401 / 706 |
| 32 | 0.83 | 9 → **1** | 167 (0) → 87 (0) | 330 / 640 |

Every other registered attack has zero accepted errors with the floor. The floor meets the
target of at most one wrong accept per 500 missing-intent requests and costs roughly 45% of
automatic answers; those requests go to the person instead.

## Model audit of calibration labels

Three language models replaced human labellers under Amendment 4 of the protocol. This is a model
audit, not a human audit. Full votes, changed labels and results are in `model-audit.json`.

- MTOP: 0 of 100 controls overturned. 12 of 26 confident disagreements changed label.
- BANKING77: 2 of 60 controls overturned, above the 2% rule, so that audit is not used and its
  labels and gate stay as they were.

The gate was refit on audited MTOP calibration labels and scored once from saved probabilities.
Precision is measured against unchanged test labels, which keep any errors they contain. The 400M
reader now certifies in every MTOP condition. Support floors were fitted before this audit and do
not apply to the refit gate until refit.
