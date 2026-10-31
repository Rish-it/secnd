---
language:
  - en
license: cc-by-sa-4.0
library_name: sentence-transformers
pipeline_tag: text-classification
base_model: jhu-clsp/ettin-encoder-400m
tags:
  - ren
  - secnd
  - selective-classification
  - calibration
  - human-in-the-loop
---

# Ren

Built with Muse Spark Open. Ren is independently developed for Meta's platform and is not an official
Meta product.

Ren reads a question, its options, a message and a person's nearest past decisions, then scores each
option. It is trained once; a new business supplies history, not a new training run. A temperature, a
confidence threshold and a support floor come from held-out calibration rows, and a Clopper–Pearson lower
bound decides whether any answer may be automatic. Anything below it goes to a person. English only.

## What is in this repository

| Path, tag `v1.1.0` | What it is | Use it for |
| --- | --- | --- |
| `reader/` | 400M reader, `ic-ettin-400m-s6000-near-wide-fine-seed1` | Choosing among up to 128 options from a person's history |
| `meta/` | The reader above, fine-tuned on typed decisions and written Meta-app tasks | Ad, inbox and account decisions |
| root files | MiniLM encoder from [`v1.0.0`](https://huggingface.co/Secnd-AI/ren/blob/v1.0.0/README.md) | Retrieval vectors and the per-business classifier head |

```sh
hf download Secnd-AI/ren --revision v1.1.0 --include "reader/*" --local-dir ren
```

Serving code and setup: [github.com/Rish-it/secnd](https://github.com/Rish-it/secnd).

## Results

Fixed official test sets, measured on these exact weights. Calibration chose this seed; test sets chose nothing.

| Set | No examples | 8 nearest | 32 nearest |
| --- | ---: | ---: | ---: |
| MTOP | 75.99% | 92.38% | 93.59% |
| BANKING77 | 68.99% | 89.90% | 86.23% |

With a person's history, the reader against a logistic head on the same retrieval vectors:

| History | MTOP reader | MTOP head | BANKING77 reader | BANKING77 head |
| --- | ---: | ---: | ---: | ---: |
| 200 labels | 82.51% | 69.54% | 75.39% | 54.16% |
| 1,000 labels | 88.12% | 80.46% | 82.60% | 85.32% |
| All labels | 92.45% | 93.00% | 89.87% | 92.50% |

Answering alone, on all 4,386 MTOP test rows, with the gate and support floor fixed on calibration. Twelve
calibration labels were changed by a model audit, not a human one. Test labels were never changed, so
precision is measured against labels that keep their errors.

| Examples | Answered alone | Accepted | Errors | Precision | 95% low | Wrong accepts with the right option removed, of 500 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 0 | 0 | n/a | n/a | 0 |
| 8 | 21.04% | 923 | 1 | 99.89% | 99.40% | 2 |
| 32 | 14.46% | 634 | 1 | 99.84% | 99.12% | 2 |

`meta/` on 342 and 338 fresh written tasks: 84.8% agreement with a three-model panel and 76.0% with Muse Spark
1.3, against 44.4% and 49.7% for always picking the most common answer. Ad decisions are the weakest, at 70.3%
and 54.3%. This is agreement with models on written tasks, not accuracy on real data. On the fixed typed
benchmark of 2,000 decisions, before the Meta tuning, the same reader scores 79.70%.

One A10G over HTTP: 17.5 s cold start, then 3.68 requests a second with four clients, about $0.08 of GPU per
1,000 decisions when saturated. Four CPUs: 2.2 s p95 per request at 8 nearest, measured on the same architecture.

## Limits

- One seed. Certified coverage moved 19 points between two seeds of one earlier pool, and on this pool a second
  seed scores 0.55 points higher on MTOP and 0.72 on BANKING77 at 8 nearest. Report a range, never a single best run.
- The gate certifies full MTOP history only. Another business gets no inherited permission to answer alone; it
  earns it from its own reviewed decisions.
- With the right option missing, the support floor missed its target of at most 1 wrong accept per 500, with 2.
- BANKING77 certifies zero at the 99% bar, including after a model audit of its calibration labels.
- With no examples on BANKING77 this model does not lead: a published typed decision model scores higher.
- With thousands of labels, a logistic head on retrieval vectors beats the reader.
- Written training rows come from one model and carry its style.
- Benchmark numbers are not a promise about a new business or future traffic.

## Training data

Excluded: all MTOP and BANKING77 texts, calibration rows, test rows, real customer text, rows traced to
real posts, ATIS and a FEVER mirror (no licence found). No output of a hosted Meta Model API trained this model.

| Source | Licence |
| --- | --- |
| Ettin encoder 400M (base weights) | MIT |
| TOPv2 | CC BY-SA 4.0 |
| CLINC150 | CC BY 3.0 |
| MASSIVE (English) | CC BY 4.0 |
| Snips | Apache 2.0 |
| GoEmotions | Apache 2.0 |
| Bitext customer support | CDLA-Sharing 1.0 |
| DBpedia-14 | CC BY-SA 3.0 |
| Civil Comments | CC0 |
| PAWS, Google LLC | Free use, with attribution |
| SMS Spam (UCI) | CC BY 4.0 |
| SQuAD | CC BY-SA 4.0 |
| WANLI | CC BY 4.0 |
| Written Meta-app tasks and 23 taxonomies | Written and labelled with Muse Spark Open |
