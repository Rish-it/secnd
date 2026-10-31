---
language:
  - en
license: cc-by-sa-4.0
pipeline_tag: text-classification
base_model: jhu-clsp/ettin-encoder-400m
tags:
  - selective-classification
  - calibration
  - secnd
---

# Ren, 400M reader (DRAFT)

**Release status: RELEASE_TODO. No downloadable reader is claimed by this draft.**
The public encoder at this repository's `v1.0.0` tag is a separate artifact.

Built with Muse Spark Open. Ren is independently developed for Meta's platform and is not an official
Meta product.

Ren reads a question, its options, a message and a person's nearest past decisions, then scores each
option. It is trained once; a new business supplies history, not a new training run. A temperature and
threshold come from held-out calibration rows, and a Clopper–Pearson lower bound decides whether any
answer may be automatic. Anything below it goes to a person. English only.

## Results for the current candidate

Fixed official test sets, seed 1 of two, chosen on calibration, raw labels. **RELEASE_TODO: replace with results for the
exact uploaded weights, across seeds.** [Full tables](https://github.com/Rish-it/secnd/blob/main/docs/results/s2/wide-round.md).

| Set | No examples | 8 nearest | 32 nearest | Certified coverage at 8 |
| --- | ---: | ---: | ---: | ---: |
| MTOP | 75.99% | 92.38% | 93.59% | 33.22%, 99.93% accepted precision |
| BANKING77 | 68.99% | 89.90% | 86.23% | 0% |

## Limits

- Coverage moved 19 points between two seeds of one pool. Report a range, never a single best run.
- With the correct intent missing from the options, the gate can accept wrong answers.
- BANKING77 certifies zero at the 99% bar, even after a model audit. With no examples on BANKING77 this model has no lead.
- With thousands of labels, a logistic head on retrieval vectors beats the reader.
- Written training rows come from one model and carry its style.
- Benchmark numbers are not a promise about a new business or future traffic.

## Training data

Excluded: all MTOP and BANKING77 texts, calibration rows, test rows, real customer text, rows traced to
real posts, ATIS and the FEVER mirror (no licence found).

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
| PAWS | Free use, with attribution |
| SMS Spam (UCI) | CC BY 4.0 |
| SQuAD | CC BY-SA 4.0 |
| WANLI | CC BY 4.0 |
| Written Meta-app tasks and 23 taxonomies | Written and labelled with Muse Spark Open |

**RELEASE_TODO:** confirm each licence notice against the source, and that CC BY-SA 4.0 suits every
source before upload.
