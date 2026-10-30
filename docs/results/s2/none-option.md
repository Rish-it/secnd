# A "none of these" option for the reader (tried, not adopted)

The red-team audit found that when a message's intent is missing from the option list, the gate
still accepts a few answers, all wrong (`docs/results/s3r/README.md`). This tried a reader that
can say so: one extra option, `none of these`, after the labels. In a tenth of training episodes
the right label and every example carrying it are left out and `none of these` is the answer.
Everything else matches the 150M in-context reader (6,000 steps, same data and seed). The extra
column takes part in the temperature but never answers.

| MTOP, all history, 32 nearest | Accuracy | Certified at 99% |
|---|---|---|
| reader without the option | 93.2% | 32.4% |
| reader with the option | 91.4% | 0% |

The option costs about two points of accuracy and, on the certify half of calib, one confident
wrong answer at rank 275 (the gate needs 427 answers without error at the top), so the gate
certifies nothing at any history size or set; the frozen attack audit then accepts nothing in any
condition (`docs/results/s3r/attacks-none.json`). That is no evidence of detecting a missing
intent. The served reader is unchanged and the missing-intent limit stays documented.

Results: `incontext-ic-ettin-150m-s6000-near-none.json` (training run),
`incontext-ic-ettin-150m-s6000-near-none-clean.json` (history budgets).
