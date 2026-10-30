# typed-decisions ensemble (fixed before its members ran)

Single runs, already reported, sit below the strongest published small model (0.796), and their
validation spread (600 decisions) is too wide to choose a config. So the final score is fixed now:

- members: the in-context 150M reader, seeds 0, 1, 2, and Ettin-400M, seeds 0, 1; 8 epochs
  each, best epoch by validation accuracy, as in every single run;
- the mean of the members' probabilities, one temperature per type fitted on validation;
- test scored once. Every member counts; none is dropped on any result.

Members are retrained to save their probabilities, so their single-run numbers replace the
earlier files (GPU training is not bit-for-bit repeatable).

## Result (added after the run; nothing above was changed)

| Run | Validation | Test accuracy | Brier | ECE |
|---|---|---|---|---|
| in-context 150M, seed 0 | 0.808 | 0.7865 | 0.108 | 0.031 |
| in-context 150M, seed 1 | 0.800 | 0.7840 | 0.092 | 0.022 |
| in-context 150M, seed 2 | 0.805 | 0.7965 | 0.105 | 0.020 |
| Ettin-400M, seed 0 | 0.803 | 0.7965 | 0.099 | 0.019 |
| Ettin-400M, seed 1 | 0.813 | 0.7950 | 0.081 | 0.026 |
| **ensemble of all five** | 0.818 | **0.8025** | 0.098 | 0.019 |

Ensemble per type: choice 0.768, yes/no 0.872, score 0.776; soft accuracy 0.636.
Above the strongest published small model (0.796) by 0.65 points, within one standard error of
2,000 decisions (about 0.9 points, before accounting for decisions shared by a case). Brier,
scored against the benchmark's teacher distributions, is not the best published.
