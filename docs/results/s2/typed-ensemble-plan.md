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

## Temperature target (added after the result above; a second operating point, not a new model)

The temperatures above minimise each type's NLL of the gold label on validation. They come out
near 0.4, which sharpens the probabilities toward that one label: good for ECE, which scores the
top probability against correctness, but Brier and soft accuracy score the whole distribution
against the benchmark's teacher distributions, and a sharpened distribution sits far from them.

Fitting the same three temperatures to the cross-entropy against the teacher distributions on the
same validation decisions (`python -m backend.typed --fit teacher --ensemble ...`) gives
temperatures near 1.0, so the members' training objective already matches the teacher. Same
members, same probabilities, test scored once:

| Temperatures fit to | Test accuracy | Brier | Soft accuracy | ECE |
|---|---|---|---|---|
| gold label (above) | 0.8025 | 0.098 | 0.636 | **0.019** |
| teacher distribution | 0.8025 | **0.042** | 0.573 | 0.169 |

This choice was made after the first result showed the Brier gap, so it is reported as a second
operating point next to the pre-registered one, never in place of it. Accuracy cannot change
(one temperature per type keeps every top option). One distribution cannot be best at both
targets: matching the teacher's spread raises ECE against correctness, and sharpening raises
Brier against the teacher.
