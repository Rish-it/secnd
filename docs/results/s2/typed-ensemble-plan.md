# typed-decisions ensemble (fixed before its members ran)

Single runs, already reported, sit below the strongest published small model (0.796), and their
validation spread (600 decisions) is too wide to choose a config. So the final score is fixed now:

- members: the in-context 150M reader, seeds 0, 1, 2, and Ettin-400M, seeds 0, 1; 8 epochs
  each, best epoch by validation accuracy, as in every single run;
- the mean of the members' probabilities, one temperature per type fitted on validation;
- test scored once. Every member counts; none is dropped on any result.

Members are retrained to save their probabilities, so their single-run numbers replace the
earlier files (GPU training is not bit-for-bit repeatable).
