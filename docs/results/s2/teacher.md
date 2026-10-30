# Meta's models as Ren's teachers

A new business has messages but no labels. Here Meta's models supply the labels (or invent the
history outright), and Ren learns from that with no human label on its history. Calib keeps its
gold labels: one half fits the temperature, the other certifies the 99% gate, exactly as for
every other Ren result. Test is scored once. Only history rows (official train minus calib) are
ever sent to a model; invented examples send only the label names.

Code: `backend/teacher.py`. Spend is in `docs/results/s5/spend.csv` (Baseten) and
`spend-meta.csv` (Meta API), tasks `s2-teach-*`.

## Pilot: which teacher (100 history rows per set, gold known)

| Teacher | MTOP | BANKING77 | cost per row |
|---|---|---|---|
| Muse Wonder 1.0 (Meta API) | 93% | 80% | about $0.005 |
| Muse Spark 1.3 (Meta API) | 92% | 78% | about $0.002 |
| Muse Spark open weights (Baseten) | 91% | 81% | about $0.002 |

On 100 rows the differences are within sampling error (about 3 points). BANKING77's gold labels
put every teacher near 80%. On MTOP rows where Spark reported confidence below 0.9 (a fifth of
them), Wonder was right more often, so the run below is a cascade: Spark labels every row and
Wonder relabels the uncertain ones. BANKING77's pilot showed no such gain, so it was not run there.

## Teacher-labelled history: 3,000 rows, no human label

Spark: open weights on Baseten for the first 2,073 MTOP rows, Spark 1.3 on the Meta API for the
rest and for BANKING77. Wonder relabelled the 583 MTOP rows Spark gave below 0.9: 70.5% of those
right against Spark's 66.0%, lifting the history from 87.7% to 88.5% correct.

| Set | History labels | Reader, 32 nearest: accuracy | certified at 99% | precision of certified |
|---|---|---|---|---|
| MTOP | teachers (88.5% correct) | 87.7% | **42.5%** | 99.7% |
| MTOP | gold, same 3,000 rows | 91.6% | 51.9% | 99.65% |
| MTOP | gold, 200 rows (earlier result) | 85.8% | 25.5% | |
| BANKING77 | teacher (80.4% correct) | 76.0% | 0% | |
| BANKING77 | gold, same 3,000 rows | 81.1% | 0% | |

With 8 nearest examples MTOP certifies 29.2% (teachers) and 27.7% (gold). The logistic head
never certifies on these histories. On BANKING77 the head is the stronger model (80.2% from the
teacher's labels, 90.4% from gold) and still certifies nothing: BANKING77's calibration problem is
separate from its labels.

## Invented history: 10 messages per label from the label names alone

Muse Wonder wrote 10 messages for every label, seeing only the list of label names: 1,130 for
MTOP ($1.04) and 770 for BANKING77 ($0.84). No real record was labelled or sent.

| Set | History | Reader, 8 nearest | Reader, 32 nearest | Head | certified (reader-32) |
|---|---|---|---|---|---|
| MTOP | none | 80.9% (0-shot) | | | 23.4% |
| MTOP | invented | 84.2% | 84.95% | 64.7% | 25.6% |
| MTOP | 200 gold (earlier) | 85.9% | 85.8% | 69.5% | 25.5% |
| BANKING77 | none | 61.4% (0-shot) | | | 0% |
| BANKING77 | invented | 68.2% | 66.6% | 73.6% | 0% |
| BANKING77 | 200 gold (earlier) | 68.5% | 68.0% | 54.2% | 0% |

Invented messages are worth about 200 human labels to the reader, and more than that to the head
on BANKING77, before the business has labelled anything.

## Spend this round and one incident

- Meta API: $14.30 (pilots $1.34, invented examples $1.88, Spark 1.3 $7.56, Wonder cascade $3.52).
- Baseten: one run at 8 parallel calls was rate-limited; 1,927 of 2,221 calls failed. The ledger
  books every failed call at its full reservation, so it records $16.40 for that run although
  only 294 rows were answered (about $0.55 at the measured price per answered row; what the
  rejected calls cost is unknown here, and the provider's bill is the reference). An earlier run stopped by hand was reconciled from its cached per-row usage
  ($3.22). Paid runs now stop by themselves once failures outnumber answers.
