# Three-minute demo draft

Rehearsal script, not a completed recording. Use a frozen release revision and
real API responses. Cold start, learning, rollback and cost segments are pending
live integration; do not simulate them or describe placeholders as measurements.

| Time | Screen and action | Spoken line / evidence |
| --- | --- | --- |
| 0:00–0:20 | Open the companion beside a recurring decision queue. | “Secnd learns a decision you repeat. It answers when the evidence supports 99% precision and asks you about the rest.” |
| 0:20–0:45 | Create a fresh MTOP workspace; show zero human labels. Run the Muse cold start after it is connected. | “Muse Spark 1.3, through Meta's API, supplies the first labels. They count at weight 0.3; human answers count at 1. These requests are visible and budgeted.” |
| 0:45–1:10 | Open the measured comparison and its dataset labels. | “On the same 500 IPSA test rows, Muse 1.3 scored 39.4%; the fallback scored 37.4%. This is a descriptive comparison. MTOP is a separate assistant-intent task.” |
| 1:10–1:55 | Answer real queued MTOP questions. Show background retraining and the measured Muse → Ren → human handoff. | “Each answer trains the classifier head. Calibration stays held out. The gate can refuse a version, so an answer need not increase coverage.” Show actual counters, not a scripted increase. |
| 1:55–2:20 | Open MTOP Metrics, then briefly BANKING77 and IPSA. | “The recorded MTOP run answers 31.68% automatically at 99.62% test precision. The claim holds in 99.05% of 2,000 re-splits. BANKING77 answers 36.9% automatically at 99.03% test precision, and its claim holds in 98.95% of re-splits. IPSA certifies nothing at this label budget.” Show the diagonal, 99% line and bound. |
| 2:20–2:35 | Restore an earlier real model, then return to the chosen version. | Show the registry pointer and a subsequent decision carrying the restored version. Do not claim rollback works from a pointer change alone. |
| 2:35–2:50 | Open the final measured cost table and its assumptions; one short explanation or chat if live. | “Muse helps teach and explain. Ren handles repeat decisions on CPU.” Give the verified per-1,000 costs; leave this numerical line unwritten until the CPU speed/cost report is merged. |
| 2:50–3:00 | Show the release revision, model card and links. | “This demonstration covers the decision model, API and companion built for the hackathon. The source and reproducible evaluation are linked.” |

The reference MTOP card is [here](../s3/card-mtop-minilm+lr-b2000.json),
[BANKING77 here](../s3/card-banking77-minilm+lr-b2000.json) with its
[2,000 re-splits](../s3/robustness-banking77-minilm+lr-b2000.json), and
[IPSA here](../s3/card-ipsa-minilm+lr-b2000.json). The paired Muse run is
[here](../s5/provider-comparison.json). These references are not a claim that a
freshly trained demo workspace has the same metrics.

Before recording: complete live integration; regenerate final measured results
from the release; run this exact sequence twice against the container; time it
at or under 180 seconds. Verify zero-label state, answer persistence, version
refusal, real rollback predictions and cost provenance. Record real failures
honestly or choose a measured, repeatable demo run in advance.
