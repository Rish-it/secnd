# README outline for the release

This is an outline. Keep the current run instructions until the full container
demo is verified; replace future-tense behavior only after it works.

1. **Problem.** Secnd learns recurring decisions and asks a person when its
   held-out evidence cannot support the precision target.
2. **Three measured claims.** Coverage at the precision target, its re-split
   evidence, and a live learning curve. Link each to the final release's raw
   result. Current reference: MTOP 31.68% automatic / 99.62% test precision /
   99.05% pass rate over 2,000 re-splits at 2,000 labels. These are benchmark
   results, not a promise for uploaded workspaces.
3. **Quickstart.** `docker compose up --build`; open localhost:8000. Include a
   real MTOP demo-workspace preparation command once merged. Say which steps
   download model/data files and what remains in the volume.
4. **One API flow.** Import a labelled CSV, predict, answer with an idempotency
   key, read the resulting version/metrics. Use commands verified against the
   shipped image; [the current API contract](../../api/README.md) documents the
   available behavior. Mark any remaining fixture route explicitly.
5. **How the gate works.** One calibrated classifier head; calibration and test
   excluded from training; a one-sided 95% lower bound for the 99% precision
   claim; fixed-sequence threshold selection; candidate refusal and rollback.
   Explain precision separately from coverage and the re-split pass rate.
6. **Where Meta models appear.** Muse Spark 1.3 through the Meta Model API:
   measured baseline and training-only paraphrases now; cold start, explanation
   and tool chat when connected. Muse Spark Open 1026 via Baseten is the tested
   fallback. Muse stays off Ren's automatic decision path.
7. **Measured comparison and costs.** Link the matched IPSA comparison and final
   CPU cost table. Keep datasets, label budgets, token rates, volumes, latency
   and hosting assumptions visible. Never compare MTOP accuracy to IPSA Muse
   accuracy as if they were the same test.
8. **Model release.** Link Ren (`secnd-ren-1`) and the exact `ren-1.<n>` artifact
   only after an approved upload. Include hashes, ordered labels, fixed encoder
   revision, runtime memory and measured card. Until then say “release pending”.
9. **Limits.** English only; one workspace process; uncertain traffic goes to
   people; IPSA abstains; the [BANKING77 card](../s3/card-banking77-minilm+lr-b2000.json)
   records 36.9% coverage at 99.03% precision, with a [98.95% re-split pass rate](../s3/robustness-banking77-minilm+lr-b2000.json). No 99% guarantee for arbitrary future traffic. Link
   [the model document](../../model.md).
10. **Build period and credits.** Review the new-work statement and release
    commit. Credit IPSA/OGL v3, BANKING77/CC BY 4.0, MTOP/CC BY-SA 4.0 and the
    pinned MiniLM encoder/Apache 2.0. State the actual release license once chosen.

Do not fill the cost, final learning curve, live route behavior or downloadable
weights with estimates. The release README should link measured artifacts from
its own revision, not depend on a draft pull request.
