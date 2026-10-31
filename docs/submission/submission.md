# Secnd

Track: Indie developer

Secnd runs a person's Meta business with them. Connect business accounts, choose
what Ren may watch, and set the evidence bar for acting. The home screen is a
decision inbox spanning Ads, WhatsApp, Messenger and Instagram: the situation,
Ren's proposed action, confidence, nearest past decisions, and one-tap review.
Corrections and undo keep the person in control.

Ren is our 400M decision model, trained once and reading the person's past choices
as examples (`docs/results/s3r/README.md`). Muse Spark Open supports the project's
language work and never decides. Permissions are separate for Ads, Inbox and
Account. Watch proposes; Ask waits for confirmation; Act is available only when
the relevant evidence and permission checks are satisfied. A correction can lower
the bound and remove Act. Sensitive actions require additional permission;
account-management writes are not implemented.

The product makes the evidence visible. Reviews from the person's business are
labelled as personal reviews. Certification from audited public history is labelled
separately. Neither substitutes for the other. Activity shows the agent log,
observed ad results after seven days and a stop switch with an inline confirmation.
An observation is not a causal claim.

The local submission demo is a fictional, visibly synthetic business. It demonstrates
setup, review, correction, earned permission, a permission drop, acted decisions,
observations and stopping. Proposals, scores and outcomes are scripted; the 400M
checkpoint is not loaded. Reviews and undo use the local SQLite database.
The running identifier is shown as `synthetic-scripted-demo` (`README.md`).
No paid calls, live OAuth, message delivery or ad changes are part of that demo.

The recorded public-history result for the released 400M reader with eight examples and a
support floor is **21.04% coverage**, with **923 accepted answers, 1 error**
and a **99.40% lower confidence bound** across **4,386 fixed MTOP test rows**
(`docs/results/s3r/floor-full-test-400m-fine-seed1.json`). Calibration labels were audited by models,
not humans; test labels were unchanged. This operating point requires the full
fixed public history and does not establish performance on a new business.
The separate missing-intent audit accepted 2 wrong answers per 500 at this operating
point, above its target of 1 (`docs/results/s3r/README.md`).

These weights trained on licence-checked data and are public at
[https://huggingface.co/Secnd-AI/ren/tree/v1.1.0](https://huggingface.co/Secnd-AI/ren/tree/v1.1.0). They measured **92.38%** on MTOP and **89.90%**
on BANKING77 with eight examples (`docs/results/s2/wide-round.md`). BANKING77 certifies no
automatic coverage at the 99% bar.

Measured 400M reader over HTTP on one A10G, written tasks: 17.5 s cold start, 3.68 requests a
second with four clients, 441 ms p95 for one client, from India to us-east
(`docs/results/s8/load-gpu-secnd-ren-1.json`). The synthetic demo does not call it.
Measured customer time saved or advertising lift: **TBD**.
Live Meta sign-in ran on 2026-10-31: it connected, listed and saved one ad account, and read its
insights, which were empty because the account has no campaigns (`docs/submission/verification.md`).
Live message delivery was not tested.
Public product URL: **TBD**. Video URL: **TBD**.

Run from the repository root after installing the locked dependencies:

```sh
npm run demo --workspace @secnd/frontend
```

Open `http://127.0.0.1:8000/?setup=1`. A fresh disposable database is created on every
launch. See `README.md` for installation, the synthetic story and keyboard controls.

Built for Meta's platform. Built with Muse Spark Open. Secnd is independently
developed and is not an official Meta product.
