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

The recorded public-history result for the 400M model with eight examples and a
support floor is **13.89% coverage**, with **609 accepted answers, zero errors**
and a **99.40% lower confidence bound** across **4,386 fixed MTOP test rows**
(`docs/results/s3r/floor-full-test.json`). Calibration labels were audited by models,
not humans; test labels were unchanged. This operating point requires the full
fixed public history and does not establish performance on a new business.
The separate missing-intent audit includes a wrong acceptance at this operating
point (`docs/results/s3r/README.md`).

Measured 400M latency for the submitted product build: **TBD**.
Measured customer time saved or advertising lift: **TBD**.
Live Meta OAuth and delivery verification for this submission: **TBD**.
Public product URL: **TBD**. Video URL: **TBD**.

Run from the repository root after installing the locked dependencies:

```sh
npm run demo --workspace @secnd/frontend
```

Open `http://127.0.0.1:8000/?setup=1`. A fresh disposable database is created on every
launch. See `README.md` for installation, the synthetic story and keyboard controls.

Built for Meta's platform. Built with Muse Spark Open. Secnd is independently
developed and is not an official Meta product.
