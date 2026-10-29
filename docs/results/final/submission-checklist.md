# Submission checklist draft

Target: Meta Global AI Developer Hackathon, Indie track. Submit early on Nov 4.
Deadline: **Nov 4, 2026, 9:00 PM ET = Nov 5, 2026, 7:30 AM IST**.

- [ ] Freeze the submitted `secnd-bootstrap` revision: **RELEASE_TODO**.
- [ ] Confirm all shown work and assets fall within the official building period
      (Oct 27–Nov 4) and accurately state the team's actual development dates.
- [ ] Review the new-work statement: “This entry demonstrates the Secnd decision
      model, evaluation, API and companion developed for this hackathon. The older
      waitlist site is excluded from the demonstration.” Link the
      [bootstrap starting point](https://github.com/Rish-it/secnd/commit/a69d063)
      and final submitted revision; verify the links and dates before submitting.
- [ ] Public repository: https://github.com/Rish-it/secnd, with the submitted
      branch/commit clearly linked. The judge's quickstart runs twice from a clean
      clone; all displayed numbers link to results from that revision.
- [ ] Public live-product URL: **RELEASE_TODO**. Test without the developer's session.
- [ ] Public video URL: **RELEASE_TODO**. Maximum 3 minutes; show real cold start,
      answers, the precision gate, handoff, rollback and sourced cost comparison.
- [ ] Project name, short description, team names, roles and contact information
      match the submission form. List every participant.
- [ ] Model-use statement is accurate: Muse Spark 1.3 via the Meta Model API for
      baseline/paraphrase work and connected cold start/explanations/chat; Muse
      Spark Open 1026 via Baseten as the tested fallback. Distinguish implemented
      clients from features actually exercised in the recorded product.
- [ ] Ren model URL: **RELEASE_TODO**, model id `secnd-ren-1`, exact `ren-1.<n>`.
      Do not call weights public until the approved upload exists. Ren's current
      MiniLM encoder is not itself a Meta model. Describe Meta's models separately.
- [ ] Review the open-weight prize flag against the official eligibility rules
      and actual Meta model usage; record the relevant model/provider accurately.
- [ ] Dataset and encoder credits, artifact rights and the selected release license
      are in the README/model card. MTOP archive includes CC BY-SA 4.0; MiniLM is
      Apache 2.0. Keep the applicable notices with redistributed assets.
- [ ] Approve the model card and model files before any Hub upload. No raw uploads,
      training caches, personal workspace data or credentials in the release bundle.
- [ ] X demo-post URL: **RELEASE_TODO**. A person publishes it and adds the link
      for the viral award; the recorded engagement cutoff is Nov 8.
- [ ] Gallery/community-vote URL saved after submission: **RELEASE_TODO**.
- [ ] A person reviews the final form, records the video and presses submit.
      Save the receipt and submission time.

Pending live integration and final evidence: cold-start/handoff run, candidate
refusal, rollback affecting predictions, speed/cost table, final container demo,
and the approved downloadable model. This checklist does not mark them complete.
