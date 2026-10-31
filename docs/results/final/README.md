# Docker rehearsal evidence

[Technical walkthroughs](rehearsal.json) ran twice against the built image with
the pinned CPU encoder: import, predict, queue, durable answer/replay, background
fitting, history/models and rollback, plus signed WhatsApp and Messenger delivery,
idempotence and human answers. Both README curl walkthroughs passed. Offline
encoder verification/loading/encoding passed as the unprivileged container user.

The image measured 804,590,446 bytes. Desktop and 390px browser checks showed the
released encoder reference without overflow or console errors. Full checks passed
twice (196 tests), then from a clean committed revision. Container code hashes
match the committed source files. No GPU job, merge, deployment or real outbound
message was part of this rehearsal.

This rehearsal predates the harness. The current product demo, video script and
verification notes are in [docs/submission](../../submission/submission.md). Recording
and final submission remain human steps.
