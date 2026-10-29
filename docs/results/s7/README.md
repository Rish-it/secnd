# Companion workspace preview

The web workspace uses the committed API contract. Teach previews a CSV, chooses
the target column and sends it to the import endpoint. The floating companion
opens a keyboard-accessible panel with questions, other-category answers,
automatic-decision evidence, corrections and Muse chat. Metrics displays
coverage, precision, calibration, model versions and the evidence gate.

The current API returns fixtures. Sample decisions and chat are labelled as
previews; answers are explicitly not saved. A zero-coverage model says
**“Answers nothing yet.”** Unknown precision, cost and training time remain
unmeasured. Reliability and the Muse → Ren handoff have empty states until
measured data is available. No learning curve or coverage increase is invented.

[Browser record](browser-check.json): import/target selection, two runs of ten
fixture answers, unchanged zero coverage, correction, decision evidence, chat,
Enter/Escape/focus restoration, 390-pixel mobile layout, and a simulated
connection failure followed by successful retry. No console warnings or errors
were observed. This checks the preview; the live learning gate remains pending.

Screenshots: [Teach](teach.jpg), [Companion](companion.jpg),
[Metrics](metrics.jpg), [mobile companion](companion-mobile.jpg).
