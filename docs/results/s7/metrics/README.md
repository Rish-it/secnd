# Published evaluation view

The Metrics page uses the published MTOP, BANKING77 and IPSA cards and curves
from [the read API](https://github.com/Rish-it/secnd/pull/14). MTOP opens first.
These are fixed held-out evaluations with 2,000 training labels and seed 0,
separate from an uploaded workspace. The recorded frontend revision is
`34544ed`; the API revision is `52e7d87`.

[MTOP](mtop.jpg) shows 31.68% automatic coverage at 99.62% precision and a 99.05%
re-split pass rate. [BANKING77](banking77.jpg) exposes its 98.99% fixed-test
precision below the 99% target. [IPSA](ipsa.jpg) shows zero coverage and no
certified threshold. The charts draw the measured reliability bins and
coverage–precision points; a diagonal and the 99% target give their references.
Both include expandable tables. [Mobile](mtop-mobile.jpg) fits a 390px viewport.

The [browser record](browser-check.json) contains two desktop runs and one
mobile run against real API responses, checking all displayed dataset values,
chart counts, target/abstention states, active-selection refresh, data tables,
503 retry recovery and isolation of registry failures. No console errors or
warnings were observed. Learning history and the handoff remain unmeasured.
Live answer learning is not claimed by this view.

To repeat, serve the built frontend with the read API, open Metrics, and pass
the function in [browser-check.js](browser-check.js) to the browser's script
evaluator (or invoke that async function in DevTools). It checks actual HTTP
responses and briefly replaces `fetch` inside the test page for error states,
restoring it in `finally`. Screenshots retain the benchmark labels and the
empty learning/handoff panels.
