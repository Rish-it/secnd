# Meta connector browser check

[Recorded browser assertions](browser.json) · [check results](checks.json) · [desktop](desktop.png) ·
[390px](mobile-390.png).

The app explicitly marks this rehearsal as a synthetic demo. It uses a deterministic
Ren test fixture and `httpx.MockTransport` for Ads. Incoming WhatsApp, Messenger and
Instagram payloads are signed synthetic fixtures posted to the local app. This is
an integration check, not a model evaluation or a live Meta account demonstration.
Displayed spend, CPA and CTR are fixture inputs, not measured campaign performance.

Run after `uv sync --locked` and `npm ci`:

```sh
npx playwright install chromium
npm run test:meta
```

The scenario verifies signed messages enter the shared queue, delivery replay is
deduplicated, Ads insights produce a typed proposal, auto-act persists after reload,
the server switch remains off, the action log contains no execution attempt, and
the 390px document has no horizontal overflow. The mocked transport refuses all
mutations. Neither the helper nor the browser can make a live Meta or provider call.

The backend connector tests separately cover execution with mocked transport:
certification and both switches, sandbox/account scoping, budget snapshots, action
types, missing CPA/budget, pagination, redaction, override cancellation, restart and
ambiguous-delivery replay protection. No benchmark accuracy is claimed.

The connector-to-inbox browser scenario runs when the decision inbox is available;
it skips on the standalone connector branch and checks all four source groups when
both app changes are present.

The [combined preview result](integration.json) records checks with both app patches
present, including the connector-to-inbox source-group scenario. Additional screenshots
show that preview's connector screen: [desktop](integration-desktop.png) and
[390px](integration-mobile-390.png). This preview does not merge either pull request.
