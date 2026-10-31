# Secnd workspace setup

Secnd is independently built for Meta’s platform. It is not an official Meta product.
Ren makes decisions. Customer messages, ad changes and background processing default off.

## Start and sign in

Install with `uv sync --locked` and `npm ci`, then `npm run dev`.
For local HTTP development only, set `SECND_SESSION_COOKIE_SECURE=false`.
Use HTTPS and secure cookies for a live installation. Run one API process.
Create a local account, create a workspace, then open Connections → Connect Meta.
The app uses opaque, expiring, HttpOnly sessions and salted password hashes.
Workspace ownership is enforced for reads, imports, answers, exports, models and integrations.
Previously unowned workspaces stay available only in explicit demo mode; they cannot be claimed
through registration.

Use Teach to import the person's past labelled decisions before enabling background work.
The existing fixed calibration holdout and model promotion checks remain in place.
Use a separate Ads workspace with the four labels `pause`, `keep`, `raise budget`, `lower budget`.
Message workspaces use the person's approved short response templates as labels. This version
sends the selected template verbatim; it does not generate freeform customer replies.

## Facebook Login for Business

Create a Business login configuration that returns a **user access token**. Configure the
permissions below in that configuration. Before Advanced Access, only people with a role
on this Meta app can connect. The connection screen explains refusals and role requirements.

Required live environment settings (replace placeholders privately):

```sh
SECND_DEMO_MODE=false
SECND_SESSION_COOKIE_SECURE=true
SECND_META_APP_ID=<app-id>
SECND_META_LOGIN_CONFIG_ID=<business-login-configuration-id>
SECND_META_APP_SECRET=<server-only-app-secret>
SECND_META_REDIRECT_URI=https://YOUR_HOST/api/v1/meta/oauth/callback
SECND_META_PUBLIC_URL=https://YOUR_HOST
SECND_META_TOKEN_KEY=<Fernet-key>
SECND_META_VERIFY_TOKEN=<random-webhook-verification-token>
SECND_META_GRAPH_VERSION=v25.0
SECND_ADS_SANDBOX_ACCOUNTS='["act_YOUR_SANDBOX_ID"]'
SECND_ADS_SEND_ENABLED=false
SECND_WHATSAPP_SEND_ENABLED=false
SECND_META_MESSAGE_SEND_ENABLED=false
SECND_COLD_START_PAID=false
```

`SECND_META_TOKEN_KEY` must be a URL-safe base64-encoded 32-byte Fernet key. Generate and
store it privately. Keep the same key across restarts; losing it requires reconnecting.
Do not put tokens or the key into frontend configuration. Short-lived and long-lived token
exchanges run on the server. Tokens, including Page tokens, are encrypted at rest.
Disconnect deletes stored credentials and all of that person's account mappings.

These scopes live in one Settings list. The default is:

```json
["ads_read", "ads_management", "business_management", "pages_show_list", "pages_messaging", "instagram_basic", "instagram_manage_messages", "whatsapp_business_management", "whatsapp_business_messaging"]
```

An optional `SECND_META_SCOPES` JSON override changes that one list. Keep the Business login
configuration synchronized with it. Add **exactly** the configured redirect URI to Meta's
Valid OAuth Redirect URIs. Set the data deletion request callback to
`https://YOUR_HOST/api/v1/meta/data-deletion`. Its signed request removes credentials and the
person's derived workspace data, history and local model artifacts, even after disconnect.
Its returned confirmation URL reports completion without exposing personal identifiers.

Set the shared webhook callback to `https://YOUR_HOST/api/v1/meta/webhook` with the configured
verification token. Subscribe the app and its selected assets to the relevant Meta messaging
webhook fields through Meta's setup tools. Asset subscription provisioning is manual in this
version; selecting an asset in Secnd maps delivery but does not grant missing Meta permissions.
The callback verifies raw-body signatures and dispatches by the receiving account's selected
workspace. Legacy workspace-specific callbacks remain for existing fixture integrations.
These machine callbacks use Meta signatures instead of a browser session.

After login, the person picks their ad account, Pages, linked Instagram accounts and WhatsApp
numbers. Only operator-allowed sandbox ad accounts can be selected. Signed-in workspaces never
use the old server-setting account mappings or shared Ads/WhatsApp tokens. Credentials are
never returned by API responses. Tests use a mockable transport exclusively.

## Background work and permission

Enable the worker process setting:

```sh
SECND_AGENT_ENABLED=true
SECND_AGENT_POLL_SECONDS=30
SECND_AGENT_INSIGHTS_SECONDS=3600
SECND_AGENT_RESET_KILL_ON_START=false
```

Then turn on each desired workspace in Activity. Both the server setting and the workspace
preference are needed for scheduled work. One workspace failure does not stop the next.
Signed webhook messages persist while a workspace is paused and resume when enabled.
Activity shows steps and observed outcomes. The global stop is durable across restarts and
blocks new sends and ad execution. To reset it deliberately, restart once with
`SECND_AGENT_RESET_KILL_ON_START=true`, then return that setting to false.
Already submitted provider requests may finish after a stop.

Watch proposes only. Ask accepts a person's confirm or correction. Act requires the selected
workspace and decision-type evidence bar as well as the existing model certification.
The review window is the latest 1,000 distinct reviewed decisions at or above the chosen
confidence cut. A repeated answer to the same item does not create extra evidence.
The UI shows actual reviewed/confirmed/corrected counts, the one-sided confidence lower bound,
and the additional consecutive confirmations needed, assuming no corrections.
Targets are labelled as targets. Aggregate opt-in counts never grant permission to act.

For automatic budget increases or customer messages, the workspace must select the highest
bar, meet it, and enable the specific action. Each real Ads change also requires
`SECND_ADS_SEND_ENABLED=true` and the workspace Ads auto-act preference for automatic execution.
Real WhatsApp delivery requires `SECND_WHATSAPP_SEND_ENABLED=true`. Messenger and Instagram
delivery require `SECND_META_MESSAGE_SEND_ENABLED=true`. Leave these false for a read-only
hackathon run. Explicit human confirmations in Ask may approve an action without an automatic
model certificate, but still require the relevant send switch and selected account.

The existing random spot-checks and guard remain active. Undoing an answer preserves guard
evidence. Answer undo changes history; it cannot retract a delivered message. Supported Ads
changes store their previous values and offer a separate undo, refusing restoration if the
target has changed in the meantime. Ambiguous send or undo results are never retried blindly.
No Meta account-management writes are implemented; those decisions remain workspace labels.

## Observed outcomes and learning

Each ad proposal saves the before metrics and is scheduled for observation seven days after
creation. Fetches use the same transport and sandbox mapping. Results contain spend, CPA and
CTR for the completed seven-day window, with missing values left unavailable. Pending proposals
also receive observations, with their execution status preserved. This is an observed result,
not a claim that the proposal caused a change.

Outcomes create an inbox review that includes the proposed action and execution status.
Only the person's reviewed label enters training. Hindsight reviews do not count toward the
before-action evidence used to unlock Act. Confirmations and corrections use the same
durable history and retraining path as existing decisions. The existing model promotion checks,
rollback, fixed evaluation split and reader exact-history restriction remain unchanged.
The latter restriction is deliberately retained; the workspace bound adds a requirement and
does not authorize a new model scope.

The authenticated workspace `outcomes.csv` route exports observed results joined to the latest
active review label. No generated success labels are invented from numerical changes.
The aggregate track record shares only opted-in decision-type counts and workspace counts.
It contains no message content, names or identifiers and is not a model training input.

## Synthetic demo and verification

The existing seeded synthetic inbox demo provides two fictional workspaces and local CPU
inference. Start it with `uv run python -m backend.tests.inbox_demo --port 8000` after building
the frontend. The synthetic connector demo uses mocked transport. Demo mode is visibly marked,
its fallback session cannot open signed-in workspaces, and the app's default demo transport
cannot call Meta. Demo flags and observations are not customer measurements.

The [measured verification record](../results/c4-harness/result.json) contains the checks and limitations. Browser execution and
live OAuth remain separate checks; no claim of live Meta verification is made here.
