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
["ads_read", "ads_management", "business_management", "pages_show_list", "pages_manage_metadata", "pages_messaging", "instagram_basic", "instagram_manage_messages", "whatsapp_business_management", "whatsapp_business_messaging"]
```

An optional `SECND_META_SCOPES` JSON override changes that one list. Keep the Business login
configuration synchronized with it. Add **exactly** the configured redirect URI to Meta's
Valid OAuth Redirect URIs. Set the data deletion request callback to
`https://YOUR_HOST/api/v1/meta/data-deletion`. Its signed request removes credentials and the
person's derived workspace data, history and local model artifacts, even after disconnect.
Its returned confirmation URL reports completion without exposing personal identifiers.

Set the shared webhook callback to `https://YOUR_HOST/api/v1/meta/webhook` with the configured
verification token, and configure the relevant app-level webhook objects in Meta's setup tools.
Saving selected assets attempts asset-level subscriptions through the connector transport:
Pages use `/{page}/subscribed_apps` with messaging fields; linked Instagram accounts use their
Page; WhatsApp numbers use `/{waba}/subscribed_apps`. The response and connection status include
`subscriptions`, with a status for each asset. A failed subscription preserves the selection;
saving again retries it. Reconnecting first attempts to remove the old subscriptions, then
requires selecting assets again. Disconnect attempts to unsubscribe current and previously selected
assets before erasing local credentials, even if an unsubscribe fails. Missing permissions still
require reconnecting or operator setup. Older saved inventories without Page/WABA parent IDs
need reconnecting before those subscriptions can be provisioned.
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

Watch proposes only. Ask accepts a person's confirm or correction. For a signed-in owned workspace, Act can use the person's selected workspace and decision-type
evidence bar independently of global model certification and the reader's exact-history rule.
A decision must meet the chosen confidence cut and the current bound. Demo and unowned
workspaces retain the existing certification and exact-history requirements.
The review window is the latest 1,000 distinct reviewed decisions at or above the chosen
confidence cut. A repeated answer to the same item does not create extra evidence.
The UI shows actual reviewed/confirmed/corrected counts, the one-sided confidence lower bound,
and the additional consecutive confirmations needed, assuming no corrections.
Evidence is isolated by owner, workspace, decision type, model id and model revision. A new
model receives no evidence from the previous model. Reader corrections change context, not
weights, so they do not reset the reader's model fingerprint. Legacy reviews without model and
owner provenance do not unlock owned workspaces. Targets are labelled as targets. Aggregate
opt-in counts never grant permission to act.

Every automatic decision includes `automatic_basis` (`workspace_bound` or `model_certified`),
`automatic_evidence`, `model_id` and `model_revision`. Workspace evidence includes reviewed and
confirmed counts, its one-sided 95% lower bound, selected bar and confidence cut. Its
`cp_lower_bound` is empty: it must never be displayed as the public 99% benchmark claim.
Execution rechecks permission, model identity, guard, confidence and applicable switches.
Random spot-checks still enter the person's queue; their failures can pause automatic decisions,
and undo does not erase that guard evidence.

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
rollback and fixed evaluation split remain unchanged. Model-certified reader decisions retain
the exact-history restriction. Only owned workspace-bound decisions may use the separate
review evidence described above; this does not extend the public benchmark claim.

The authenticated workspace `outcomes.csv` route exports observed results joined to the latest
active review label. No generated success labels are invented from numerical changes.
The aggregate track record shares only opted-in decision-type counts and workspace counts.
It contains no message content, names or identifiers and is not a model training input.

## Optional Meta decision reader

The existing serving behavior remains the default. To route ad, inbox and account decisions
through a saved local reader, set both `SECND_META_DECISION_MODEL_FOLDER` (a folder containing
`encoder/` and `scorer.pt`) and `SECND_META_DECISION_MODEL_ID` (the release name). Files are loaded
locally on CPU; missing or invalid configured files fail startup. No weights are downloaded.
The revision fingerprints every saved file. Treat the folder as immutable while the app runs;
restart after replacing a release. This reader uses the existing model loader and typed
renderer with the decision's own option keys and meanings. It retrieves up to eight nearest
workspace examples and excludes the fixed calibration holdout. Labels and corrections enter
context immediately without retraining. Inbox options are approved response templates; account
options are workspace labels and still have no account-management execution path. The saved
reader has no inherited global certificate. Only the owner's earned bound can authorize it.

For HTTP reader inference instead, configure `SECND_READER_ENDPOINT_URL`,
`SECND_READER_ENDPOINT_TOKEN`, `SECND_META_DECISION_MODEL_ID`, and optionally
`SECND_READER_ENDPOINT_REVISION` to pin the remote fingerprint. The URL takes precedence over
the local folder. Use an HTTPS URL; redirects are refused. The token is server-only. Timeout is
`SECND_READER_ENDPOINT_TIMEOUT_SECONDS` (default 10 seconds). Timeout, HTTP failure, invalid
probabilities or a mismatched model/revision produce a queued question with `inference_error`;
there is no automatic fallback. An unpinned successful remote revision updates the active
identity and therefore requires its own review evidence. Requests send only the current
workspace's decision and up to its latest 1,000 active history rows, excluding calibration.
The endpoint performs retrieval to choose at most eight examples. Do not point it at an
untrusted service.

`backend/modal_serve.py` defines an optional POST endpoint using an A10G, one container, one
input at a time, zero minimum containers and a 60-second idle scale-down window. It mounts the
existing `secnd-bench` volume at `/vol`. A Modal secret named `secnd-reader` supplies
`SECND_READER_MODEL_PATH` (an absolute path under `/vol`), `SECND_READER_MODEL_ID`, and
`SECND_READER_BEARER_TOKEN`. The volume must already contain the reader and pinned local
MiniLM retrieval cache under `/vol/hf`. The endpoint requires `Authorization: Bearer ...` and
returns model id, fingerprint and option probabilities. Its JSON request contains `item_id`,
`text`, `decision_type` (`ad`, `inbox`, `account`), `keys`, `meanings`, and `history` rows with
`id`, `text`, `label`. It never executes an action. The optional Modal SDK is an operator tool,
not an app runtime dependency. No deployment or GPU serving measurement was performed here.
See Modal's [web application](https://modal.com/docs/guide/webhooks) and
[scaling](https://modal.com/docs/guide/scale) configuration.

`backend/load_test.py` is an opt-in HTTP probe using httpx and the standard library. Its CLI
requires URL, `--request` JSON file, `--count`, `--rate` requests/second, and `--output` JSON.
The bearer token comes from the environment variable named by `--token-env`, defaulting to
`SECND_READER_BEARER_TOKEN`. It schedules a fixed number of requests at fixed arrival intervals,
including concurrent requests when responses take longer than an interval. Results contain
`p50_ms`, `p95_ms` (nearest-rank latency over all attempts), `error_count`, and `wall_seconds`.
No automatic retry is made. It has not been run; no latency or capacity claim is made.

## Abuse controls and review limits

Failed sign-ins are limited in SQLite to five per normalized username and twenty per client
address in fifteen minutes. Registration attempts are limited to five per client address per
hour. In-flight sign-ins reserve capacity atomically; success releases its own reservation.
Limits survive process restart and ignore forwarded-address headers. Put the app behind a
proxy that preserves distinct client addresses if needed; otherwise users share the proxy's
address limit. Unknown and known sign-in names have the same failure/retry messages and hashing
work. A 429 response includes `Retry-After`. Secure, HttpOnly, SameSite=Lax opaque sessions expire
after one day. State-changing browser requests check both origin scheme and authority.

OAuth state is single-use and session-bound; credential saving rechecks a live owned session.
Connection changes serialize within the one supported API process. Deletion purges workspace
SQL rows and artifacts, clears runtime guard/error state, and rejects stale work on deleted
names until process restart. Choose a new workspace name during that process lifetime.
SQLite atomic claims protect action/undo idempotency, with ambiguous deliveries left for manual
review. A stop is rechecked after Ads snapshot reads. Already submitted provider requests may
finish, and disconnect/unsubscribe failures cannot revoke Meta-side app roles. Local accounts
still have no email verification or password recovery. Spot-check counters remain in-process;
reviews, policy, stop, history and action claims are durable. Workspace evidence uses the latest
1,000 distinct eligible reviews; it is not a causal outcome estimate or a population guarantee.

## Synthetic demo and verification

The existing seeded synthetic inbox demo provides two fictional workspaces and local CPU
inference. Start it with `uv run python -m backend.tests.inbox_demo --port 8000` after building
the frontend. The synthetic connector demo uses mocked transport. Demo mode is visibly marked,
its fallback session cannot open signed-in workspaces, and the app's default demo transport
cannot call Meta. Demo flags and observations are not customer measurements.

The [measured verification record](../results/c4-harness/result.json) contains the checks and limitations. Browser execution and
live OAuth remain separate checks; no claim of live Meta verification is made here.
