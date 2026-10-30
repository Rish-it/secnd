# Ren for Meta apps

Ren is an open decision model built for Meta's platform. Its encoder is tuned
on Meta's open assistant data; this integration routes WhatsApp Business Cloud
API and Messenger text messages through your workspace's calibrated decision loop.
The reply text is the predicted workspace label, not generated conversation.

## Configure a workspace

Import historical labelled decisions first using the [workspace API](../api/README.md).
This freezes calibration separately from training. The webhook does not create
an untrained workspace or invent a prediction: it returns 404 for an unknown
workspace, or 503 until a first model exists.

Configure these backend settings in `.env`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `SECND_META_APP_SECRET` | empty | Meta app secret for raw-body HMAC-SHA256 validation |
| `SECND_META_VERIFY_TOKEN` | empty | Your webhook verification token |
| `SECND_WHATSAPP_TOKEN` | empty | WhatsApp access token; keep it out of logs |
| `SECND_WHATSAPP_SEND_ENABLED` | `false` | Explicitly enable WhatsApp sends |
| `SECND_META_GRAPH_VERSION` | `v25.0` | Pinned Graph API version for sends |

Register this workspace-specific callback with your Meta app:
`https://YOUR_HOST/api/v1/workspaces/default/integrations/meta/webhook`.
Subscribe to WhatsApp messages and/or Messenger Page messages. Meta requires a
public HTTPS callback for real deliveries. Local curl requests exercise the same
handler. Start local demos with access logging disabled so verification tokens
in query strings stay out of terminal logs:

```sh
uv run uvicorn backend.api:app --host 127.0.0.1 --port 8010 --no-access-log
```

## Verify and deliver with curl

For this local handshake example, set your verification token to `local-demo`.
The response body is the exact challenge, as plain text:

```sh
API=http://127.0.0.1:8010/api/v1/workspaces/default
curl --fail-with-body -sS "$API/integrations/meta/webhook?hub.mode=subscribe&hub.verify_token=local-demo&hub.challenge=1158201444"
```

Sign the exact fixture bytes using the app secret already in `.env`; the script
prints only the signature. Keep the fixture bytes unchanged between signing and
posting. Whitespace and Unicode escaping affect the signature.

```sh
SIGNATURE=$(uv run python - <<'PY'
import hashlib
import hmac
from pathlib import Path
from backend.config import Settings
secret = Settings().meta_app_secret.get_secret_value()
assert secret, 'Set SECND_META_APP_SECRET in .env'
body = Path('backend/tests/fixtures/meta/whatsapp-text.json').read_bytes()
print('sha256=' + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest())
PY
)
curl --fail-with-body -sS "$API/integrations/meta/webhook" \
  -H 'Content-Type: application/json' \
  -H "X-Hub-Signature-256: $SIGNATURE" \
  --data-binary @backend/tests/fixtures/meta/whatsapp-text.json
unset SIGNATURE
```

Use `messenger-text.json` in both places to exercise Messenger. Both fixtures use
synthetic identifiers. Missing, incorrect or tampered signatures return 403 before
any item or prediction is saved; missing signing configuration returns 503.
Payloads are capped at 1 MiB and 256 supported messages.

## Decisions and human answers

Each message becomes a deterministic workspace item. Automatic decisions return
`status: dry_run` and a reply label. No request is sent to Meta by default.
WhatsApp sends occur only when both an access token and the explicit send flag
are configured. Messenger replies remain dry-run previews; a Page token/sender
adapter is not included. The integration never calls a teacher model to answer.

Uncertain items and random spot-checks return `status: queued` and appear in the
ordinary workspace queue. Answer from the Companion UI, or use the returned
`item_id` with a valid workspace label:

```sh
curl --fail-with-body -sS "$API/queue"
curl --fail-with-body -sS "$API/answers" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: meta-human-example-1' \
  -d '{"item_id":"ITEM_ID_FROM_WEBHOOK","label":"YOUR_WORKSPACE_LABEL"}'
curl --fail-with-body -sS "$API/metrics"
```

The answer is durable before background `Loop.on_answer` runs; spot-check labels
retain their source and can trigger `guarded`. No calibration or test rows enter
training. Human labels use the existing answer API and do not send an outbound
message themselves.

Message IDs are persisted per workspace, platform and receiving account.
Duplicate deliveries return the saved result without rerouting or resending,
including after restart. Reusing an ID with different content returns 409.
After an outbound timeout or interrupted send, `delivery_unknown` requires a
person to check delivery; duplicates never retry a possibly delivered message.
This follows the service's single-process limit.

Delivery receipts, echoes, media and unknown message types are acknowledged and
counted in `ignored`; text is the supported input. Malformed signed messages
return 422. The connector preserves the existing workspace queue and model gate.

## Public sources

Payload fixtures adapt these public Meta references, read October 29, 2026:

- [WhatsApp incoming messages](https://developers.facebook.com/documentation/business-messaging/whatsapp/webhooks/reference/messages).
- [Messenger verification, events and signature validation](https://developers.facebook.com/docs/messenger-platform/webhooks).
- [WhatsApp text sends](https://developers.facebook.com/documentation/business-messaging/whatsapp/messages/text-messages).
- [Graph API v25.0](https://developers.facebook.com/blog/post/2026/02/18/introducing-graph-api-v25-and-marketing-api-v25/).

## Instagram and receiving-account mapping

The same callback also accepts Instagram messaging events (`object: instagram`).
It applies the raw-body signature check, persistent message deduplication and
human-answer path used by WhatsApp and Messenger. Echoes, receipts and media are
acknowledged without creating decisions. Instagram and Messenger do not send replies.

Every supported message must match an explicit receiving-account mapping. A signed
message from an unmapped account, or an account mapped to another workspace, returns
403 before any messages in that batch are routed. Existing installations must add
this mapping when upgrading. Configure synthetic sandbox IDs as follows:

```dotenv
SECND_META_WORKSPACE_ACCOUNTS={"whatsapp:100000000000002":"messages","messenger:100000000000003":"messages","instagram:100000000000005":"messages"}
```

Keys identify the receiving phone-number ID, Page ID or Instagram professional
account ID; values identify an existing workspace. Tokens remain in backend Settings.
The Teach screen shows connection status without exposing credentials. Verification
credentials are redacted from the Uvicorn access log.

Provenance is saved as `source_app` in each prediction. The decision inbox uses it
for source badges and groups, including after restart. Confirmations and corrections
use the existing answers endpoint; no teacher participates in a decision.

[Instagram webhook reference](https://developers.facebook.com/docs/instagram-platform/webhooks/).
The supplied Instagram payload is synthetic. No live Instagram delivery is claimed.

## Ads sandbox insights and execution

Use a workspace whose allowed labels are exactly `pause`, `keep`, `raise budget`,
`lower budget`. Train it from your own decisions, or use label-only onboarding;
a reader workspace without independent calibration remains entirely human-reviewed.
Do not infer performance from the connector's synthetic transport tests.

```dotenv
SECND_ADS_TOKEN=
SECND_ADS_WORKSPACE_ACCOUNTS={"ads":"act_900"}
SECND_ADS_SANDBOX_ACCOUNTS=["act_900"]
SECND_ADS_SEND_ENABLED=false
SECND_ADS_CPA_ACTION_TYPE=purchase
SECND_ADS_BUDGET_STEP_PERCENT=10
```

Only explicitly allowlisted test/sandbox accounts can be read or changed. Configure
read access for insights; execution needs the appropriate separate account permissions.
The container reads runtime `.env` settings through Compose's optional `env_file`
(Compose 2.24 or newer). No secret is baked into an image or returned to the app.

The Teach screen can fetch campaign or ad set insights for a completed date range.
`POST /api/v1/workspaces/ads/integrations/ads/insights:sync` accepts `level`, `since`
and `until`. Spend, action-specific CPA and CTR become a typed, immutable Ren
proposal. Missing CPA stays unavailable. The proposal's source is Ads. Uncertain
proposals enter the ordinary human queue; every answer updates Ren's history.

The default is proposals only. A mutation requires **all** of:

- `SECND_ADS_SEND_ENABLED=true`, explicitly set by the human;
- the person has saved `auto_act: true` for that workspace;
- Ren marked the proposal automatic, the current model and threshold still match,
  its calibration precision lower bound clears 99%, and the live guard is inactive;
- no human has answered the proposal, its target still belongs to the mapped sandbox
  account, and its daily budget still matches the saved snapshot.

Confirming an uncertified proposal updates history; it does not bypass these execution
conditions. Toggle auto-act under Teach. `PUT /integrations/ads/preferences` saves the
workspace preference; `POST /integrations/ads/proposals/{item_id}:execute` checks an
existing proposal. Sync also attempts eligible proposals when both switches are on.

Pause writes `status=PAUSED`. Keep makes no account mutation. Budget changes adjust
only the selected target's own daily budget using integer minor currency units and
the configured step. An ad set with campaign-managed budget cannot silently change
its parent campaign; fetch a separate campaign-level proposal instead.

Each proposal and execution attempt is logged before mutation. A durable claim permits
one attempted action per workspace, target and date range, including when insights
change. Replays do not repeat writes. Timeout, ambiguous success or interrupted execution
becomes `delivery_unknown`; inspect the account manually, as the app never retries a
possibly applied mutation. No rollback of external Ads changes is implied by inbox undo.

`GET /integrations/ads/log?after=0&limit=20` returns workspace-scoped audit entries and
`next_after` for subsequent entries. Error bodies omit remote responses, credentials and
provider exception details. Pagination reuses a fixed Graph endpoint and cursor; it never
follows a returned URL that could contain a token or redirect to another host.

The implementation uses fields and update shapes defined by Meta's primary SDK:
[Insights](https://github.com/facebook/facebook-python-business-sdk/blob/main/facebook_business/adobjects/adsinsights.py),
[Ad account insights](https://github.com/facebook/facebook-python-business-sdk/blob/main/facebook_business/adobjects/adaccount.py),
[ad set budgets/status](https://github.com/facebook/facebook-python-business-sdk/blob/main/facebook_business/adobjects/adset.py).
[Mocked browser check and screenshots](../results/c2-meta-apps/README.md).
