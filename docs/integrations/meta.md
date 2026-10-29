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
