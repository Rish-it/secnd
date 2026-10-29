# Meta webhook fixtures

Payload structures adapted from Meta's public documentation, read October 29, 2026.
Names, message text and identifiers are synthetic.

- [WhatsApp incoming messages reference](https://developers.facebook.com/documentation/business-messaging/whatsapp/webhooks/reference/messages) (`whatsapp-text.json`).
- [Messenger webhook events and text messages](https://developers.facebook.com/docs/messenger-platform/webhooks#event-notifications) (`messenger-text.json`).
- [Messenger payload signature validation](https://developers.facebook.com/docs/messenger-platform/webhooks#validate-payloads).

Tests derive status notifications, attachments, echoes and unsupported message types
from these fixtures. No fixture is an evaluation dataset or training benchmark.
