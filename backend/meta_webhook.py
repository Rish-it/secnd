"""Signed Meta messages enter the same workspace decision and answer loop."""

import hashlib
import hmac
import json
from typing import Annotated, Literal

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.connector_source import attach_source
from backend.data import merchant_key
from backend.schemas import Error, WorkspaceId
from backend.store import IdempotencyConflict, encode
from backend.workspace import WorkspaceService

MAX_BODY = 1024 * 1024
MetaId = Annotated[str, Field(pattern=r"^\d{1,32}$")]


class Message(BaseModel):
    platform: Literal["whatsapp", "messenger", "instagram"]
    message_id: Annotated[str, Field(min_length=1, max_length=512)]
    sender: MetaId
    account: MetaId
    text: Annotated[str, Field(min_length=1, max_length=10000)]

    @property
    def item_id(self):
        key = encode([self.platform, self.account, self.message_id])
        return "meta-" + hashlib.sha256(key.encode()).hexdigest()[:48]


def parse(payload: dict) -> tuple[list[Message], int]:
    messages, ignored = [], 0
    try:
        for entry in payload.get("entry", []):
            if not isinstance(entry, dict):
                raise ValueError("Invalid Meta message payload.")
            if payload.get("object") == "whatsapp_business_account":
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    if change.get("field") != "messages" or not value.get("messages"):
                        ignored += 1
                        continue
                    for message in value["messages"]:
                        if message.get("type") != "text":
                            ignored += 1
                            continue
                        messages.append(
                            Message(
                                platform="whatsapp",
                                message_id=message["id"],
                                sender=message["from"],
                                account=value["metadata"]["phone_number_id"],
                                text=message["text"]["body"],
                            )
                        )
            elif payload.get("object") in ("page", "instagram"):
                for event in entry.get("messaging", []):
                    message = event.get("message", {})
                    if message.get("is_echo") or not message.get("text"):
                        ignored += 1
                        continue
                    messages.append(
                        Message(
                            platform="instagram"
                            if payload["object"] == "instagram"
                            else "messenger",
                            message_id=message["mid"],
                            sender=event["sender"]["id"],
                            account=event["recipient"]["id"],
                            text=message["text"],
                        )
                    )
            else:
                ignored += 1
        if len(messages) > 256:
            raise ValueError("Too many messages.")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid Meta message payload.") from error
    return messages, ignored


class MetaWebhook:
    def __init__(self, settings: Settings, service: WorkspaceService, *, transport=None):
        self.settings, self.service = settings, service
        self.transport = transport
        with service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS meta_messages (
                workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                item_id TEXT NOT NULL, message_json TEXT NOT NULL, result_json TEXT,
                PRIMARY KEY (workspace_id, item_id)
            )""")

    def _save(self, workspace, item_id, result):
        with self.service.store.connect(write=True) as db:
            db.execute(
                "UPDATE meta_messages SET result_json=? WHERE workspace_id=? AND item_id=?",
                (encode(result), workspace, item_id),
            )

    def handle(self, workspace, message: Message) -> dict:
        if (
            self.settings.meta_workspace_accounts.get(f"{message.platform}:{message.account}")
            != workspace
        ):
            raise ValueError("Receiving account is not mapped to this workspace.")
        service = self.service
        with service._lock(workspace):
            content = message.model_dump_json()
            with service.store.connect(write=True) as db:
                old = db.execute(
                    "SELECT * FROM meta_messages WHERE workspace_id=? AND item_id=?",
                    (workspace, message.item_id),
                ).fetchone()
                if old and old["message_json"] != content:
                    raise IdempotencyConflict(
                        "A Meta message ID was reused with different content."
                    )
                if old and old["result_json"]:
                    result = json.loads(old["result_json"])
                    if result["status"] == "replying":
                        result["status"] = "delivery_unknown"
                    return {**result, "duplicate": True}
                if old is None:
                    db.execute(
                        "INSERT INTO meta_messages VALUES (?, ?, ?, NULL)",
                        (workspace, message.item_id, content),
                    )
            decision = service.store.latest_prediction(workspace, message.item_id)
            if decision is not None:
                item = service.store.item(workspace, message.item_id)
                if (item["text"], item["amount"], item["date"], item["merchant_key"]) != (
                    message.text,
                    None,
                    None,
                    merchant_key(message.text),
                ):
                    raise IdempotencyConflict("A message item ID already has different content.")
            if decision is None:
                frame = pd.DataFrame(
                    [
                        dict(
                            id=message.item_id,
                            text=message.text,
                            amount=None,
                            date=None,
                            merchant_key=merchant_key(message.text),
                            source_app=message.platform,
                        )
                    ]
                )
                decision = service.route(workspace, frame)[0]
            attach_source(service, workspace, message.item_id, message.platform)
            result = {
                "message_id": message.message_id,
                "item_id": message.item_id,
                "platform": message.platform,
                "status": "dry_run" if decision["auto"] else "queued",
                "reply": decision["label"] if decision["auto"] else None,
                "duplicate": False,
            }
            token = self.settings.whatsapp_token.get_secret_value()
            send = (
                message.platform == "whatsapp"
                and decision["auto"]
                and self.settings.whatsapp_send_enabled
                and token
            )
            if send:
                result["status"] = "replying"
                self._save(workspace, message.item_id, result)
                try:
                    with httpx.Client(transport=self.transport, timeout=10) as client:
                        response = client.post(
                            f"https://graph.facebook.com/{self.settings.meta_graph_version}"
                            f"/{message.account}/messages",
                            headers={"Authorization": f"Bearer {token}"},
                            json={
                                "messaging_product": "whatsapp",
                                "to": message.sender,
                                "type": "text",
                                "text": {"body": result["reply"]},
                            },
                        )
                        response.raise_for_status()
                    result["status"] = "sent"
                except httpx.HTTPError:
                    # A timeout can follow delivery; replay must never send it again.
                    result["status"] = "delivery_unknown"
            self._save(workspace, message.item_id, result)
            return result


def router(settings: Settings) -> APIRouter:
    routes = APIRouter(
        prefix="/api/v1/workspaces/{ws}/integrations/meta",
        tags=["Meta integration"],
        responses={status: {"model": Error} for status in (403, 404, 409, 413, 422, 503)},
    )

    @routes.get("/status")
    def status(ws: WorkspaceId, request: Request):
        service = request.app.state.workspaces
        accounts = {
            platform: [
                key.split(":", 1)[1]
                for key, workspace in settings.meta_workspace_accounts.items()
                if workspace == ws and key.startswith(platform + ":")
            ]
            for platform in ("whatsapp", "messenger", "instagram")
        }
        try:
            service.store.workspace(ws)
            ready = service.has_model(ws)
        except KeyError:
            ready = False
        return {
            "accounts": accounts,
            "model_ready": ready,
            "signing_configured": bool(settings.meta_app_secret.get_secret_value()),
            "whatsapp_send_enabled": settings.whatsapp_send_enabled,
        }

    @routes.get("/webhook", response_class=PlainTextResponse)
    def verify(
        ws: WorkspaceId,
        mode: Annotated[str, Query(alias="hub.mode")],
        token: Annotated[str, Query(alias="hub.verify_token")],
        challenge: Annotated[str, Query(alias="hub.challenge", min_length=1, max_length=256)],
    ):
        expected = settings.meta_verify_token.get_secret_value()
        if not expected:
            raise HTTPException(503, "Meta webhook verification is not configured.")
        if mode != "subscribe" or not hmac.compare_digest(token.encode(), expected.encode()):
            raise HTTPException(403, "Meta webhook verification failed.")
        return challenge

    @routes.post("/webhook")
    async def receive(ws: WorkspaceId, request: Request):
        secret = settings.meta_app_secret.get_secret_value()
        if not secret:
            raise HTTPException(503, "Meta webhook signing is not configured.")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY:
                raise HTTPException(413, "Meta payload exceeds 1 MiB.")
            chunks.append(chunk)
        raw = b"".join(chunks)
        expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
        supplied = request.headers.get("X-Hub-Signature-256", "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(403, "Meta signature verification failed.")
        try:
            messages, ignored = parse(json.loads(raw))
        except ValueError as error:
            raise HTTPException(422, "Invalid Meta message payload.") from error
        service = request.app.state.workspaces
        try:
            service.store.workspace(ws)
        except KeyError as error:
            raise HTTPException(404, "Workspace was not found.") from error
        if any(
            settings.meta_workspace_accounts.get(f"{m.platform}:{m.account}") != ws
            for m in messages
        ):
            raise HTTPException(403, "Receiving account is not mapped to this workspace.")
        if messages and not service.has_model(ws):
            raise HTTPException(503, "Import labelled decisions before receiving Meta messages.")
        try:
            results = [
                await run_in_threadpool(request.app.state.meta_webhook.handle, ws, message)
                for message in messages
            ]
        except IdempotencyConflict as error:
            raise HTTPException(409, str(error)) from error
        return {"accepted": len(results), "ignored": ignored, "items": results}

    return routes
