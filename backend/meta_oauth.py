"""Facebook Login for Business, encrypted credentials and user-selected assets."""

import base64
import hashlib
import hmac
import json
import secrets
import shutil
import time
from urllib.parse import urlencode

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.auth import digest
from backend.meta_ads import AdsUnavailable, MarketingClient
from backend.schemas import WorkspaceId
from backend.store import encode

REFUSED = (
    "Meta declined the connection. Until Advanced Access is approved, use an account "
    "with an admin, developer or tester role on this Meta app."
)


class Selection(BaseModel):
    assets: list[str] = Field(max_length=100)


class MetaOAuth:
    def __init__(self, settings, service, transport=None):
        self.settings, self.service, self.transport = settings, service, transport
        with service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS meta_oauth_states (
                session_id TEXT PRIMARY KEY, state_hash TEXT NOT NULL, workspace_id TEXT NOT NULL,
                expires REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS meta_connections (
                user_id TEXT PRIMARY KEY, meta_user_id TEXT NOT NULL,
                token_blob BLOB NOT NULL, expires REAL NOT NULL, assets_json TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS meta_identities (
                user_id TEXT PRIMARY KEY, meta_user_id TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS meta_assets (
                kind TEXT NOT NULL, asset_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL, PRIMARY KEY(kind, asset_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS meta_deletions (
                code TEXT PRIMARY KEY, completed BOOLEAN NOT NULL)""")

            db.execute(
                "INSERT OR IGNORE INTO meta_identities "
                "SELECT user_id,meta_user_id FROM meta_connections"
            )

    def cipher(self):
        try:
            return Fernet(self.settings.meta_token_key.get_secret_value().encode())
        except (ValueError, TypeError):
            raise HTTPException(503, "Configure a valid Meta token encryption key.") from None

    def client(self, token="", *, anonymous=False):
        return MarketingClient(self.settings, self.transport, token=token, anonymous=anonymous)

    def start(self, user, session, workspace):
        config = self.settings
        if not session or user == "demo":
            raise HTTPException(401, "Sign in before connecting Meta.")
        if not all(
            (
                config.meta_app_id,
                config.meta_login_config_id,
                config.meta_redirect_uri,
                config.meta_app_secret.get_secret_value(),
                config.meta_public_url,
            )
        ):
            raise HTTPException(503, "Facebook Login for Business needs operator setup.")
        self.cipher()
        state = secrets.token_urlsafe(32)
        with self.service.store.connect(write=True) as db:
            db.execute(
                "INSERT OR REPLACE INTO meta_oauth_states VALUES (?,?,?,?)",
                (session, digest(state), workspace, time.time() + 600),
            )
        return f"https://www.facebook.com/{config.meta_graph_version}/dialog/oauth?" + urlencode(
            {
                "client_id": config.meta_app_id,
                "config_id": config.meta_login_config_id,
                "redirect_uri": config.meta_redirect_uri,
                "response_type": "code",
                "scope": ",".join(config.meta_scopes),
                "state": state,
            }
        )

    def callback(self, user, session, state, code, error):
        # Consume once, including on denial or mismatch, under the same write transaction.
        with self.service.store.connect(write=True) as db:
            row = db.execute(
                "SELECT * FROM meta_oauth_states WHERE session_id=?", (session,)
            ).fetchone()
            db.execute("DELETE FROM meta_oauth_states WHERE session_id=?", (session,))
        if (
            not row
            or row["expires"] < time.time()
            or not hmac.compare_digest(row["state_hash"], digest(state))
        ):
            raise HTTPException(403, "Meta sign-in expired or state did not match. Connect again.")
        if error or not code:
            raise HTTPException(403, REFUSED)
        try:
            config = self.settings
            auth = self.client(anonymous=True)
            short = auth.request(
                "POST",
                "oauth/access_token",
                data={
                    "client_id": config.meta_app_id,
                    "client_secret": config.meta_app_secret.get_secret_value(),
                    "redirect_uri": config.meta_redirect_uri,
                    "code": code,
                },
            )["access_token"]
            long = auth.request(
                "POST",
                "oauth/access_token",
                data={
                    "grant_type": "fb_exchange_token",
                    "client_id": config.meta_app_id,
                    "client_secret": config.meta_app_secret.get_secret_value(),
                    "fb_exchange_token": short,
                },
            )
            token = long["access_token"]
            client = self.client(token)
            meta_user = client.request("GET", "me", params={"fields": "id"})["id"]
            assets, tokens = self.inventory(client)
            blob = self.cipher().encrypt(encode({"user": token, "assets": tokens}).encode())
            with self.service.store.connect(write=True) as db:
                # Reconnection invalidates previous selections; user must pick assets again.
                db.execute("INSERT OR REPLACE INTO meta_identities VALUES (?,?)", (user, meta_user))
                db.execute("DELETE FROM meta_assets WHERE user_id=?", (user,))
                db.execute(
                    "INSERT OR REPLACE INTO meta_connections VALUES (?,?,?,?,?)",
                    (
                        user,
                        meta_user,
                        blob,
                        time.time() + int(long["expires_in"]),
                        encode(assets),
                    ),
                )
            return row["workspace_id"]
        except (AdsUnavailable, KeyError, ValueError, TypeError):
            raise HTTPException(403, REFUSED) from None

    @staticmethod
    def rows(client, path, fields):
        params, rows, seen = {"fields": fields, "limit": "100"}, [], set()
        for _ in range(10):
            data = client.request("GET", path, params=params)
            rows.extend(data["data"])
            paging = data.get("paging", {})
            if not paging.get("next"):
                return rows
            after = paging.get("cursors", {}).get("after")
            if not isinstance(after, str) or not 1 <= len(after) <= 512 or after in seen:
                break
            seen.add(after)
            params["after"] = after
        raise AdsUnavailable("Meta asset list is incomplete. Narrow access and reconnect.")

    def inventory(self, client):
        assets, tokens = [], {}

        def add(kind, identifier, name, token=None):
            identifier = str(identifier)
            raw = identifier.removeprefix("act_") if kind == "ads" else identifier
            if not raw.isdigit() or len(raw) > 32:
                raise ValueError("Invalid asset id")
            key = f"{kind}:{identifier}"
            assets.append({"key": key, "kind": kind, "id": identifier, "name": str(name)[:500]})
            if token:
                tokens[key] = token

        for row in self.rows(client, "me/adaccounts", "id,name"):
            add("ads", row["id"], row.get("name", "Ad account"))
        for row in self.rows(
            client, "me/accounts", "id,name,access_token,instagram_business_account{id,username}"
        ):
            add("messenger", row["id"], row.get("name", "Page"), row.get("access_token"))
            ig = row.get("instagram_business_account")
            if ig:
                add("instagram", ig["id"], ig.get("username", "Instagram"), row.get("access_token"))
        for business in self.rows(client, "me/businesses", "id"):
            if not str(business["id"]).isdigit():
                raise ValueError("Invalid business id")
            for waba in self.rows(
                client, f"{business['id']}/owned_whatsapp_business_accounts", "id,name"
            ):
                if not str(waba["id"]).isdigit():
                    raise ValueError("Invalid business id")
                for number in self.rows(client, f"{waba['id']}/phone_numbers", "id,verified_name"):
                    add("whatsapp", number["id"], number.get("verified_name", "WhatsApp"))
        return assets, tokens

    def status(self, user, workspace):
        with self.service.store.connect() as db:
            row = db.execute(
                "SELECT expires,assets_json FROM meta_connections WHERE user_id=?", (user,)
            ).fetchone()
            selected = db.execute(
                "SELECT kind,asset_id FROM meta_assets WHERE user_id=? AND workspace_id=?",
                (user, workspace),
            ).fetchall()
        return {
            "connected": bool(row and row["expires"] > time.time()),
            "assets": json.loads(row["assets_json"]) if row else [],
            "selected": [f"{r[0]}:{r[1]}" for r in selected],
            "access_note": REFUSED,
        }

    def select(self, user, workspace, keys):
        status = self.status(user, workspace)
        allowed = {row["key"] for row in status["assets"]}
        if (
            not status["connected"]
            or not set(keys).issubset(allowed)
            or len(set(keys)) != len(keys)
        ):
            raise HTTPException(422, "Choose accounts from your current Meta connection.")
        if sum(key.startswith("ads:") for key in keys) > 1:
            raise HTTPException(422, "Choose one sandbox ad account for this workspace.")
        if any(
            key[4:] not in self.settings.ads_sandbox_accounts
            for key in keys
            if key.startswith("ads:")
        ):
            raise HTTPException(422, "Only operator-allowed sandbox ad accounts can be selected.")
        with self.service.store.connect(write=True) as db:
            for key in keys:
                kind, identifier = key.split(":", 1)
                existing = db.execute(
                    "SELECT workspace_id,user_id FROM meta_assets WHERE kind=? AND asset_id=?",
                    (kind, identifier),
                ).fetchone()
                if existing and tuple(existing) != (workspace, user):
                    raise HTTPException(409, "An account is already mapped to another workspace.")
            db.execute(
                "DELETE FROM meta_assets WHERE workspace_id=? AND user_id=?", (workspace, user)
            )
            db.executemany(
                "INSERT INTO meta_assets VALUES (?,?,?,?)",
                [(*k.split(":", 1), workspace, user) for k in keys],
            )
        return self.status(user, workspace)

    def owned(self, workspace):
        with self.service.store.connect() as db:
            return (
                db.execute(
                    "SELECT user_id FROM workspace_owners WHERE workspace_id=?", (workspace,)
                ).fetchone()
                is not None
            )

    def mappings(self, workspace):
        with self.service.store.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM meta_assets WHERE workspace_id=?", (workspace,)
                ).fetchall()
            ]

    def mapped(self, workspace, kind, identifier):
        if self.owned(workspace):
            return any(
                r["kind"] == kind and r["asset_id"] == identifier for r in self.mappings(workspace)
            )
        return (
            self.settings.demo_mode
            and self.settings.meta_workspace_accounts.get(f"{kind}:{identifier}") == workspace
        )

    def token(self, workspace, kind, identifier):
        with self.service.store.connect() as db:
            row = db.execute(
                "SELECT c.* FROM meta_connections c JOIN meta_assets a ON "
                "a.user_id=c.user_id WHERE a.workspace_id=? AND a.kind=? AND a.asset_id=?",
                (workspace, kind, identifier),
            ).fetchone()
        if not row or row["expires"] <= time.time():
            raise AdsUnavailable("Reconnect Meta to refresh this workspace's access.")
        try:
            tokens = json.loads(self.cipher().decrypt(row["token_blob"]))
            return tokens["assets"].get(f"{kind}:{identifier}", tokens["user"])
        except (InvalidToken, ValueError, KeyError):
            raise AdsUnavailable("Reconnect Meta to refresh this workspace's access.") from None

    def disconnect(self, user):
        with self.service.store.connect(write=True) as db:
            db.execute("DELETE FROM meta_assets WHERE user_id=?", (user,))
            db.execute("DELETE FROM meta_connections WHERE user_id=?", (user,))
            db.execute(
                "DELETE FROM meta_oauth_states WHERE session_id IN (SELECT id FROM "
                "sessions WHERE user_id=?)",
                (user,),
            )
        return {"connected": False}

    def delete(self, signed):
        try:
            signature, payload = signed.split(".", 1)

            def decode(v):
                return base64.urlsafe_b64decode(v + "=" * (-len(v) % 4))

            expected = hmac.new(
                self.settings.meta_app_secret.get_secret_value().encode(),
                payload.encode(),
                hashlib.sha256,
            ).digest()
            if not self.settings.meta_app_secret.get_secret_value() or not hmac.compare_digest(
                decode(signature), expected
            ):
                raise ValueError
            data = json.loads(decode(payload))
            if data["algorithm"].upper() != "HMAC-SHA256":
                raise ValueError
            meta_user = str(data["user_id"])
        except (ValueError, KeyError, TypeError):
            raise HTTPException(403, "Invalid Meta deletion signature.") from None
        with self.service.store.connect() as db:
            users = [
                r[0]
                for r in db.execute(
                    "SELECT user_id FROM meta_identities WHERE meta_user_id=?", (meta_user,)
                )
            ]
        # Remove derived workspace data too, including local history and trained artifacts.
        try:
            self.service.drain()
        except RuntimeError:
            # Failed retraining must never prevent a person deleting their data.
            pass
        for user in users:
            with self.service.store.connect() as db:
                workspaces = [
                    r[0]
                    for r in db.execute(
                        "SELECT workspace_id FROM workspace_owners WHERE user_id=?", (user,)
                    )
                ]
            self.disconnect(user)
            for ws in workspaces:
                with self.service._lock(ws), self.service.store.connect(write=True) as db:
                    tables = {
                        r[0]
                        for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    }
                    ordered = [
                        "answer_undos",
                        "answers",
                        "labels",
                        "predictions",
                        "items",
                        "model_versions",
                    ]
                    other = [
                        t
                        for t in tables
                        if t not in ordered + ["workspaces", "workspace_owners"]
                        and "workspace_id"
                        in {r[1] for r in db.execute(f'PRAGMA table_info("{t}")')}
                    ]
                    for table in other + ordered + ["workspaces", "workspace_owners"]:
                        db.execute(f'DELETE FROM "{table}" WHERE workspace_id=?', (ws,))
                    self.service._loops.pop(ws, None)
                    self.service._positions.pop(ws, None)
                shutil.rmtree(self.settings.data_dir / "workspaces" / ws, ignore_errors=True)
        with self.service.store.connect(write=True) as db:
            db.execute("DELETE FROM meta_identities WHERE meta_user_id=?", (meta_user,))
        code = secrets.token_urlsafe(24)
        with self.service.store.connect(write=True) as db:
            db.execute("INSERT INTO meta_deletions VALUES (?,1)", (code,))
        return {
            "url": f"{self.settings.meta_public_url.rstrip('/')}/api/v1/meta/deletion/{code}",
            "confirmation_code": code,
        }


def router():
    routes = APIRouter(tags=["Meta login"])

    def identity(request):
        user = request.app.state.auth.principal(request)
        if user == "demo":
            raise HTTPException(401, "Sign in before connecting Meta.")
        return user

    @routes.get("/api/v1/workspaces/{ws}/integrations/meta/oauth/start")
    def start(ws: WorkspaceId, request: Request):
        user = identity(request)
        return RedirectResponse(
            request.app.state.meta_oauth.start(user, request.state.session_id, ws), 303
        )

    @routes.get("/api/v1/meta/oauth/callback")
    def callback(request: Request, state: str = "", code: str = "", error: str = ""):
        user = identity(request)
        with request.app.state.workspaces.store.connect() as db:
            pending = db.execute(
                "SELECT workspace_id FROM meta_oauth_states WHERE session_id=?",
                (request.state.session_id,),
            ).fetchone()
        try:
            ws = request.app.state.meta_oauth.callback(
                user, request.state.session_id, state, code, error
            )
        except HTTPException:
            if "text/html" in request.headers.get("accept", ""):
                return RedirectResponse(
                    "/?"
                    + urlencode(
                        {"workspace": pending[0] if pending else "default", "meta_error": "refused"}
                    ),
                    303,
                )
            raise
        return RedirectResponse("/?" + urlencode({"workspace": ws, "connected": "true"}), 303)

    @routes.get("/api/v1/workspaces/{ws}/integrations/meta/connection")
    def status(ws: WorkspaceId, request: Request):
        return request.app.state.meta_oauth.status(identity(request), ws)

    @routes.put("/api/v1/workspaces/{ws}/integrations/meta/accounts")
    def select(ws: WorkspaceId, body: Selection, request: Request):
        return request.app.state.meta_oauth.select(identity(request), ws, body.assets)

    @routes.post("/api/v1/meta/disconnect")
    def disconnect(request: Request):
        return request.app.state.meta_oauth.disconnect(identity(request))

    @routes.post("/api/v1/meta/data-deletion")
    def delete(request: Request, signed_request: str = Form(max_length=16384)):
        return request.app.state.meta_oauth.delete(signed_request)

    @routes.get("/api/v1/meta/deletion/{code}")
    def deleted(code: str, request: Request):
        with request.app.state.workspaces.store.connect() as db:
            row = db.execute(
                "SELECT completed FROM meta_deletions WHERE code=?", (code,)
            ).fetchone()
        if not row:
            raise HTTPException(404, "Deletion request was not found.")
        return {"completed": bool(row[0])}

    return routes
