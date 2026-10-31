"""Opaque cookie sessions and workspace ownership; demo access is explicit."""

import hashlib
import hmac
import secrets
import sqlite3
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, SecretStr

from backend.schemas import WorkspaceId

COOKIE = "secnd_session"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Credentials(BaseModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.@-]{3,128}$")
    password: SecretStr = Field(min_length=12, max_length=256)


class NewWorkspace(BaseModel):
    workspace_id: WorkspaceId


class Auth:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        with store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL, password_hash TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                expires REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS workspace_owners (
                workspace_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id))""")

    def principal(self, request):
        token = request.cookies.get(COOKIE, "")
        with self.store.connect() as db:
            row = db.execute(
                "SELECT user_id FROM sessions WHERE id=? AND expires>?",
                (digest(token), time.time()),
            ).fetchone()
        if row:
            request.state.session_id = digest(token)
            return row[0]
        if self.settings.demo_mode:
            request.state.session_id = None
            return "demo"
        raise HTTPException(401, "Sign in to open your workspace.")

    def workspace(self, user, ws):
        if user == "demo":
            with self.store.connect() as db:
                if db.execute(
                    "SELECT 1 FROM workspace_owners WHERE workspace_id=?", (ws,)
                ).fetchone():
                    raise HTTPException(404, "Workspace was not found.")
            return
        with self.store.connect() as db:
            row = db.execute(
                "SELECT user_id FROM workspace_owners WHERE workspace_id=?", (ws,)
            ).fetchone()
        if not row or row[0] != user:
            raise HTTPException(404, "Workspace was not found.")

    def issue(self, user, response):
        token = secrets.token_urlsafe(32)
        with self.store.connect(write=True) as db:
            db.execute("DELETE FROM sessions WHERE expires<=?", (time.time(),))
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?)", (digest(token), user, time.time() + 86400)
            )
        response.set_cookie(
            COOKIE,
            token,
            httponly=True,
            secure=self.settings.session_cookie_secure,
            samesite="lax",
            max_age=86400,
            path="/",
        )

    @staticmethod
    def password(value, salt):
        return hashlib.pbkdf2_hmac("sha256", value.encode(), bytes.fromhex(salt), 600000).hex()


def router():
    routes = APIRouter(prefix="/api/v1", tags=["Accounts"])

    @routes.post("/auth/register")
    def register(body: Credentials, request: Request, response: Response):
        auth = request.app.state.auth
        user, salt = secrets.token_hex(16), secrets.token_hex(16)
        try:
            with auth.store.connect(write=True) as db:
                db.execute(
                    "INSERT INTO users VALUES (?,?,?,?)",
                    (
                        user,
                        body.username.casefold(),
                        salt,
                        auth.password(body.password.get_secret_value(), salt),
                    ),
                )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Choose a different sign-in name.") from None
        auth.issue(user, response)
        return {"signed_in": True, "demo": False}

    @routes.post("/auth/login")
    def login(body: Credentials, request: Request, response: Response):
        auth = request.app.state.auth
        with auth.store.connect() as db:
            row = db.execute(
                "SELECT * FROM users WHERE username=?", (body.username.casefold(),)
            ).fetchone()
        salt = row["salt"] if row else "00" * 16
        candidate = auth.password(body.password.get_secret_value(), salt)
        if not row or not hmac.compare_digest(candidate, row["password_hash"]):
            raise HTTPException(401, "Sign-in name or password is incorrect.")
        auth.issue(row["id"], response)
        return {"signed_in": True, "demo": False}

    @routes.get("/auth/session")
    def session(request: Request):
        auth = request.app.state.auth
        try:
            user = auth.principal(request)
        except HTTPException:
            return {"signed_in": False, "demo": False}
        return {"signed_in": user != "demo", "demo": auth.settings.demo_mode}

    @routes.post("/auth/logout")
    def logout(request: Request, response: Response):
        auth = request.app.state.auth
        with auth.store.connect(write=True) as db:
            db.execute(
                "DELETE FROM sessions WHERE id=?", (digest(request.cookies.get(COOKIE, "")),)
            )
        response.delete_cookie(COOKIE, path="/")
        return {"signed_in": False}

    @routes.get("/workspaces")
    def workspaces(request: Request):
        auth = request.app.state.auth
        user = auth.principal(request)
        with auth.store.connect() as db:
            rows = db.execute(
                "SELECT w.workspace_id FROM workspaces w LEFT JOIN workspace_owners o "
                "ON o.workspace_id=w.workspace_id WHERE o.workspace_id IS NULL"
                if user == "demo"
                else "SELECT workspace_id FROM workspace_owners WHERE user_id=?",
                () if user == "demo" else (user,),
            ).fetchall()
        return {"items": [row[0] for row in rows], "synthetic": user == "demo"}

    @routes.post("/workspaces")
    def create(body: NewWorkspace, request: Request):
        auth = request.app.state.auth
        user = auth.principal(request)
        if user == "demo":
            raise HTTPException(409, "Use a signed-in account to create a workspace.")
        try:
            with auth.store.connect(write=True) as db:
                # Existing unowned demo workspaces cannot be claimed by a new account.
                if db.execute(
                    "SELECT 1 FROM workspaces WHERE workspace_id=?", (body.workspace_id,)
                ).fetchone():
                    raise HTTPException(409, "Choose a different workspace name.")
                db.execute("INSERT INTO workspace_owners VALUES (?,?)", (body.workspace_id, user))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Choose a different workspace name.") from None
        return {"workspace_id": body.workspace_id}

    return routes


def same_origin(request):
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site" or (
        origin and urlsplit(origin).netloc != request.url.netloc
    ):
        raise HTTPException(403, "Use this app's own page to make changes.")
