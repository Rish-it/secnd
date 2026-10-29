"""Workspace storage. Each write owns one connection and commits atomically."""

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

SOURCES = ("seed", "answer", "spot_check", "muse_cold_start")
SCHEMA_VERSION = 2
TABLES = (
    """CREATE TABLE workspaces (
        workspace_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, created_at TEXT NOT NULL
    )""",
    """CREATE TABLE items (
        workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
        id TEXT NOT NULL, text TEXT NOT NULL, amount DOUBLE PRECISION, date TEXT,
        merchant_key TEXT NOT NULL, PRIMARY KEY (workspace_id, id)
    )""",
    """CREATE TABLE labels (
        workspace_id TEXT NOT NULL, id TEXT NOT NULL, item_id TEXT NOT NULL,
        label TEXT NOT NULL, source TEXT NOT NULL
          CHECK (source IN ('seed','answer','spot_check','muse_cold_start')),
        weight DOUBLE PRECISION NOT NULL CHECK (weight > 0 AND weight <= 1),
        position INTEGER NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (workspace_id, id), UNIQUE (workspace_id, position),
        FOREIGN KEY (workspace_id, item_id) REFERENCES items(workspace_id, id)
    )""",
    """CREATE TABLE predictions (
        workspace_id TEXT NOT NULL, id TEXT NOT NULL, item_id TEXT NOT NULL,
        model_version TEXT NOT NULL, label TEXT NOT NULL,
        prob DOUBLE PRECISION NOT NULL CHECK (prob >= 0 AND prob <= 1),
        auto BOOLEAN NOT NULL, threshold DOUBLE PRECISION, spot_check BOOLEAN NOT NULL,
        decision_json TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (workspace_id, id),
        FOREIGN KEY (workspace_id, item_id) REFERENCES items(workspace_id, id)
    )""",
    """CREATE TABLE answers (
        workspace_id TEXT NOT NULL, item_id TEXT NOT NULL, label TEXT NOT NULL,
        idempotency_key TEXT NOT NULL, label_id TEXT NOT NULL,
        result_json TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (workspace_id, idempotency_key),
        FOREIGN KEY (workspace_id, item_id) REFERENCES items(workspace_id, id),
        FOREIGN KEY (workspace_id, label_id) REFERENCES labels(workspace_id, id)
    )""",
    """CREATE TABLE model_versions (
        workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
        version TEXT NOT NULL, card_json TEXT NOT NULL, is_current BOOLEAN NOT NULL,
        PRIMARY KEY (workspace_id, version)
    )""",
    "CREATE INDEX predictions_created ON predictions(workspace_id, created_at)",
)


def initialize(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path, timeout=30)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN IMMEDIATE")
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"Database schema {version} is newer than {SCHEMA_VERSION}.")
            if version < 2:
                for statement in TABLES:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


def check_connection(database_path: Path) -> None:
    with closing(
        sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
    ) as connection:
        connection.execute("PRAGMA schema_version").fetchone()


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def encode(value) -> str:
    return json.dumps(value, allow_nan=False)


class IdempotencyConflict(ValueError):
    """A key was reused for a different answer."""


class Store:
    def __init__(self, database_path: Path):
        self.path = database_path
        initialize(database_path)

    @contextmanager
    def connect(self, *, write=False):
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection

    def create_workspace(self, workspace: str, config: dict) -> None:
        with self.connect(write=True) as db:
            db.execute(
                "INSERT INTO workspaces VALUES (?, ?, ?)", (workspace, encode(config), timestamp())
            )

    def workspace(self, workspace: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT config_json FROM workspaces WHERE workspace_id=?", (workspace,)
            ).fetchone()
        if row is None:
            raise KeyError(workspace)
        return json.loads(row[0])

    @staticmethod
    def _items(db, workspace, items):
        for item in items:
            values = (
                workspace,
                item["id"],
                item["text"],
                item.get("amount"),
                item.get("date"),
                item["merchant_key"],
            )
            old = db.execute(
                "SELECT * FROM items WHERE workspace_id=? AND id=?", values[:2]
            ).fetchone()
            if old is not None:
                if tuple(old) != values:
                    raise ValueError("An item ID cannot be reused with different content.")
            else:
                db.execute("INSERT INTO items VALUES (?, ?, ?, ?, ?, ?)", values)

    def put_items(self, workspace: str, items: list[dict]) -> None:
        with self.connect(write=True) as db:
            self._items(db, workspace, items)

    def item(self, workspace: str, item_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM items WHERE workspace_id=? AND id=?", (workspace, item_id)
            ).fetchone()
        if row is None:
            raise KeyError(item_id)
        return dict(row)

    @staticmethod
    def _label(db, workspace, item_id, label, source):
        if source not in SOURCES:
            raise ValueError(f"Unknown label source: {source}")
        position = db.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM labels WHERE workspace_id=?", (workspace,)
        ).fetchone()[0]
        label_id = uuid4().hex
        db.execute(
            "INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                workspace,
                label_id,
                item_id,
                label,
                source,
                0.3 if source == "muse_cold_start" else 1.0,
                position,
                timestamp(),
            ),
        )
        return label_id

    def add_label(self, workspace: str, item_id: str, label: str, source="seed") -> None:
        with self.connect(write=True) as db:
            self._label(db, workspace, item_id, label, source)

    def labels(self, workspace: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT l.*, i.text, i.amount, i.date, i.merchant_key FROM labels l
                JOIN items i ON i.workspace_id=l.workspace_id AND i.id=l.item_id
                WHERE l.workspace_id=? ORDER BY l.created_at, l.position""",
                (workspace,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_predictions(self, workspace: str, items: list[dict], decisions: list[dict]) -> None:
        if [row["id"] for row in items] != [row["item_id"] for row in decisions]:
            raise ValueError("Decisions must match the routed items in order.")
        with self.connect(write=True) as db:
            self._items(db, workspace, items)
            for decision in decisions:
                db.execute(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        workspace,
                        uuid4().hex,
                        decision["item_id"],
                        decision["model_version"],
                        decision["label"],
                        decision["prob"],
                        decision["auto"],
                        decision["threshold"],
                        decision.get("spot_check", False),
                        encode(decision),
                        timestamp(),
                    ),
                )

    def predictions(self, workspace: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT decision_json FROM predictions WHERE workspace_id=? ORDER BY created_at",
                (workspace,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def answer(self, workspace, item_id, label, idempotency_key, result, source="answer"):
        """Commit the answer and training label together; return the original result on replay."""
        if not idempotency_key:
            raise ValueError("An idempotency key is required.")
        if source not in ("answer", "spot_check"):
            raise ValueError("Answers must come from a person.")
        with self.connect(write=True) as db:
            old = db.execute(
                "SELECT * FROM answers WHERE workspace_id=? AND idempotency_key=?",
                (workspace, idempotency_key),
            ).fetchone()
            if old is not None:
                if (old["item_id"], old["label"]) != (item_id, label):
                    raise IdempotencyConflict("This key already belongs to a different answer.")
                return json.loads(old["result_json"]), False
            label_id = self._label(db, workspace, item_id, label, source)
            db.execute(
                "INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)",
                (workspace, item_id, label, idempotency_key, label_id, encode(result), timestamp()),
            )
        return result, True

    def set_models(self, workspace: str, versions: list[dict]) -> None:
        if sum(bool(row["is_current"]) for row in versions) > 1:
            raise ValueError("Only one model may be current.")
        with self.connect(write=True) as db:
            db.execute("DELETE FROM model_versions WHERE workspace_id=?", (workspace,))
            db.executemany(
                "INSERT INTO model_versions VALUES (?, ?, ?, ?)",
                [
                    (workspace, row["version"], encode(row["card"]), row["is_current"])
                    for row in versions
                ],
            )

    def models(self, workspace: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM model_versions WHERE workspace_id=? ORDER BY version", (workspace,)
            ).fetchall()
        return [
            dict(row)
            | {"card": json.loads(row["card_json"]), "is_current": bool(row["is_current"])}
            for row in rows
        ]
