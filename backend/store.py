import sqlite3
from contextlib import closing
from pathlib import Path


def initialize(database_path: Path) -> None:
    """Create local storage at startup; later migrations will add decision tables."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA user_version=1")


def check_connection(database_path: Path) -> None:
    # Read-only mode makes a missing database a failure instead of creating an empty one.
    with closing(
        sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
    ) as connection:
        connection.execute("PRAGMA schema_version").fetchone()
