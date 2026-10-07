"""SQLite connection helper."""
from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path


def connect(path: str | Path = "data/hkracing.db") -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(files("hkracing").joinpath("schema.sql").read_text())
    return conn
