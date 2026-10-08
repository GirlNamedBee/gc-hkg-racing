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
    # Columns added after a database was first created.
    have = {r[1] for r in conn.execute("PRAGMA table_info(horses)")}
    for col in ("colour", "import_type", "sire", "dam", "dam_sire"):
        if col not in have:
            conn.execute(f"ALTER TABLE horses ADD COLUMN {col} TEXT")
    return conn
