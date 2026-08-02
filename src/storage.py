"""SQLite logging. One table for responses, one for rubric gradings.

Everything is INSERT OR IGNORE + 'already done' checks, so any phase can be
stopped and restarted: it simply skips work that is already saved.
"""
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .config import path

DRY_RUN = False  # set by run_pipeline.py --dry-run: use a separate db file
                 # so fake responses can never be mistaken for real ones

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    phase       TEXT NOT NULL,     -- baseline | exp1 | exp2 | exp1_mit | exp2_mit
    model       TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    condition   TEXT NOT NULL,     -- private | P1..P6 | G_k2..G_k4 | probe ...
    round       INTEGER NOT NULL DEFAULT 0,
    prompt      TEXT,
    response    TEXT,
    ts          TEXT,
    UNIQUE(phase, model, item_id, condition, round)
);
CREATE TABLE IF NOT EXISTS judgments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id   INTEGER NOT NULL,
    criterion_idx INTEGER NOT NULL,
    criterion     TEXT,
    weight        REAL,
    dimension     TEXT,
    met           INTEGER,         -- 1 yes | 0 no | -1 judge failed
    judge_model   TEXT,
    UNIQUE(response_id, criterion_idx, judge_model)
);
"""


def _conn():
    db_path = path("db")
    if DRY_RUN:
        db_path = db_path.with_name("dryrun_" + db_path.name)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # safe to stop mid-write
    return conn


def init_db():
    with _conn() as c:
        c.executescript(_SCHEMA)


def already_done(phase, model, item_id, condition, round_=0) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM responses WHERE phase=? AND model=? AND item_id=?"
            " AND condition=? AND round=?",
            (phase, model, item_id, condition, round_)).fetchone()
    return row is not None


def save_response(phase, model, item_id, condition, round_, prompt, response):
    """Returns the response id (existing one if this was a duplicate)."""
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO responses"
            " (phase, model, item_id, condition, round, prompt, response, ts)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (phase, model, item_id, condition, round_, prompt, response, ts))
        row = c.execute(
            "SELECT id FROM responses WHERE phase=? AND model=? AND item_id=?"
            " AND condition=? AND round=?",
            (phase, model, item_id, condition, round_)).fetchone()
    return row[0]


def save_judgment(response_id, criterion_idx, criterion, weight, dimension,
                  met, judge_model):
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO judgments"
            " (response_id, criterion_idx, criterion, weight, dimension, met,"
            "  judge_model) VALUES (?,?,?,?,?,?,?)",
            (response_id, criterion_idx, criterion, weight, dimension, met,
             judge_model))


def judgment_done(response_id, criterion_idx, judge_model) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM judgments WHERE response_id=? AND criterion_idx=?"
            " AND judge_model=?",
            (response_id, criterion_idx, judge_model)).fetchone()
    return row is not None


def get_response(phase, model, item_id, condition, round_=0):
    """Fetch one saved response -> dict(id, response) or None."""
    with _conn() as c:
        row = c.execute(
            "SELECT id, response FROM responses WHERE phase=? AND model=?"
            " AND item_id=? AND condition=? AND round=?",
            (phase, model, item_id, condition, round_)).fetchone()
    return {"id": row[0], "response": row[1]} if row else None


def to_dataframe(table: str) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query(f"SELECT * FROM {table}", c)
