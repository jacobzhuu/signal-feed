"""Durable online state; experiment data never enters this database."""

import os
import sqlite3
from contextlib import contextmanager
from .settings import ROOT

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS articles(id TEXT PRIMARY KEY,title TEXT NOT NULL,url TEXT NOT NULL,domain TEXT NOT NULL,published REAL NOT NULL,score INTEGER NOT NULL DEFAULT 0,deleted INTEGER NOT NULL DEFAULT 0,updated REAL NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS article_fts USING fts5(id UNINDEXED,title,domain);
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,interests TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,user_id TEXT NOT NULL,created REAL NOT NULL,strategy TEXT NOT NULL,model_version TEXT NOT NULL,items TEXT NOT NULL,query TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,request_id TEXT NOT NULL,user_id TEXT NOT NULL,item_id TEXT NOT NULL,kind TEXT NOT NULL,position INTEGER NOT NULL,created REAL NOT NULL,is_test INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS request_snapshots(request_id TEXT PRIMARY KEY REFERENCES requests(id),payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS user_events ON events(user_id,created);
CREATE INDEX IF NOT EXISTS item_events ON events(item_id,kind,user_id);
CREATE INDEX IF NOT EXISTS request_events ON events(request_id,is_test,kind,created);
CREATE INDEX IF NOT EXISTS request_time ON requests(created);
CREATE TABLE IF NOT EXISTS bookmarks(user_id TEXT NOT NULL,item_id TEXT NOT NULL,created REAL NOT NULL,PRIMARY KEY(user_id,item_id));
CREATE TABLE IF NOT EXISTS ingest_runs(started REAL PRIMARY KEY,finished REAL,count INTEGER,errors INTEGER);
"""


@contextmanager
def db(durable=True):
    conn = connection()
    # Serving request lists are reconstructible. Actual user events always use
    # FULL; their commit also flushes preceding request records in the same WAL.
    if not durable:
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def connection():
    return sqlite3.connect(
        os.environ.get("SIGNALFEED_DB", str(ROOT / "online.sqlite")), timeout=15
    )


def initialize():
    with db() as c:
        c.executescript(SCHEMA)
        # Migrate the old index without discarding existing events. Bookmark
        # transitions deduplicate by event UUID; impressions/clicks also by item.
        c.execute("BEGIN IMMEDIATE")
        c.execute("DROP INDEX IF EXISTS once_event")
        c.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS once_exposure_click "
            "ON events(request_id,item_id,kind) WHERE kind IN ('impression','click')"
        )


def cleanup(now, retention_days=90, batch_size=1000):
    """Bound snapshot retention; preserve events, bookmarks and their request rows."""
    if retention_days < 2:
        raise ValueError("Retention must cover the complete 24h label window")
    cutoff = now - retention_days * 86400
    with db() as c:
        removed = c.execute(
            "DELETE FROM request_snapshots WHERE request_id IN "
            "(SELECT r.id FROM requests r JOIN request_snapshots s ON s.request_id=r.id "
            "WHERE r.created<? ORDER BY r.created LIMIT ?)",
            (cutoff, batch_size),
        ).rowcount
        c.execute(
            "DELETE FROM requests WHERE id IN (SELECT r.id FROM requests r "
            "WHERE r.created<? AND NOT EXISTS(SELECT 1 FROM events e WHERE e.request_id=r.id) "
            "AND NOT EXISTS(SELECT 1 FROM request_snapshots s WHERE s.request_id=r.id) "
            "ORDER BY r.created LIMIT ?)",
            (cutoff, batch_size),
        )
    return removed
