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
