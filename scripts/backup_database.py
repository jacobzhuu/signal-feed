"""Consistent SQLite backup, including WAL; refuses overwrite and validates output."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3


def backup(source, destination):
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        src.execute("PRAGMA query_only=ON")
        with sqlite3.connect(destination) as dest:
            src.backup(dest)
            check = dest.execute("PRAGMA integrity_check").fetchall()
            if check != [("ok",)]:
                raise RuntimeError(f"Backup integrity failed: {check}")
            counts = {
                table: dest.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                for table in (
                    "users",
                    "requests",
                    "events",
                    "bookmarks",
                    "request_snapshots",
                    "articles",
                )
            }
    result = dict(
        file=str(destination),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        counts=counts,
        integrity="ok",
    )
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(backup(args.source, args.destination)))
