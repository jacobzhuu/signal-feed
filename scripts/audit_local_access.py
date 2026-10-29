"""Bounded loopback/static audit using an existing empty session; no feed/events."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse
import httpx


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("src"):
            self.paths.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.paths.append(attrs["href"])


def inventory(connection):
    return {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("users", "requests", "events", "request_snapshots")
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", default="http://127.0.0.1:8765")
    p.add_argument("--dist", type=Path, default=Path("frontend/dist"))
    p.add_argument("--log", type=Path, required=True)
    args = p.parse_args()
    url = urlparse(args.base)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.path not in ("", "/")
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        raise ValueError("Audit is restricted to the existing loopback HTTP service")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        before = inventory(connection)
        # Reuse a pre-existing empty anonymous probe session, never a reading user.
        row = connection.execute(
            "SELECT u.id FROM users u WHERE interests='[]' "
            "AND NOT EXISTS(SELECT 1 FROM requests r WHERE r.user_id=u.id) "
            "AND NOT EXISTS(SELECT 1 FROM events e WHERE e.user_id=u.id) "
            "AND NOT EXISTS(SELECT 1 FROM bookmarks b WHERE b.user_id=u.id) "
            "ORDER BY u.id LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError(
                "No existing unused session; refusing to create audit users"
            )
        log_before = args.log.read_bytes()
        responses = []
        with httpx.Client(
            base_url=args.base.rstrip("/"),
            trust_env=False,
            timeout=15,
            follow_redirects=False,
            cookies={"signalfeed_session": row[0]},
            headers={"User-Agent": "SignalFeed-LocalAccessAudit/v7"},
        ) as client:

            def get(path):
                with client.stream("GET", path) as response:
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_bytes(65536):
                        data.extend(chunk)
                        if len(data) > 2 * 1024 * 1024:
                            raise ValueError("Response exceeds bounded audit limit")
                    if response.headers.get("set-cookie"):
                        raise ValueError("Audit unexpectedly received a new session")
                    responses.append(
                        dict(
                            path=path,
                            status=response.status_code,
                            bytes=len(data),
                            content_type=response.headers.get("content-type"),
                            sha256=sha(data),
                        )
                    )
                    return bytes(data)

            index = get("/")
            if index != (args.dist / "index.html").read_bytes():
                raise ValueError("Served HTML differs from expected dist")
            parser = Assets()
            parser.feed(index.decode())
            if not parser.paths or len(parser.paths) > 10:
                raise ValueError("Unexpected asset list")
            for asset in parser.paths:
                parsed = urlparse(asset)
                if (
                    parsed.scheme
                    or parsed.netloc
                    or not parsed.path.startswith("/assets/")
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError("Only same-origin static assets may be probed")
                local = (args.dist / parsed.path.lstrip("/")).resolve()
                if not local.is_relative_to(args.dist.resolve()):
                    raise ValueError("Asset path escapes dist")
                if get(asset) != local.read_bytes():
                    raise ValueError("Served asset differs from disk")
            health = json.loads(get("/api/health"))
            readiness = json.loads(get("/api/ready"))
            if (
                not health.get("ready")
                or health.get("test_events")
                or health.get("strategy") != "content-v1"
                or not readiness.get("ready")
            ):
                raise ValueError("Existing content service is not ready as expected")
        after = inventory(connection)
        log_after = args.log.read_bytes()
        counts = Counter(
            re.findall(r'"(GET|POST|PUT|DELETE) ([^ ?]+)', log_before.decode())
        )
        report = dict(
            observed_at_utc=datetime.now(timezone.utc).isoformat(),
            base=args.base,
            before=before,
            after=after,
            inventory_delta={k: after[k] - before[k] for k in before},
            existing_empty_session_reused=True,
            cookie_saved=False,
            browser_javascript_executed=False,
            requests=responses,
            health=health,
            all_static_bytes_match=True,
            log_before=dict(
                bytes=len(log_before),
                sha256=sha(log_before),
                requests={" ".join(k): v for k, v in counts.items()},
            ),
            log_after=dict(
                bytes=len(log_after),
                sha256=sha(log_after),
                prefix_unchanged=log_after.startswith(log_before),
            ),
            diagnostic_requests_only=True,
            feed_requested=False,
            events_posted=False,
            scope="Host loopback only; no claim of an external client or SSH tunnel being exercised",
            script_sha256=sha(Path(__file__).read_bytes()),
        )
        with (args.output / "audit.json").open("x") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        with (args.output / "failure.json").open("x") as f:
            json.dump(dict(error=type(exc).__name__, message=str(exc)), f, indent=2)
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
