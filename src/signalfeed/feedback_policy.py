"""Server-owned test identities; no database mutation or client-controlled flags."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class FeedbackPolicy:
    user_ids: frozenset[str]
    sha256: str

    def exclude_sql(self, column):
        """Return a predicate and bound parameters; callers supply a code identifier."""
        if not re.fullmatch(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?", column):
            raise ValueError("Invalid feedback-policy SQL column")
        ids = tuple(sorted(self.user_ids))
        if not ids:
            return "1=1", ()
        marks = ",".join("?" for _ in ids)
        return f"{column} NOT IN ({marks})", ids

    def metadata(self):
        # Never publish session credentials in health endpoints or model reports.
        return dict(version=1, sha256=self.sha256, test_user_count=len(self.user_ids))


def load_feedback_policy():
    """Read once per operation; configured missing/invalid files fail closed.

    Operators should replace the file atomically. A running learning call keeps
    one immutable policy through sample collection and reporting.
    """
    file = os.environ.get("SIGNALFEED_TEST_USERS_FILE")
    if not file:
        # Bind the default policy to the database, including CLI/audit consumers
        # that do not inherit the web server's environment. Explicit config wins.
        from .settings import ROOT

        database = Path(os.environ.get("SIGNALFEED_DB", str(ROOT / "online.sqlite")))
        sidecar = Path(str(database.resolve()) + ".feedback-policy.json")
        if sidecar.exists() or sidecar.is_symlink():
            file = sidecar
    data = json.loads(Path(file).read_text()) if file else {"version": 1, "users": []}
    if (
        not isinstance(data, dict)
        or set(data) != {"version", "users"}
        or type(data["version"]) is not int
        or data["version"] != 1
        or not isinstance(data["users"], list)
        or len(data["users"]) > 256
    ):
        raise ValueError(
            "Invalid test-user policy: expected version 1 and up to 256 users"
        )
    ids = set()
    for entry in data["users"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "reason"}
            or not isinstance(entry["id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", entry["id"])
            or not isinstance(entry["reason"], str)
            or not entry["reason"].strip()
            or entry["id"] in ids
        ):
            raise ValueError(
                "Invalid test-user entry: unique full ID and reason required"
            )
        ids.add(entry["id"])
    canonical = json.dumps(
        dict(version=1, users=sorted(data["users"], key=lambda e: e["id"])),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return FeedbackPolicy(frozenset(ids), hashlib.sha256(canonical).hexdigest())
