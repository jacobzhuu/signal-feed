"""Synthetic audit contracts only; no fabricated production feedback."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest
from signalfeed.store import SCHEMA
from signalfeed.snapshots import pack
from signalfeed.temporal_audit import audit, load_protocol, phase_for, readonly, stamp

ROOT = Path(__file__).resolve().parents[1]


def observer():
    spec = importlib.util.spec_from_file_location(
        "audit_feedback", ROOT / "scripts/audit_feedback.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path):
    file = tmp_path / "source.sqlite"
    with sqlite3.connect(file) as c:
        c.executescript(SCHEMA)
    protocol = load_protocol(ROOT / "configs/temporal_v4.json")
    start = stamp(protocol["phases"]["select"]["start"]) + 3600
    user = next(
        "u" + str(i)
        for i in range(100)
        if phase_for("u" + str(i), start, protocol) == "select"
    )
    return file, protocol, start, user


def request(file, rid, user, time, history=None, items=("target",), snapshot=True):
    history = [] if history is None else history
    feature = np.zeros(1158, dtype="float32")
    feature[-3] = len(history) / 20
    feature[-1] = 0.4
    payload = dict(
        created=time,
        feature_version="online-text-v2",
        model_version="content-v1",
        history=history,
        interests=[],
        features={i: feature.tolist() for i in items},
        content_scores={i: 0.4 for i in items},
    )
    with sqlite3.connect(file) as c:
        c.execute("INSERT OR IGNORE INTO users VALUES(?,?)", (user, "[]"))
        c.execute(
            "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
            (rid, user, time, "content-v1", "content-v1", json.dumps(items), ""),
        )
        if snapshot:
            c.execute("INSERT INTO request_snapshots VALUES(?,?)", (rid, pack(payload)))


def event(file, rid, user, time, kind="click", item="target", position=0, test=0):
    with sqlite3.connect(file) as c:
        number = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        c.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
            (str(number), rid, user, item, kind, position, time, test),
        )


def prior(file, user, start, count=1):
    for i in range(count):
        rid = f"prior{i}"
        request(file, rid, user, start - 100 + i, items=("read",), snapshot=False)
        event(file, rid, user, start - 99 + i, item="read")


def test_empty_feedback_and_missing_snapshots_fail_closed_readonly(setup):
    file, p, start, u = setup
    request(file, "r", u, start, snapshot=False)
    before = hashlib.sha256(file.read_bytes()).hexdigest()
    result, rows = audit(file, start + 86401, p)
    assert result["inventory"]["users"] == 1
    assert result["requests"]["missing_snapshot"] == 1
    assert result["mature_exposed_users"] == 0 and rows == []
    assert not result["readiness"]["efficacy_authorized"]
    assert not result["readiness"]["full_catalog_replay_supported"]
    assert hashlib.sha256(file.read_bytes()).hexdigest() == before
    c = readonly(file)
    with pytest.raises(sqlite3.OperationalError):
        c.execute("DELETE FROM users")
    c.close()
    assert '"' + u + '"' not in json.dumps(result)


def test_maturity_click_inferred_exposure_and_late_boundary(setup):
    file, p, start, u = setup
    prior(file, u, start)
    request(file, "r", u, start, ["read"], items=("target", "late"))
    event(file, "r", u, start + 2)
    event(file, "r", u, start + 86400, item="late", position=1)
    immature, rows = audit(file, start + 86399, p)
    assert immature["mature_exposed_users"] == 0
    assert len(rows) == 1 and rows[0]["clicks"] is None
    mature, rows = audit(file, start + 86401, p)
    assert (
        mature["mature_exposed_users"] == 1
        and rows[0]["exposures"] == rows[0]["clicks"] == 1
    )
    assert mature["events"]["outside_24h_click"] == 1
    daily = mature["daily"][
        datetime.fromtimestamp(start, timezone.utc).date().isoformat() + "/1-5"
    ]
    assert daily["click_inferred_exposures"] == 1 and daily["viewable_exposures"] == 0


def test_reserve_before_feedback_never_replace_unexposed_request(setup):
    file, p, start, u = setup
    prior(file, u, start)
    request(file, "first", u, start, ["read"])
    request(file, "second", u, start + 10, ["read"])
    event(file, "second", u, start + 11)
    result, rows = audit(file, start + 86420, p)
    assert len(rows) == 1 and rows[0]["created"] == start
    assert rows[0]["exposures"] == 0
    assert result["mature_exposed_users"] == 1
    assert result["phases"]["select"]["1-5"]["mature_exposed_users"] == 0


def test_raw_history_cannot_be_manufactured_from_filtered_snapshot(setup):
    file, p, start, u = setup
    prior(file, u, start, 21)
    request(file, "r", u, start, ["read"])
    result, rows = audit(file, start + 86401, p)
    assert rows == []
    key = datetime.fromtimestamp(start, timezone.utc).date().isoformat() + "/11+"
    assert result["daily"][key]["compatible"] == 1
    assert result["requests"]["requests_with_repeated_positive_actions"] == 1


def test_future_history_and_corrupt_payload_are_rejected(setup):
    file, p, start, u = setup
    request(file, "r", u, start, ["not_seen"])
    request(file, "broken", u, start + 10)
    with sqlite3.connect(file) as c:
        c.execute(
            "UPDATE request_snapshots SET payload=? WHERE request_id=?",
            (b"SF2\0broken", "broken"),
        )
    result, rows = audit(file, start + 86420, p)
    assert result["requests"]["invalid_snapshot_history"] == 1
    assert result["requests"]["corrupt_snapshot"] == 1 and rows == []


def test_test_user_contamination_and_invalid_events(setup):
    file, p, start, u = setup
    prior(file, u, start)
    request(file, "r", u, start, ["read"])
    event(file, "r", u, start + 1, kind="impression", test=1)
    event(file, "r", "wrong-user", start + 2)
    result, rows = audit(file, start + 86403, p)
    assert result["source_test_users"] == 1 and result["events"]["test_events"] == 1
    assert result["events"]["invalid_event_identity_position_time"] == 1
    assert result["requests"]["test_contaminated_user_requests"] == 2 and rows == []


def test_partition_windows_and_purge_are_enforced(setup, tmp_path):
    _, p, start, u = setup
    assert phase_for(u, start, p) == "select"
    assert phase_for(u, stamp(p["phases"]["select"]["end"]), p) is None
    assert phase_for(u, stamp(p["phases"]["guard"]["start"]), p) is None
    p["phases"]["guard"]["start"] = p["phases"]["select"]["end"]
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="label windows"):
        load_protocol(file)


def test_capture_preserves_wal_and_refuses_overwrite_and_future_time(setup, tmp_path):
    file, p, start, u = setup
    # Keep a WAL connection open so a raw copy of the main file would be stale.
    c = sqlite3.connect(file)
    c.execute("PRAGMA journal_mode=WAL")
    request(file, "r", u, 1000000, snapshot=False)
    output = tmp_path / "audit"
    mod = observer()
    result = mod.collect(file, output, ROOT / "configs/temporal_v4.json", 1000001)
    assert result["inventory"]["requests"] == 1
    assert (output / "source.sqlite").stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "completed.json").exists()
    with pytest.raises(FileExistsError):
        mod.collect(file, output, ROOT / "configs/temporal_v4.json", 1000001)
    with pytest.raises(ValueError, match="future"):
        mod.collect(file, tmp_path / "future", ROOT / "configs/temporal_v4.json", 1e12)
    c.close()


def test_observer_stops_at_fixed_close_without_training(setup, tmp_path, monkeypatch):
    file, p, _, _ = setup
    mod = observer()
    end = stamp(p["close_at"]) + 1
    monkeypatch.setattr(mod.time, "time", lambda: end)
    calls = []
    monkeypatch.setattr(mod, "collect", lambda *args: calls.append(args))
    mod.watch(file, tmp_path / "watch", ROOT / "configs/temporal_v4.json")
    assert len(calls) == 1
    record = json.loads((tmp_path / "watch/observer.json").read_text())
    assert record["training"] is record["promotion"] is False
    tick = mod.next_tick(stamp("2026-10-31T23:59:59Z"))
    assert tick == stamp("2026-11-01T00:10:00Z")
