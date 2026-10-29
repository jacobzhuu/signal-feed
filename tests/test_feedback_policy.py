"""Isolation regressions on synthetic temporary databases; never train a model."""

import json
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signalfeed.feedback_policy import load_feedback_policy
from signalfeed.snapshots import pack
from signalfeed.store import db

TEST_USER = "a" * 32
REAL_USER = "b" * 32
PEER = "c" * 32


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.delenv("SIGNALFEED_TEST_USERS_FILE", raising=False)
    monkeypatch.setenv("SIGNALFEED_PREWARM", "0")
    import signalfeed.online_learning as learning

    def forbidden(*args, **kwargs):
        pytest.fail("Isolation tests must not invoke training")

    monkeypatch.setattr(learning, "retrain", forbidden)
    monkeypatch.setattr(learning, "_retrain", forbidden)


def configure(tmp_path, monkeypatch, users=(TEST_USER,)):
    file = tmp_path / "test-users.json"
    file.write_text(
        json.dumps(
            {
                "version": 1,
                "users": [
                    {"id": uid, "reason": "synthetic regression"} for uid in users
                ],
            }
        )
    )
    monkeypatch.setenv("SIGNALFEED_TEST_USERS_FILE", str(file))
    return file


def seed_request(uid, rid, stamp, cold=False):
    version = "coldfuse-test" if cold else "content-v1"
    snapshot = dict(
        created=stamp,
        model_version=version,
        feature_version="cold-semantic-v1" if cold else "online-text-v2",
        history=[],
        interests=[],
        content_scores={"0": 0.2, "1": 0.1},
        features={i: [0.0] * 1158 for i in ("0", "1")},
    )
    with db() as c:
        c.execute("INSERT OR IGNORE INTO users VALUES(?, '[]')", (uid,))
        c.execute(
            "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
            (rid, uid, stamp, version, version, '["0","1"]', ""),
        )
        c.execute("INSERT INTO request_snapshots VALUES(?,?)", (rid, pack(snapshot)))
        for kind, item in [("impression", "0"), ("impression", "1"), ("click", "0")]:
            c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                (f"{rid}-{kind}-{item}", rid, uid, item, kind, int(item), stamp + 1, 0),
            )


def contents():
    with db() as c:
        return {
            name: [tuple(r) for r in c.execute(f"SELECT * FROM {name} ORDER BY rowid")]
            for name in (
                "users",
                "requests",
                "events",
                "bookmarks",
                "request_snapshots",
            )
        }


@pytest.mark.parametrize("cold", [False, True])
def test_legacy_feedback_excluded_before_train_validation_and_preserved(
    online_env,
    tmp_path,
    monkeypatch,
    cold,
):
    from signalfeed.online_learning import feedback_records, chronological_split

    for i in range(6):
        for uid in (TEST_USER, REAL_USER):
            seed_request(uid, f"{uid}-{i}", 1000000 + i * 86400, cold)
    with db() as c:
        c.execute("INSERT INTO bookmarks VALUES(?,?,?)", (TEST_USER, "0", 1000001))
    before = contents()
    baseline, _ = feedback_records(2000000, cold=cold)
    configure(tmp_path, monkeypatch)
    records, excluded = feedback_records(2000000, cold=cold)
    assert len(records) == 12
    assert excluded == {"test_user_requests": 6}
    assert {r[0]["user_id"] for r in records} == {REAL_USER}
    expected = [r for r in baseline if r[0]["user_id"] == REAL_USER]
    for actual, original in zip(records, expected):
        assert actual[0] == original[0] and actual[2] == original[2]
        np.testing.assert_array_equal(actual[1], original[1])
    train, valid, _ = chronological_split(records)
    assert train and valid
    assert all(r[0]["user_id"] == REAL_USER for r in train + valid)
    # Policy rollback affects filtering only, never rewrites old flags/snapshots.
    monkeypatch.delenv("SIGNALFEED_TEST_USERS_FILE")
    assert len(feedback_records(2000000, cold=cold)[0]) == len(baseline)
    assert contents() == before


def catalog(monkeypatch):
    from signalfeed.serving import ContentService

    service = ContentService()
    service.items = [
        dict(
            id=str(i),
            title=f"Article {i}",
            score=i,
            published=1000000,
            url="https://example.com",
            domain="example.com",
        )
        for i in range(4)
    ]
    service.vectors = np.eye(384, dtype="float32")[:4]
    service.index = {str(i): i for i in range(4)}
    monkeypatch.setattr(service, "refresh", lambda *a: None)
    monkeypatch.setattr(service, "active_ranker", lambda: None)
    monkeypatch.setattr(service, "encode_query", lambda text: service.vectors[0])
    with db() as c:
        for a in service.items:
            c.execute(
                "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)",
                (
                    a["id"],
                    a["title"],
                    a["url"],
                    a["domain"],
                    a["published"],
                    a["score"],
                    0,
                    1000000,
                ),
            )
    return service


def click(uid, item):
    with db() as c:
        c.execute("INSERT OR IGNORE INTO users VALUES(?, '[]')", (uid,))
        rid = str(uuid.uuid4())
        c.execute(
            "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
            (rid, uid, 1000000, "content-v1", "content-v1", json.dumps([item]), ""),
        )
        c.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
            (rid, rid, uid, item, "click", 0, 1000001, 0),
        )


@pytest.mark.parametrize("cold", [False, True])
def test_test_peer_cannot_change_other_users_collaborative_recall(
    online_env,
    tmp_path,
    monkeypatch,
    cold,
):
    import signalfeed.cold_serving as cs
    import signalfeed.cold_worker as worker
    import signalfeed.serving as serving
    from types import SimpleNamespace

    service = catalog(monkeypatch)
    clock = SimpleNamespace(time=lambda: 1000100)
    monkeypatch.setattr(serving, "time", clock)
    monkeypatch.setattr(cs, "time", clock)
    monkeypatch.setattr(cs, "active_ranker", lambda: None)
    from signalfeed.background import BoundedRunner

    monkeypatch.setattr(cs, "research_jobs", BoundedRunner(8))
    cf_inputs = []

    def research(payload, timeout=2):
        cf_inputs.append(payload["cf_scores"].copy())
        return None, "unavailable-fallback", None

    monkeypatch.setattr(worker, "recommend", research)
    click(REAL_USER, "0")
    click(PEER, "0")
    click(PEER, "1")
    configure(tmp_path, monkeypatch)

    def recommend(uid):
        snapshot = {}
        rows = (
            cs.recommend(service, uid, 4, snapshot)
            if cold
            else service.recommend(uid, snapshot=snapshot)
        )
        return rows, snapshot

    before_rows, before = recommend(REAL_USER)
    click(TEST_USER, "0")
    click(TEST_USER, "2")
    after_rows, after = recommend(REAL_USER)
    assert before_rows == after_rows
    assert before == after
    if cold:
        np.testing.assert_array_equal(cf_inputs[0], cf_inputs[1])
        assert cf_inputs[1][1] > 0 and cf_inputs[1][2] == 0
    # The test identity retains its own browsing history and recommendation ability.
    own_rows, own = recommend(TEST_USER)
    assert own_rows and set(own["history"]) == {"0", "2"}
    monkeypatch.delenv("SIGNALFEED_TEST_USERS_FILE")
    _, unfiltered = recommend(REAL_USER)
    if cold:
        assert cf_inputs[-1][2] > 0
    else:
        assert unfiltered["content_scores"]["2"] > after["content_scores"]["2"]


def test_api_marks_future_events_keeps_personal_features_and_filters_stats(
    online_env,
    tmp_path,
    monkeypatch,
):
    import signalfeed.api as api
    import signalfeed.cold_serving as cs
    import signalfeed.cold_worker as worker

    service = catalog(monkeypatch)
    monkeypatch.setattr(api, "service", service)
    monkeypatch.setattr(cs, "active_ranker", lambda: None)
    monkeypatch.setattr(
        worker,
        "recommend",
        lambda payload, timeout=2: (None, "unavailable-fallback", None),
    )
    for uid in (TEST_USER, REAL_USER):
        seed_request(uid, uid, time.time() - 90000)
        with db() as c:
            c.execute("INSERT INTO bookmarks VALUES(?,?,?)", (uid, "0", 1000001))
    original = contents()
    configure(tmp_path, monkeypatch)
    with TestClient(api.app) as client:
        stats = client.get("/api/stats").json()
        assert (
            stats["impressions"],
            stats["clicked_impressions"],
            stats["bookmarks"],
        ) == (2, 1, 1)
        for uid, flag in ((TEST_USER, 1), (REAL_USER, 0)):
            client.cookies.set("signalfeed_session", uid)
            assert (
                client.put("/api/interests", json={"topics": ["AI"]}).status_code == 200
            )
            assert client.get("/api/interests").json() == {"topics": ["AI"]}
            for url in ("/api/feed", "/api/feed?cold=true", "/api/search?q=AI"):
                response = client.get(url)
                assert response.status_code == 200
                feed = response.json()
                item = feed["items"][0]
                for kind in (
                    "impression",
                    "click",
                    "bookmark",
                    "unbookmark",
                    "bookmark",
                ):
                    eid = str(uuid.uuid4())
                    body = dict(
                        event_id=eid,
                        request_id=feed["request_id"],
                        item_id=item["id"],
                        position=0,
                        kind=kind,
                        is_test=not flag,
                    )
                    assert client.post("/api/events", json=body).json()["inserted"]
                    assert not client.post("/api/events", json=body).json()["inserted"]
                    with db() as c:
                        assert (
                            c.execute(
                                "SELECT is_test FROM events WHERE id=?", (eid,)
                            ).fetchone()[0]
                            == flag
                        )
                saved = client.get("/api/bookmarks").json()
                assert item["id"] in [a["id"] for a in saved["items"]]
        # The pre-configuration records remain unchanged, even after real API writes.
        after = contents()
        for table in ("events", "requests", "request_snapshots"):
            assert all(row in after[table] for row in original[table])
        monkeypatch.setenv("SIGNALFEED_TEST_EVENTS", "1")
        feed = client.get("/api/feed").json()
        eid = str(uuid.uuid4())
        client.post(
            "/api/events",
            json=dict(
                event_id=eid,
                request_id=feed["request_id"],
                item_id=feed["items"][0]["id"],
                kind="click",
                position=0,
            ),
        )
        with db() as c:
            assert (
                c.execute("SELECT is_test FROM events WHERE id=?", (eid,)).fetchone()[0]
                == 1
            )


@pytest.mark.parametrize(
    "data",
    [
        {"version": 2, "users": []},
        {"version": 1, "users": [{"id": "aaaaaaaa", "reason": "prefix forbidden"}]},
        {"version": 1, "users": [{"id": TEST_USER, "reason": ""}]},
        {"version": 1, "users": [{"id": TEST_USER, "reason": "x"}] * 2},
    ],
)
def test_invalid_policy_rejected(tmp_path, monkeypatch, data):
    file = configure(tmp_path, monkeypatch)
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_feedback_policy()


def test_missing_or_broken_configuration_does_not_fall_back(
    online_env, tmp_path, monkeypatch
):
    from signalfeed.online_learning import feedback_records

    file = configure(tmp_path, monkeypatch)
    old = load_feedback_policy()
    configure(tmp_path, monkeypatch, (TEST_USER, REAL_USER))
    current = load_feedback_policy()
    assert old.user_ids == {TEST_USER} and current.user_ids == {TEST_USER, REAL_USER}
    assert old.sha256 != current.sha256
    assert TEST_USER not in json.dumps(current.metadata())
    file.unlink()
    with pytest.raises(FileNotFoundError):
        feedback_records(2000000)
    file.write_text("{")
    with pytest.raises(ValueError):
        feedback_records(2000000)


def test_database_policy_applies_without_service_environment(tmp_path, monkeypatch):
    database = tmp_path / "shared.sqlite"
    monkeypatch.setenv("SIGNALFEED_DB", str(database))
    explicit = configure(tmp_path, monkeypatch)
    expected = load_feedback_policy()
    sidecar = tmp_path / "shared.sqlite.feedback-policy.json"
    sidecar.symlink_to(explicit)
    monkeypatch.delenv("SIGNALFEED_TEST_USERS_FILE")
    assert load_feedback_policy() == expected
    explicit.unlink()
    with pytest.raises(FileNotFoundError):
        load_feedback_policy()
