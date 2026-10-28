import hashlib
import json
from types import SimpleNamespace
import uuid

import pytest
import torch
from fastapi.testclient import TestClient
from signalfeed.store import db, initialize
from signalfeed.ranker import Ranker
import signalfeed.online_learning as learning
import signalfeed.serving as serving
import numpy as np
from signalfeed.online_learning import online_matrix
from signalfeed.online_schema import CONTRACT, policy_scores
from signalfeed.snapshots import unpack


def test_cold_user_features_are_finite_and_history_changes_interactions():
    vectors = np.eye(384, dtype="float32")[:3]
    cold = online_matrix(vectors, [0, 1], [])
    warm = online_matrix(vectors, [0, 1], [0])
    assert cold.shape == (2, 1158)
    assert np.isfinite(cold).all()
    assert cold[0, -6] == 0
    assert warm[0, -6] == 1
    assert warm[1, -6] == 0
    assert np.all(cold[:, -2:] == 0)


def test_bad_model_publication_falls_back_and_rollback_clears_model(
    tmp_path, monkeypatch
):
    import signalfeed.serving as serving

    monkeypatch.setattr(serving, "path", lambda name: tmp_path)
    service = serving.ContentService()
    marker = tmp_path / "active.json"
    marker.write_text("{invalid")
    assert service.active_ranker() is None
    previous = object()
    service.online_ranker = previous
    service.online_version = "old"
    assert service.active_ranker() is previous
    marker.unlink()
    assert service.active_ranker() is None
    assert service.online_version is None


@pytest.fixture
def online_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALFEED_DB", str(tmp_path / "online.sqlite"))
    monkeypatch.setenv("SIGNALFEED_TEST_EVENTS", "0")
    initialize()

    def path(name):
        dest = tmp_path / name
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr(learning, "path", path)
    monkeypatch.setattr(serving, "path", path)
    return path


def seed_feedback(count=150, *, start=1000000, step=86400, predictive=True):
    """Isolated synthetic fixture, never put in the real feedback database."""
    vectors = np.eye(384, dtype="float32")[:7]
    features = online_matrix(vectors, [0, 1], list(range(2, 7)))
    with db() as c:
        for i in range(count):
            stamp = start + i * step
            uid, rid = f"u{i}", f"r{i}"
            positive = 0 if predictive else i % 2
            c.execute(
                "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
                (rid, uid, stamp, "content-v1", "content-v1", '["0","1"]', ""),
            )
            snapshot = dict(
                created=stamp,
                **CONTRACT,
                model_version="content-v1",
                history=[str(j) for j in range(2, 7)],
                interests=[],
                features={"0": features[0].tolist(), "1": features[1].tolist()},
                content_scores={"0": 0.0, "1": 1.0},
            )
            c.execute(
                "INSERT INTO request_snapshots VALUES(?,?)", (rid, json.dumps(snapshot))
            )
            # Click arrives before both delayed exposure events.
            c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                (f"c{i}", rid, uid, str(positive), "click", positive, stamp + 0.1, 0),
            )
            for j in range(2):
                c.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                    (f"e{i}-{j}", rid, uid, str(j), "impression", j, stamp + 1, 0),
                )
    return start + count * step + 86400


def publish_ranker(path, preferred=0, version="incumbent", data_until=0):
    model = Ranker()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[0].weight[0, preferred] = 1
        model.net[3].weight[0, 0] = 1
        model.net[5].weight[0, 0] = 1
    dest = path("online-models/" + version)
    torch.save(model.state_dict(), dest / "model.pt")
    manifest = dict(
        version=version,
        **CONTRACT,
        data_until=data_until,
        model_sha256=hashlib.sha256((dest / "model.pt").read_bytes()).hexdigest(),
    )
    (path("online-models") / "active.json").write_text(json.dumps(manifest))
    return model.eval()


def test_request_snapshot_survives_early_click_delayed_impression_and_catalog_change(
    online_env, monkeypatch
):
    import signalfeed.api as api

    clock = [1000000.0]
    for module in [api, serving]:
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0]))
    service = serving.ContentService()
    service.items = [
        dict(
            id=str(i),
            title=f"Item {i}",
            url="https://example.com",
            domain="example.com",
            published=clock[0],
            score=i,
            deleted=0,
        )
        for i in range(2)
    ]
    service.vectors = np.eye(384, dtype="float32")[:2]
    service.index = {"0": 0, "1": 1}
    monkeypatch.setattr(service, "refresh", lambda *a: None)
    monkeypatch.setattr(api, "service", service)
    with TestClient(api.app) as client:
        feed = client.get("/api/feed").json()
        iid = feed["items"][0]["id"]
        with db() as c:
            original = c.execute("SELECT payload FROM request_snapshots").fetchone()[0]
        payload = unpack(original)
        assert payload["history"] == []
        assert payload["features"][iid][-6] == 0
        assert payload["content_scores"][iid] > 0
        body = dict(request_id=feed["request_id"], item_id=iid, position=0)
        clock[0] += 0.1
        assert client.post(
            "/api/events", json=dict(body, kind="click", event_id=str(uuid.uuid4()))
        ).json()["inserted"]
        clock[0] += 0.9
        assert client.post(
            "/api/events",
            json=dict(body, kind="impression", event_id=str(uuid.uuid4())),
        ).json()["inserted"]
        # Later preference edits/catalog drift must not rewrite old features.
        client.put("/api/interests", json={"topics": ["new interest"]})
        service.vectors[:] = 0
        records, excluded = learning.feedback_records(1000000 + 86401)
        assert excluded == {}
        assert len(records) == 1 and records[0][2] == 1
        np.testing.assert_array_equal(records[0][1], payload["features"][iid])
        assert records[0][1][-6] == 0
        with db() as c:
            assert (
                c.execute("SELECT payload FROM request_snapshots").fetchone()[0]
                == original
            )


def test_fixed_label_window_excludes_late_clicks_tests_and_legacy_requests(online_env):
    now = seed_feedback(6)
    with db() as c:
        c.execute("UPDATE events SET created=1000000+86400 WHERE id='c0'")
        c.execute("UPDATE events SET is_test=1 WHERE request_id='r1'")
        c.execute("DELETE FROM request_snapshots WHERE request_id='r2'")
        c.execute("UPDATE requests SET strategy='bookmarks' WHERE id='r3'")
        c.execute("UPDATE events SET created=1000000+4*86400+86400 WHERE id='e4-0'")
    records, excluded = learning.feedback_records(now)
    assert all(r[2] == 0 for r in records if r[0]["request_id"] == "r0")
    assert not any(r[0]["request_id"] in {"r1", "r2", "r3"} for r in records)
    assert excluded == {
        "missing_snapshot": 2,
        "non_feed": 2,
        "outside_window_or_invalid_exposure": 1,
    }
    # Requests remain immature until the complete label window closes.
    assert learning.feedback_records(1000000 + 86399)[0] == []


def test_temporal_split_purges_label_overlap_and_keeps_timestamp_groups(online_env):
    now = seed_feedback(150, step=3600)
    records, _ = learning.feedback_records(now)
    # Include two requests at exactly the same boundary timestamp.
    for row in records:
        if row[0]["request_id"] == "r119":
            row[0]["request_created"] = 1000000 + 120 * 3600
            row[0]["label_end"] = row[0]["request_created"] + 86400
    train, valid, purged = learning.chronological_split(records)
    assert purged > 0
    assert max(r[0]["label_end"] for r in train) <= min(
        r[0]["request_created"] for r in valid
    )
    assert not (
        {r[0]["request_id"] for r in train} & {r[0]["request_id"] for r in valid}
    )
    assert {"r119", "r120"} <= {r[0]["request_id"] for r in valid}
    assert not any(r[0]["request_id"] in {"r119", "r120"} for r in train)
    cutoff = min(r[0]["request_created"] for r in valid) + 2 * 3600
    _, new_valid, _ = learning.chronological_split(records, cutoff)
    assert min(r[0]["request_created"] for r in new_valid) > cutoff


def test_promotion_compares_complete_content_scores_and_requires_twenty_requests(
    online_env,
):
    now = seed_feedback(predictive=False)
    records, _ = learning.feedback_records(now)
    _, valid, _ = learning.chronological_split(records)
    candidate = [r[2] for r in valid]
    complete_content = [r[2] for r in valid]
    a, b, count, promote, _ = learning.compare_policies(
        valid, candidate, complete_content
    )
    assert count == 30 and a["auc"] == b["auc"] == 1
    assert not promote  # A weak single feature would allow this candidate.
    weak = [r[1][-6] for r in valid]
    assert learning.compare_policies(valid, candidate, weak)[3]
    assert not learning.compare_policies(valid[:38], candidate[:38], weak[:38])[3]


@pytest.mark.parametrize(
    "incumbent_preferred,expected",
    [(None, "promoted"), (0, "kept_for_research"), (1, "promoted")],
)
def test_real_training_and_promotion_against_current_deployed_ranker(
    online_env, monkeypatch, incumbent_preferred, expected
):
    now = seed_feedback()
    monkeypatch.setattr(
        learning,
        "time",
        SimpleNamespace(time=lambda: now, time_ns=lambda: int(now * 1e9)),
    )
    if incumbent_preferred is not None:
        publish_ranker(online_env, preferred=incumbent_preferred)
    report = learning.retrain(min_exposures=100, min_clicks=50)
    assert report["status"] == expected
    assert report["train_label_end"] <= report["valid_request_start"]
    assert report["dnn"]["auc"] == 1
    assert report["incumbent"]["auc"] == (1 if incumbent_preferred == 0 else 0)
    assert report["incumbent_version"] == (
        "content-v1" if incumbent_preferred is None else "incumbent"
    )
    active = json.loads((online_env("online-models") / "active.json").read_text())
    assert active["version"] == (
        "incumbent" if expected == "kept_for_research" else report["version"]
    )
    service = serving.ContentService()
    assert service.active_ranker() is not None
    assert service.online_version == active["version"]
    # Reusing an incumbent's validation period cannot authorize another promotion.
    if expected == "promoted":
        again = learning.retrain(min_exposures=100, min_clicks=50)
        assert again["status"] == "insufficient_chronological_feedback"


def test_unknown_incumbent_provenance_blocks_publication(online_env):
    seed_feedback()
    publish_ranker(online_env)
    pointer = online_env("online-models") / "active.json"
    manifest = json.loads(pointer.read_text())
    del manifest["data_until"]
    pointer.write_text(json.dumps(manifest))
    before = pointer.read_bytes()
    assert learning.retrain(100, 50)["status"] == "incumbent_provenance_unavailable"
    assert pointer.read_bytes() == before


def test_incumbent_change_during_training_blocks_publication(online_env, monkeypatch):
    seed_feedback()
    compare = learning.compare_policies

    def changed(*args):
        result = compare(*args)
        publish_ranker(online_env, version="new-incumbent")
        return result

    monkeypatch.setattr(learning, "compare_policies", changed)
    report = learning.retrain(100, 50)
    assert report["policy_changed_during_training"]
    assert report["status"] == "kept_for_research"
    assert (
        json.loads((online_env("online-models") / "active.json").read_text())["version"]
        == "new-incumbent"
    )


def test_serving_freezes_full_policy_and_exact_online_inputs(online_env, monkeypatch):
    clock = 1000000.0
    monkeypatch.setattr(serving, "time", SimpleNamespace(time=lambda: clock))
    service = serving.ContentService()
    service.items = [
        dict(id=str(i), title=f"item {i}", published=clock, score=i) for i in range(3)
    ]
    service.vectors = np.eye(384, dtype="float32")[:3]
    service.index = {str(i): i for i in range(3)}
    monkeypatch.setattr(service, "refresh", lambda: None)
    monkeypatch.setattr(service, "encode_query", lambda text: service.vectors[1])
    with db() as c:
        c.execute("INSERT INTO users VALUES(?,?)", ("user", '["interest"]'))
        # Read penalty applies to item 0; interest boosts item 1; a peer's
        # co-click boosts item 2. A future click must not enter either history.
        for name, user, item, stamp in [
            ("h", "user", "0", clock - 3),
            ("peer1", "peer", "0", clock - 2),
            ("peer2", "peer", "2", clock - 1),
            ("future", "user", "2", clock + 1),
        ]:
            c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                (name, name, user, item, "click", 0, stamp, 0),
            )
    snapshot = {}
    rows = service.recommend("user", snapshot=snapshot)
    assert snapshot["history"] == ["0"]
    expected = {
        "0": 0.6 / np.sqrt(2) + 0.15 - 0.4 + 0.15,
        "1": 0.6 / np.sqrt(2) + 0.25 * np.log(2) / np.log(3) + 0.15,
        "2": 0.25 + 0.15 + 0.15,
    }
    assert snapshot["content_scores"] == pytest.approx(expected)
    assert [r["id"] for r in rows] == sorted(expected, key=expected.get, reverse=True)
    publish_ranker(online_env, preferred=0)
    snapshot = {}
    rows = service.recommend("user", snapshot=snapshot)
    assert snapshot["model_version"] == "online-incumbent"
    model = service.active_ranker()
    with torch.inference_mode():
        scores = policy_scores(
            model,
            np.array([snapshot["features"][r["id"]] for r in rows], dtype="float32"),
            [snapshot["content_scores"][r["id"]] for r in rows],
        )
    assert list(scores) == sorted(scores, reverse=True)
    assert snapshot["content_scores"] == pytest.approx(expected)


def test_invalid_snapshot_and_search_are_excluded(online_env):
    now = seed_feedback(3)
    with db() as c:
        payload = json.loads(
            c.execute(
                "SELECT payload FROM request_snapshots WHERE request_id='r0'"
            ).fetchone()[0]
        )
        payload["feature_version"] = "unknown"
        c.execute(
            "UPDATE request_snapshots SET payload=? WHERE request_id='r0'",
            (json.dumps(payload),),
        )
        c.execute("UPDATE requests SET query='search' WHERE id='r1'")
        payload = json.loads(
            c.execute(
                "SELECT payload FROM request_snapshots WHERE request_id='r2'"
            ).fetchone()[0]
        )
        payload["features"]["0"] = [0]
        c.execute(
            "UPDATE request_snapshots SET payload=? WHERE request_id='r2'",
            (json.dumps(payload),),
        )
    records, excluded = learning.feedback_records(now)
    assert len(records) == 1
    assert excluded == {
        "incompatible_snapshot": 2,
        "non_feed": 2,
        "invalid_features": 1,
    }
