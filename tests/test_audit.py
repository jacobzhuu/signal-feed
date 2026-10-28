import hashlib
import json
import multiprocessing
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse
from fastapi.testclient import TestClient

from test_online import publish_ranker, seed_feedback
from signalfeed.retrieval import RetrievalPolicy, adaptive, fuse, top
from signalfeed.features import bounded_itemcf
from signalfeed.online_schema import online_matrix, policy_scores
from signalfeed.snapshots import pack, unpack
from signalfeed.store import db, cleanup
from signalfeed.serving import ContentService
from signalfeed.publication import publication_lock
import signalfeed.online_learning as learning


def test_no_signal_channels_are_empty_and_ties_have_explicit_order():
    f = SimpleNamespace(
        codes=np.zeros((8, 3), dtype=int),
        index={},
        pop=np.arange(8),
        history=lambda r: [],
        scores=lambda r: (np.zeros(8), np.zeros(8)),
    )
    g = SimpleNamespace(generate=lambda *a, **k: [((0, 0, 0), 0)])
    policy = RetrievalPolicy(f, g, {"threshold": 5, "weight": 1.5})
    result = policy.candidates({"history": []}, np.ones(8, dtype=bool))
    assert result["content"] == result["itemcf"] == []
    assert result["generative_prefix"] == [7, 6, 5, 4, 3]
    assert top(np.ones(10), np.ones(10, dtype=bool), 3) == [0, 1, 2]
    assert top(np.ones(10), np.ones(10, dtype=bool), 0) == []


def test_ablation_preserves_all_other_weights():
    lists = {
        "popular": [1, 2, 3],
        "content": [3, 1, 2],
        "itemcf": [2, 1],
        "generated": [4, 3],
    }
    for n, popular in [(0, 1.5), (20, 0.5)]:
        got = adaptive(lists, n, {"threshold": 10, "weight": 2}, generated=False)
        expected = fuse(
            {k: v for k, v in lists.items() if k != "generated"},
            weights={"popular": popular},
        )
        assert got == expected
        assert 4 not in got


def test_blocked_cf_matches_full_cosine_with_deterministic_neighbor_limit():
    rng = np.random.default_rng(1)
    mat = sparse.csr_matrix((rng.random((25, 17)) > 0.7).astype("float32"))
    full = (mat.T @ mat).toarray()
    norm = np.sqrt(np.asarray(mat.sum(0)).ravel()).clip(1)
    full /= norm[:, None] * norm[None, :]
    np.fill_diagonal(full, 0)
    for i in range(len(full)):
        keep = np.lexsort((np.arange(len(full)), -full[i]))[:4]
        full[i, np.setdiff1d(np.arange(len(full)), keep)] = 0
    np.testing.assert_allclose(
        bounded_itemcf(mat, block_size=3, neighbors=4).toarray(), full
    )


def test_explicit_interests_change_features_and_cold_policy_preserves_them():
    vectors = np.eye(384, dtype="float32")[:8]
    a = online_matrix(vectors, [0, 1], [], vectors[0], [1, 0])
    b = online_matrix(vectors, [0, 1], [], vectors[1], [0, 1])
    assert not np.array_equal(a, b)

    def bad_model(x):
        return torch.ones(len(x)) * 100

    for history in [[], [2], [2, 3, 4, 5]]:
        x = online_matrix(vectors, [0, 1], history, vectors[1], [0, 1])
        np.testing.assert_array_equal(policy_scores(bad_model, x, [0, 1]), [0, 1])


@pytest.mark.parametrize("defect", ["schema", "corrupt", "nan", "encoder", "dimension"])
def test_incompatible_or_corrupt_models_keep_last_good(online_env, defect):
    publish_ranker(online_env)
    service = ContentService()
    old = service.active_ranker()
    assert old is not None
    pointer = online_env("online-models") / "active.json"
    manifest = json.loads(pointer.read_text())
    file = online_env("online-models/incumbent") / "model.pt"
    if defect == "schema":
        manifest["feature_version"] = "unknown"
    elif defect == "encoder":
        manifest["encoder_revision"] = "another"
    elif defect == "dimension":
        manifest["input_dim"] = 99
    elif defect == "corrupt":
        file.write_bytes(b"not a checkpoint")
        manifest["model_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    else:
        state = {k: v.clone() for k, v in old.state_dict().items()}
        state["net.0.weight"][0, 0] = float("nan")
        torch.save(state, file)
        manifest["model_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(manifest))
    assert service.active_ranker() is old
    assert service.load_error


def test_catalog_failure_keeps_complete_previous_snapshot(online_env):
    service = ContentService()
    service.items = [dict(id="old", title="old", score=1, published=time.time())]
    service.vectors = np.eye(384, dtype="float32")[:1]
    service.index = {"old": 0}
    service.refresh_at = time.time()
    original = service.items, service.vectors, service.index

    def fail(*a, **k):
        raise RuntimeError("encoder failure")

    service.model = SimpleNamespace(encode=fail)
    with db() as c:
        c.execute(
            "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)",
            (
                "new",
                "new",
                "https://example.com",
                "example.com",
                time.time(),
                1,
                0,
                time.time(),
            ),
        )
    service.refresh(force=True)
    assert (
        service.items is original[0]
        and service.vectors is original[1]
        and service.index is original[2]
    )
    assert service.refresh_error
    assert service.recommend("cold", snapshot={})[0]["id"] == "old"


def test_snapshot_is_lossless_compact_and_retention_preserves_feedback(online_env):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 1158)).astype("float32")
    snapshot = {
        "features": {str(i): v.tolist() for i, v in enumerate(x)},
        "created": 1,
        "history": [],
    }
    encoded = pack(snapshot)
    assert len(encoded) < len(json.dumps(snapshot)) / 2
    assert unpack(encoded) == snapshot
    now = seed_feedback(2)
    with db() as c:
        before = c.execute("SELECT count(*) FROM events").fetchone()[0]
    assert cleanup(now + 100 * 86400) == 2
    with db() as c:
        assert c.execute("SELECT count(*) FROM events").fetchone()[0] == before
        assert c.execute("SELECT count(*) FROM requests").fetchone()[0] == 2


def test_click_without_view_event_trains_and_stats_use_same_mature_window(
    online_env, monkeypatch
):
    import signalfeed.api as api

    now = seed_feedback(3)
    with db() as c:
        c.execute("DELETE FROM events WHERE kind='impression' AND request_id='r0'")
        c.execute("UPDATE requests SET strategy='bookmarks' WHERE id='r1'")
        c.execute("UPDATE events SET created=1000000+2*86400+86400 WHERE id='c2'")
    records, _ = learning.feedback_records(now)
    fast = [r for r in records if r[0]["request_id"] == "r0"]
    assert (
        len(fast) == 1
        and fast[0][2] == 1
        and fast[0][0]["exposure_kind"] == "click_inferred"
    )
    monkeypatch.setattr(api, "time", SimpleNamespace(time=lambda: now))
    with TestClient(api.app) as client:
        stats = client.get("/api/stats").json()
    assert stats["impressions"] == 3 and stats["clicked_impressions"] == 1
    assert stats["click_inferred_exposures"] == 1
    assert stats["ctr"] == pytest.approx(1 / 3)


def test_rejected_model_also_consumes_validation_and_publication_is_exclusive(
    online_env,
):
    seed_feedback()
    publish_ranker(online_env, preferred=0)
    first = learning.retrain(100, 50)
    assert first["status"] == "kept_for_research"
    assert learning.retrain(100, 50)["status"] == "insufficient_chronological_feedback"
    with publication_lock(online_env("online-models")):
        with pytest.raises(RuntimeError, match="Another"):
            learning.retrain(100, 50)


def test_offline_checkpoint_rejects_mixed_dependencies(tmp_path, monkeypatch):
    import signalfeed.provenance as provenance

    monkeypatch.setattr(provenance, "path", lambda name: tmp_path)
    for name in provenance.DEPENDENCIES:
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "model.pt").write_bytes(b"weights")
    provenance.bind_model(tmp_path)
    provenance.verify_model(tmp_path)
    (tmp_path / "codes.npy").write_bytes(b"changed item-code mapping")
    with pytest.raises(ValueError, match="mismatch"):
        provenance.verify_model(tmp_path)


def _slow_worker(connection, payload):
    time.sleep(10)


def test_research_deadline_kills_stuck_worker_and_versions_cache(monkeypatch):
    import signalfeed.experimental as experimental

    original_context = multiprocessing.get_context("fork")
    monkeypatch.setattr(
        experimental.multiprocessing, "get_context", lambda _: original_context
    )
    monkeypatch.setattr(experimental, "MAX_RUNTIME", 0.2)
    monkeypatch.setattr(experimental, "_worker", _slow_worker)
    monkeypatch.setattr(experimental, "sha256", lambda _: "model-v1")
    payload = ([{"id": "a", "title": "before"}], np.zeros((1, 384)), [])
    key = experimental.cache_key(payload)
    assert key != experimental.cache_key(
        ([{"id": "a", "title": "after"}], payload[1], [])
    )
    monkeypatch.setattr(experimental, "sha256", lambda _: "model-v2")
    assert key != experimental.cache_key(payload)
    try:
        assert (
            experimental.recommend(payload, timeout=0.01)[1]
            == "research-timeout-fallback"
        )
        process = experimental._pending["process"]
        deadline = time.monotonic() + 4
        while experimental._pending is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert experimental._pending is None
        assert not process.is_alive()
    finally:
        experimental.shutdown()


def test_report_rejects_mixed_run_before_writing(tmp_path):
    import runpy
    from pathlib import Path

    render = runpy.run_path(str(Path(__file__).parents[1] / "scripts/write_report.py"))[
        "render"
    ]
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "summary.json").write_text(
        json.dumps(
            {
                "missing_seeds": [],
                "input_sha256": {},
                "provenance": {"source_sha256": "new"},
            }
        )
    )
    for name in ["validation", "latency"]:
        (reports / f"{name}.json").write_text(
            json.dumps(
                {
                    "run_id": "old-run",
                    "provenance": {"source_sha256": "old"},
                    "passed": True,
                }
            )
        )
    output = tmp_path / "RESULTS.md"
    output.write_text("prior verified report")
    with pytest.raises(ValueError, match="Mixed"):
        render(tmp_path, output)
    assert output.read_text() == "prior verified report"
