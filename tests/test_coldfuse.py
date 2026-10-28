import json
import time
import uuid
from types import SimpleNamespace
import numpy as np
import torch
from scipy import sparse
from fastapi.testclient import TestClient
from signalfeed.coldfuse import (
    ColdFuse,
    Config,
    state,
    matrix,
    blend,
    weights,
    CONTRACT,
)
from signalfeed.cold_protocol import simulate, prototypes, sample
from signalfeed.cold_experiments import context, compose
from signalfeed.ranker import Ranker
from signalfeed.store import db
from signalfeed.snapshots import unpack


def fixture():
    x = np.eye(384, dtype="float32")[:20]
    codes = np.array([[i % 2, 0, i % 3] for i in range(len(x))])
    return x, ColdFuse(x, np.arange(len(x)), sparse.eye(len(x), format="csr"), codes)


def test_zero_and_one_click_no_fake_evidence():
    x, core = fixture()
    mask = np.ones(len(x), dtype=bool)
    zero = core.recommend([], mask)
    assert zero["ids"] == list(range(19, -1, -1))
    assert zero["channels"]["lists"]["content"] == []
    assert zero["channels"]["lists"]["itemcf"] == []
    assert zero["channels"]["lists"]["generated"] == []
    first = state(x, [0])
    assert first.profile[0] == 1 and first.reliability > 0
    assert state(x, [0] * 10).reliability == first.reliability
    prior = state(x, [], x[1])
    assert prior.profile[1] == 1
    updated = state(x, [0], x[1])
    assert updated.profile[0] > 0 and updated.profile[1] > 0
    assert np.isfinite(matrix(x, [0, 1], prior)).all()
    assert core.recommend([], np.zeros(len(x), dtype=bool))["ids"] == []


def test_selection_accepts_exact_registered_grid_and_rejects_duplicate():
    from dataclasses import asdict
    import pytest
    from signalfeed.cold_experiments import GRID, validate_selection

    source = dict(split="valid", source_sha256="fixed", data={})
    trials = [dict(config=asdict(c)) for c in GRID]
    assert len(trials) == 27
    validate_selection(dict(provenance=source, trials=trials), source)
    with pytest.raises(ValueError, match="complete validation grid"):
        validate_selection(
            dict(provenance=source, trials=trials[:-1] + trials[:1]), source
        )
    with pytest.raises(ValueError, match="sources differ"):
        validate_selection(
            dict(provenance=source, trials=trials),
            dict(source, source_sha256="changed"),
        )


def test_small_tune_to_evaluate_pipeline(tmp_path, monkeypatch):
    """Synthetic control-flow regression, never an experimental quality result."""
    import signalfeed.cold_experiments as exp
    from signalfeed.features import Features
    from signalfeed.provenance import sha256

    def path(name):
        p = tmp_path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    f = object.__new__(Features)
    f.x, core = fixture()
    f.news = [dict(id=str(i), category=i % 2) for i in range(20)]
    f.index = {str(i): i for i in range(20)}
    f.categories, f.first, f.cutoff = np.arange(20) % 2, np.arange(20), 19
    f.pop, f.cf, f.codes = core.pop, core.cf, core.codes
    model = Ranker().eval()
    row = dict(
        id="example",
        user="one",
        time=30,
        history=[str(i) for i in range(15)],
        items=["18", "19"],
        labels=[1, 0],
    )
    for name in ("coldfuse", "sid", "ranker"):
        (path(f"models/{name}-42") / "model.pt").write_bytes(b"synthetic")
    (path("models") / "fusion.json").write_text("{}")
    source = dict(
        source_sha256="fixture",
        data={},
        models={
            name: sha256(path(f"models/{name}-42") / "model.pt")
            for name in ("sid", "ranker")
        },
    )
    monkeypatch.setattr(exp, "path", path)
    monkeypatch.setattr(
        exp, "run_identity", lambda seed, split: dict(source, split=split)
    )
    monkeypatch.setattr(exp, "code_identity", lambda: source)
    monkeypatch.setattr(exp, "setup", lambda seed: (f, core, model, prototypes(f)))
    monkeypatch.setattr(exp, "load_rows", lambda split: [row])
    monkeypatch.setattr(exp, "load_ranker", lambda seed: model)
    monkeypatch.setattr(exp, "load_gate", lambda: dict(threshold=5, weight=1))
    selection = exp.tune()
    assert len(selection["trials"]) == 27
    result = exp.evaluate()
    assert result["requests"] == 1 and len(result["groups"]) == 10
    lines = (
        (path("reports/cold-test-42") / "predictions.jsonl").read_text().splitlines()
    )
    assert len(lines) == 10
    assert all(len(json.loads(line)["variants"]) == 14 for line in lines)


def test_protocol_disjoint_and_label_independent():
    x, _ = fixture()
    f = SimpleNamespace(
        x=x,
        categories=np.arange(len(x)) % 2,
        first=np.arange(len(x)),
        cutoff=10,
        history=lambda r: r["history"],
    )
    pv = prototypes(f)
    row = dict(history=list(range(15)), labels=[1, 0])
    h, prior, older = simulate(f, row, 5, "proxy_explicit", pv)
    assert h == list(range(10, 15)) and not set(h).intersection(older)
    changed = simulate(f, dict(row, labels=[0, 1]), 5, "proxy_explicit", pv)
    np.testing.assert_array_equal(changed[1], prior)
    assert changed[0] == h
    f.x = x.copy()
    f.x[10:] = 100
    for c, v in prototypes(f).items():
        np.testing.assert_array_equal(v, pv[c])
    assert sample([row, dict(history=[])], f, 300) == [row]


def test_offline_serving_parity_and_controlled_ablation():
    x, core = fixture()
    torch.manual_seed(42)
    model = Ranker().eval()
    mask = np.ones(len(x), dtype=bool)
    mask[0] = False
    config = Config()
    ctx = context(core, [0], x[1], mask, config, model, [2, 3])
    offline, _ = compose(ctx, config, [2, 3])
    serving = core.recommend([0], mask, x[1], config, model)
    assert offline == serving["ids"]
    removed = core.recommend([0], mask, x[1], config, model, "no_gen")
    assert removed["gates"] == serving["gates"]
    np.testing.assert_array_equal(ctx["state"].profile, serving["state"].profile)
    empty = matrix(x, [2, 3], state(x, []))
    np.testing.assert_array_equal(
        blend(empty, [0.2, 0.8], [100, -100]), [np.float32(0.2), np.float32(0.8)]
    )
    assert weights(state(x, []), 0, 0)["generated"] == 0


def test_generation_anchors_not_history_and_eligibility():
    x, core = fixture()
    calls = []

    class Generator:
        def generate(self, history, index, codes, **kw):
            calls.append(history)
            return [(next(iter(kw["valid_codes"])), -1)]

    core.generator = Generator()
    mask = np.ones(len(x), dtype=bool)
    mask[0] = False
    result = core.recommend([], mask, x[0])
    assert calls and not result["state"].history
    assert 0 not in result["channels"]["anchors"]
    assert 0 not in result["ids"]


def test_cache_key_binds_every_request_input():
    from signalfeed.cold_worker import cache_key

    payload = dict(
        vectors=np.eye(3, dtype="float32"),
        history=[],
        ids=["a", "b", "c"],
        explicit=np.zeros(3),
    )
    key = cache_key(payload, "v1")
    assert cache_key(payload, "v2") != key
    for field, value in [
        ("history", [0]),
        ("ids", ["c", "b", "a"]),
        ("explicit", np.ones(3)),
        ("vectors", np.ones((3, 3))),
    ]:
        assert cache_key(dict(payload, **{field: value}), "v1") != key


def _stall(connection):
    time.sleep(10)


def _echo_worker(connection):
    while True:
        payload = connection.recv()
        time.sleep(0.05)
        connection.send((True, payload))


def test_completed_worker_survives_idle_and_reuses_process(monkeypatch):
    import multiprocessing
    import signalfeed.cold_worker as worker

    worker.shutdown()
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(worker.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(worker, "MAX_RUNTIME", 0.3)
    monkeypatch.setattr(worker, "model_identity", lambda: "test")
    monkeypatch.setattr(worker, "_worker", _echo_worker)
    try:
        assert worker.recommend(dict(value=1), timeout=0.001)[1] == "timeout-fallback"
        process = worker._process
        time.sleep(0.4)
        assert worker._pending is None and process.is_alive()
        assert worker.recommend(dict(value=1))[1] == "cache"
        assert worker.recommend(dict(value=2))[0] == dict(value=2)
        assert worker._process is process
    finally:
        worker.shutdown()


def test_cold_worker_hard_deadline(monkeypatch):
    import signalfeed.cold_worker as worker

    worker.shutdown()
    monkeypatch.setattr(worker, "MAX_RUNTIME", 0.2)
    monkeypatch.setattr(worker, "model_identity", lambda: "test")
    monkeypatch.setattr(worker, "_worker", _stall)
    try:
        # Large enough to block Pipe.send: request still has a bounded wait.
        start = time.monotonic()
        result = worker.recommend(dict(vectors=np.zeros((2000, 384))), timeout=0.01)
        assert result[1] == "timeout-fallback"
        assert time.monotonic() - start < 2
        deadline = time.monotonic() + 3
        while worker._pending is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert worker._pending is None
        assert worker._process is None
    finally:
        worker.shutdown()


def test_cold_api_feedback_snapshot_and_missing_model_fallback(online_env, monkeypatch):
    import signalfeed.api as api
    import signalfeed.cold_worker as worker
    import signalfeed.cold_serving as serving
    from signalfeed.serving import ContentService
    from signalfeed.online_learning import feedback_records

    now = time.time()
    x, _ = fixture()
    service = ContentService()
    service.items = [
        dict(
            id=str(i),
            title=f"Article {i}",
            score=i,
            published=now,
            url="https://example.com",
            domain="example.com",
        )
        for i in range(20)
    ]
    service.vectors, service.index = x, {str(i): i for i in range(20)}
    monkeypatch.setattr(service, "refresh", lambda *a: None)
    monkeypatch.setattr(service, "encode_query", lambda text: x[0])
    monkeypatch.setattr(api, "service", service)
    monkeypatch.setattr(serving, "active_ranker", lambda: None)
    calls = []

    def research(payload):
        calls.append(payload)
        return None, "unavailable-fallback", None

    monkeypatch.setattr(worker, "recommend", research)
    with db() as c:
        for a in service.items:
            c.execute(
                "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)",
                (a["id"], a["title"], a["url"], a["domain"], now, a["score"], 0, now),
            )
    with TestClient(api.app) as client:
        empty = client.get("/api/feed?cold=true").json()
        assert empty["experiment"]["status"] == "no-personal-signal"
        assert not calls
        client.put("/api/interests", json={"topics": ["AI"]})
        feed = client.get("/api/feed?cold=true").json()
        assert feed["experiment"]["status"] == "unavailable-fallback"
        assert feed["strategy"].startswith("coldfuse-")
        item = feed["items"][0]
        body = dict(
            event_id=str(uuid.uuid4()),
            request_id=feed["request_id"],
            item_id=item["id"],
            kind="click",
            position=0,
        )
        assert client.post("/api/events", json=body).status_code == 200
        again = client.get("/api/feed?cold=true").json()
        assert item["id"] not in [a["id"] for a in again["items"]]
        with db() as c:
            before = unpack(
                c.execute(
                    "SELECT payload FROM request_snapshots WHERE request_id=?",
                    (feed["request_id"],),
                ).fetchone()[0]
            )
            after = unpack(
                c.execute(
                    "SELECT payload FROM request_snapshots WHERE request_id=?",
                    (again["request_id"],),
                ).fetchone()[0]
            )
        assert before["history"] == [] and after["history"] == [item["id"]]
        assert before["feature_version"] == CONTRACT["feature_version"]
        records, _ = feedback_records(time.time() + 86401, cold=True)
        assert len(records) == 1 and records[0][2] == 1
        np.testing.assert_array_equal(records[0][1], before["features"][item["id"]])
        assert feedback_records(time.time() + 86401)[0] == []
        assert client.get("/api/feed?cold=true&research=true").status_code == 422


def test_separate_cold_feedback_training_contract(online_env):
    from test_online import seed_feedback
    from signalfeed.online_learning import retrain

    now = seed_feedback(150)
    with db() as c:
        for row in list(c.execute("SELECT request_id,payload FROM request_snapshots")):
            snap = json.loads(row["payload"])
            snap.update(CONTRACT, model_version="coldfuse-test")
            c.execute(
                "UPDATE request_snapshots SET payload=? WHERE request_id=?",
                (json.dumps(snap), row["request_id"]),
            )
        c.execute(
            "UPDATE requests SET strategy='coldfuse-test',model_version='coldfuse-test'"
        )
    # Fixture has mature real-flagged synthetic feedback in an isolated tmp DB.
    assert now < time.time()
    result = retrain(100, 20, cold=True)
    assert result["feature_version"] == CONTRACT["feature_version"]
    assert result["status"] in ("promoted", "kept_for_research")
    assert (
        online_env("cold-online-models") / result["version"] / "manifest.json"
    ).exists()
    assert not (online_env("online-models") / "active.json").exists()
