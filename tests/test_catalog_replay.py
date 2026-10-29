"""Replay invariants on isolated synthetic traffic, not recommendation evidence."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import numpy as np
import pytest
from signalfeed.catalog_replay import ReplayStore, ReplayReader, canonical, digest
import signalfeed.serving as serving
from signalfeed.store import db


def fixture_service(online_env, monkeypatch):
    monkeypatch.setenv("SIGNALFEED_TEST_EVENTS", "1")
    clock = [1000000.0]
    monkeypatch.setattr(serving, "time", SimpleNamespace(time=lambda: clock[0]))
    s = serving.ContentService()
    rng = np.random.default_rng(9)
    x = rng.normal(size=(150, 384)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x.setflags(write=False)
    s.vectors = x
    s.items = [
        dict(
            id=str(i),
            title="article " + str(i),
            published=999000 + i,
            score=i % 7,
            deleted=0,
            updated=999999,
        )
        for i in range(len(x))
    ]
    s.index = {a["id"]: i for i, a in enumerate(s.items)}
    s.refresh_at = 999999.0
    monkeypatch.setattr(s, "refresh", lambda *a: None)
    monkeypatch.setattr(s, "encode_query", lambda text: x[2].copy())
    with db() as c:
        c.execute("INSERT INTO users VALUES(?,?)", ("user", '["topic"]'))
        for name, user, item, stamp, test in [
            ("h", "user", "0", 999998, 1),
            ("repeat", "user", "0", 999999, 1),
            ("future", "user", "3", 1000001, 1),
            ("unknown", "user", "gone", 999997, 1),
            ("peer1", "peer", "0", 999998, 0),
            ("peer2", "peer", "2", 999998, 0),
            ("peer3", "peer", "outside", 999998, 0),
            ("peer4", "other", "0", 999998, 0),
            ("peer5", "other", "outside", 999998, 0),
        ]:
            c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                (name, name, user, item, "click", 0, stamp, test),
            )
    return s, clock


def test_full_policy_replay_and_catalog_reuse(online_env, monkeypatch, tmp_path):
    s, _ = fixture_service(online_env, monkeypatch)
    store = ReplayStore(tmp_path / "replay")
    plain = s.recommend("user", snapshot={})
    recorded = s.recommend("user", snapshot={}, replay=store)
    assert [r["id"] for r in recorded] == [r["id"] for r in plain]
    first = store.last_request
    r = ReplayReader(store.root).replay(first)
    assert r["returned"] == [a["id"] for a in plain]
    assert len(r["top100"]) == 100
    envelope = json.loads((store.root / "requests" / f"{first}.json").read_text())
    record = envelope["record"]
    assert record["positive_actions"] == 3 and record["history_newest_first"] == ["0"]
    assert record["collaborative"]["outside"] == 2
    assert "future" not in str(record["observed_unique_history_newest_first"])
    s.recommend("user", replay=store)
    assert len(list((store.root / "catalogs").iterdir())) == 1
    assert len(list((store.root / "requests").iterdir())) == 2


def test_catalog_text_scores_vectors_delete_and_future_feedback_do_not_rewrite(
    online_env, monkeypatch, tmp_path
):
    s, clock = fixture_service(online_env, monkeypatch)
    store = ReplayStore(tmp_path / "replay")
    old = s.recommend("user", replay=store)
    request = store.last_request
    original = (store.root / "requests" / f"{request}.json").read_bytes()
    first_version = store._catalog.version
    with db() as c:
        c.execute("UPDATE users SET interests=? WHERE id=?", ('["different"]', "user"))
    clock[0] += 10
    # A refresh creates new rows/vectors, removes one item and changes content.
    s.items = [dict(a, title="changed", score=200) for a in s.items[:-1]]
    s.vectors = s.vectors[:-1].copy()
    s.vectors.setflags(write=False)
    s.index = {a["id"]: i for i, a in enumerate(s.items)}
    s.refresh_at = clock[0]
    s.recommend("user", replay=store)
    assert first_version != store._catalog.version
    assert len(list((store.root / "catalogs").iterdir())) == 2
    replay = ReplayReader(store.root).replay(request)
    assert replay["returned"] == [a["id"] for a in old]
    assert (store.root / "requests" / f"{request}.json").read_bytes() == original


def test_corruption_missing_catalog_and_wrong_budget_fail_closed(
    online_env, monkeypatch, tmp_path
):
    s, _ = fixture_service(online_env, monkeypatch)
    store = ReplayStore(tmp_path / "replay")
    s.recommend("user", replay=store)
    file = store.root / "requests" / f"{store.last_request}.json"
    original = file.read_bytes()
    frame = json.loads(original)
    frame["record"]["limit"] = 7
    file.write_bytes(canonical(frame))
    with pytest.raises(ValueError, match="checksum"):
        ReplayReader(store.root).replay(store.last_request)
    frame = json.loads(original)
    frame["record"]["candidate_budget"] = 200
    frame["sha256"] = digest(canonical(frame["record"]))
    file.write_bytes(canonical(frame))
    with pytest.raises(ValueError, match="contract"):
        ReplayReader(store.root).replay(store.last_request)
    file.write_bytes(original)
    vectors = store.root / "catalogs" / store._catalog.version / "vectors.npy"
    vectors.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        ReplayReader(store.root).replay(store.last_request)
    vectors.unlink()
    with pytest.raises(FileNotFoundError):
        ReplayReader(store.root).replay(store.last_request)


def test_prototype_is_never_implicitly_enabled(online_env, monkeypatch, tmp_path):
    s, _ = fixture_service(online_env, monkeypatch)
    store = ReplayStore(tmp_path / "replay")
    monkeypatch.setenv("SIGNALFEED_TEST_EVENTS", "0")
    with pytest.raises(ValueError, match="isolated"):
        ReplayStore(tmp_path / "production")
    with pytest.raises(ValueError, match="isolated"):
        s.recommend("user", replay=store)
    assert s.recommend("user")
    assert list((store.root / "requests").iterdir()) == []
    monkeypatch.setenv("SIGNALFEED_TEST_EVENTS", "1")
    with pytest.raises(ValueError, match="isolated"):
        s.recommend("user", query="search", replay=store)
    monkeypatch.setattr(s, "active_ranker", lambda: object())
    with pytest.raises(ValueError, match="DNN"):
        s.recommend("user", replay=store)


def test_immutable_vectors_and_concurrent_registration(
    online_env, monkeypatch, tmp_path
):
    s, _ = fixture_service(online_env, monkeypatch)
    store = ReplayStore(tmp_path / "replay")
    with pytest.raises(ValueError, match="immutable"):
        store.register(s.items, s.vectors.copy())
    with ThreadPoolExecutor(max_workers=3) as pool:
        versions = list(
            pool.map(lambda _: store.register(s.items, s.vectors), range(6))
        )
    assert len(set(versions)) == 1
    assert len(list((store.root / "catalogs").iterdir())) == 1


def test_raw_history_is_not_replaced_by_capped_effective_history(
    online_env, monkeypatch, tmp_path
):
    s, _ = fixture_service(online_env, monkeypatch)
    with db() as c:
        c.execute("DELETE FROM events WHERE user_id=?", ("user",))
        for i in range(25):
            c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                (f"long{i}", f"r{i}", "user", str(i), "click", 0, 999900 + i, 1),
            )
    store = ReplayStore(tmp_path / "replay")
    s.recommend("user", replay=store)
    r = json.loads(
        (store.root / "requests" / f"{store.last_request}.json").read_text()
    )["record"]
    assert r["positive_actions"] == 25 and len(r["history_newest_first"]) == 20
    assert len(r["observed_unique_history_newest_first"]) == 25
    assert ReplayReader(store.root).replay(store.last_request)["source_verified"]


def test_external_prefix_reads_complete_lines_only():
    file = Path(__file__).resolve().parents[1] / "scripts/audit_news_sources.py"
    spec = importlib.util.spec_from_file_location("audit_news_sources", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        directory = tarfile.TarInfo("dataset")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        payload = b'{"time":1}\n{"time":2}\n{"partial":'
        member = tarfile.TarInfo("dataset/day.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    name, rows = mod.first_tar_lines(stream.getvalue(), limit=1)
    assert name == "dataset/day.json" and rows == [{"time": 1}]
