"""Latency regressions: isolated data, controlled stalls, no training/downloads."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signalfeed.background import BoundedRunner
from signalfeed.serving import ContentService
from signalfeed.store import db


def test_cached_interest_never_waits_for_encoder_lock():
    service = ContentService()
    vector = np.ones(384, dtype="float32")
    service.query_cache["AI"] = vector
    service.encode_lock.acquire()
    try:
        started = time.monotonic()
        assert service.encode_query("AI") is vector
        assert service.encode_query_bounded("AI") is vector
        assert time.monotonic() - started < 0.1
    finally:
        service.encode_lock.release()


def test_profile_wait_bounded_single_flight_and_no_cross_interest_reuse(monkeypatch):
    service = ContentService()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def encode(text):
        calls.append(text)
        entered.set()
        release.wait(2)
        value = np.full(384, len(text), dtype="float32")
        service.query_cache[text] = value
        return value

    monkeypatch.setattr(service, "encode_query", encode)
    try:
        start = time.monotonic()
        assert service.encode_query_bounded("AI", timeout=0.01) is None
        assert entered.wait(1)
        assert service.encode_query_bounded("Security", timeout=0.01) is None
        assert service.encode_query_bounded("AI", timeout=0.01) is None
        assert time.monotonic() - start < 0.2
        assert calls == ["AI"]
    finally:
        release.set()
    assert service.profile_job[1].done.wait(1)
    assert np.all(service.encode_query_bounded("AI") == 2)
    assert np.all(service.encode_query_bounded("Security") == 8)


def test_catalog_refresh_does_not_block_complete_cached_catalog(monkeypatch):
    service = ContentService()
    service.vectors = np.ones((1, 384))
    service.items = [{"id": "old"}]
    release, entered = threading.Event(), threading.Event()
    calls = []

    def refresh():
        calls.append(1)
        entered.set()
        release.wait(2)

    monkeypatch.setattr(service, "refresh", refresh)
    try:
        start = time.monotonic()
        assert service.refresh_in_background()
        assert entered.wait(1)
        assert service.refresh_in_background()
        assert time.monotonic() - start < 0.2
        assert service.items == [{"id": "old"}] and calls == [1]
    finally:
        release.set()


def test_complete_research_wait_is_bounded_and_admission_has_no_queue(monkeypatch):
    import signalfeed.cold_serving as cold
    import signalfeed.cold_worker as worker

    release = threading.Event()
    calls = []

    def delayed(payload, timeout):
        calls.append(payload["user_id"])
        release.wait(2)
        return payload, "cache", "version"

    monkeypatch.setattr(cold, "research_jobs", BoundedRunner(2))
    monkeypatch.setattr(worker, "recommend", delayed)
    try:
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: cold.bounded_research({"user_id": str(i)}, 0.05), range(8)
                )
            )
        assert time.monotonic() - start < 0.3
        assert len(calls) == 2
        assert all(r[0] is None for r in results)
        assert {r[1] for r in results} == {"timeout-fallback", "busy-fallback"}
    finally:
        release.set()


def test_research_exception_returns_fallback(monkeypatch):
    import signalfeed.cold_serving as cold
    import signalfeed.cold_worker as worker

    def broken(*args, **kwargs):
        raise OSError("synthetic model IO failure")

    monkeypatch.setattr(cold, "research_jobs", BoundedRunner(1))
    monkeypatch.setattr(worker, "recommend", broken)
    assert cold.bounded_research({}, 0.2)[1] == "model-error-fallback"


def test_real_worker_cache_scopes_identical_profiles_to_identity(monkeypatch):
    import multiprocessing
    import signalfeed.cold_worker as worker
    from test_coldfuse import _echo_worker

    worker.shutdown()
    worker.startup()
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(worker.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(worker, "_worker", _echo_worker)
    monkeypatch.setattr(worker, "model_identity", lambda: "test-version")
    try:
        a = dict(user_id="a", history=[0], explicit=[1, 0])
        b = dict(a, user_id="b")
        for payload in (a, b):
            result, status, _ = worker.recommend(payload)
            assert result == payload and status == "research"
            result, status, _ = worker.recommend(payload)
            assert result == payload and status == "cache"
        changed = dict(b, history=[1])
        assert worker.recommend(changed)[0] == changed
        assert worker.recommend(a)[0] == a
    finally:
        worker.shutdown()


@pytest.mark.parametrize("cohort", ["zero", "interests", "history"])
def test_first_request_and_warm_request_are_fast_and_correctly_scoped(
    online_env,
    monkeypatch,
    cohort,
):
    import signalfeed.api as api
    import signalfeed.cold_serving as cold
    import signalfeed.cold_worker as worker
    from test_feedback_policy import catalog, click

    service = catalog(monkeypatch)
    service.refresh_at = time.time()
    monkeypatch.setenv("SIGNALFEED_PREWARM", "0")
    monkeypatch.setenv("SIGNALFEED_TEST_EVENTS", "1")
    monkeypatch.setattr(api, "service", service)
    monkeypatch.setattr(cold, "active_ranker", lambda: None)
    monkeypatch.setattr(cold, "research_jobs", BoundedRunner(2))
    seen = []

    def research(payload, timeout):
        seen.append(payload)
        return None, "unavailable-fallback", "version"

    monkeypatch.setattr(worker, "recommend", research)
    user = cohort
    with db() as c:
        c.execute(
            "INSERT INTO users VALUES(?,?)",
            (user, json.dumps(["Security"] if cohort == "interests" else [])),
        )
    if cohort == "history":
        click(user, "0")
    with TestClient(api.app) as client:
        client.cookies.set("signalfeed_session", user)
        for _ in range(2):
            start = time.monotonic()
            response = client.get("/api/feed?cold=true")
            assert time.monotonic() - start < 3
            assert response.status_code == 200 and response.json()["items"]
            assert "profile_db;dur=" in response.headers["Server-Timing"]
            assert "snapshot_db;dur=" in response.headers["Server-Timing"]
    assert bool(seen) == (cohort != "zero")
    assert all(p["user_id"] == user for p in seen)
    if seen:
        key = worker.cache_key(seen[0], "v")
        assert key != worker.cache_key(dict(seen[0], user_id="other"), "v")
        for field, value in [
            ("history", [3]),
            ("explicit", np.ones(384)),
            ("eligible", np.zeros(4, dtype=bool)),
            ("cf_scores", np.ones(4)),
            ("popularity", np.zeros(4)),
        ]:
            assert key != worker.cache_key(dict(seen[0], **{field: value}), "v")


def test_profile_stall_returns_truthful_base_then_recovers(online_env, monkeypatch):
    import signalfeed.api as api
    import signalfeed.cold_serving as cold
    import signalfeed.cold_worker as worker
    from test_feedback_policy import catalog

    service = catalog(monkeypatch)
    service.refresh_at = time.time()
    monkeypatch.setenv("SIGNALFEED_PREWARM", "0")
    monkeypatch.setattr(api, "service", service)
    monkeypatch.setattr(cold, "active_ranker", lambda: None)
    monkeypatch.setattr(
        worker, "recommend", lambda *a, **kw: (None, "unavailable-fallback", None)
    )
    release = threading.Event()

    def encode(text):
        release.wait(2)
        value = service.vectors[1]
        service.query_cache[text] = value
        return value

    monkeypatch.setattr(service, "encode_query", encode)
    with db() as c:
        c.execute("INSERT INTO users VALUES('slow', '[\"Science\"]')")
    with TestClient(api.app) as client:
        client.cookies.set("signalfeed_session", "slow")
        try:
            start = time.monotonic()
            response = client.get("/api/feed?cold=true")
            assert time.monotonic() - start < 0.8
            assert response.json()["experiment"]["status"] == "profile-warming-fallback"
            assert response.json()["items"]
        finally:
            release.set()
        assert service.profile_job[1].done.wait(1)
        assert (
            client.get("/api/feed?cold=true").json()["experiment"]["status"]
            == "unavailable-fallback"
        )


def test_runner_shutdown_closes_admission_and_retains_inflight_slot():
    runner = BoundedRunner(1)
    release = threading.Event()
    job = runner.submit(lambda: release.wait(2))
    try:
        assert not runner.close(timeout=0.01)
        assert runner.submit(lambda: 1) is None
        with pytest.raises(RuntimeError):
            runner.open()
    finally:
        release.set()
    assert job.done.wait(1)
    assert runner.close(timeout=1)
    runner.open()
    assert runner.submit(lambda: 42).wait(1) == (True, 42)


def test_shutdown_during_model_hash_cannot_spawn_late_worker(monkeypatch):
    import signalfeed.cold_worker as worker

    entered, release = threading.Event(), threading.Event()
    worker.shutdown()
    worker.startup()

    def identity():
        entered.set()
        release.wait(2)
        return "v"

    monkeypatch.setattr(worker, "model_identity", identity)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.recommend, {"user_id": "synthetic"})
        assert entered.wait(1)
        worker.shutdown(final=True)
        release.set()
        assert future.result(timeout=1)[1] == "unavailable-fallback"
        assert worker._process is None
        assert worker.recommend({})[1] == "unavailable-fallback"
    worker.startup()


def test_health_probes_do_not_create_anonymous_users(online_env, monkeypatch):
    import signalfeed.api as api
    from test_feedback_policy import catalog

    monkeypatch.setattr(api, "service", catalog(monkeypatch))
    monkeypatch.setenv("SIGNALFEED_PREWARM", "0")
    with db() as c:
        before = c.execute("SELECT count(*) FROM users").fetchone()[0]
    with TestClient(api.app) as client:
        for endpoint in ("health", "ready"):
            response = client.get("/api/" + endpoint)
            assert response.status_code == 200
            assert "set-cookie" not in response.headers
    with db() as c:
        assert c.execute("SELECT count(*) FROM users").fetchone()[0] == before
