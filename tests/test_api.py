import json
import time
import uuid
import pytest
from fastapi.testclient import TestClient
from signalfeed.store import initialize, db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALFEED_DB", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("SIGNALFEED_TEST_EVENTS", "1")
    initialize()
    import signalfeed.api as api

    with db() as c:
        c.execute(
            "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)",
            (
                "1",
                "Real fixture for API logic",
                "https://example.com",
                "example.com",
                time.time(),
                1,
                0,
                time.time(),
            ),
        )
    monkeypatch.setattr(
        api.service,
        "recommend",
        lambda *a, **k: [
            dict(id="1", title="Fixture", url="https://example.com", bookmarked=False)
        ],
    )
    with TestClient(api.app) as c:
        yield c


def event(feed, kind="impression", **kwargs):
    return dict(
        event_id=str(uuid.uuid4()),
        request_id=feed["request_id"],
        item_id="1",
        kind=kind,
        position=0,
        **kwargs,
    )


def test_feedback_persistent_deduplicated_and_bound_to_request(client):
    feed = client.get("/api/feed").json()
    body = event(feed)
    assert client.post("/api/events", json=body).json()["inserted"]
    assert not client.post("/api/events", json=body).json()["inserted"]
    body["event_id"] = str(uuid.uuid4())
    assert not client.post("/api/events", json=body).json()["inserted"]
    assert client.post("/api/events", json=event(feed, "bookmark")).status_code == 200
    assert client.get("/api/bookmarks").json()["items"][0]["id"] == "1"
    with db() as c:
        assert c.execute("SELECT count(*) FROM bookmarks").fetchone()[0] == 1
    body = event(feed)
    body["position"] = 1
    assert client.post("/api/events", json=body).status_code == 422
    assert client.get("/api/stats").json()["ctr"] is None


def test_foreign_session_cannot_write_another_users_request(client):
    feed = client.get("/api/feed").json()
    client.cookies.clear()
    assert client.post("/api/events", json=event(feed)).status_code == 404


def test_cross_origin_write_rejected(client):
    assert (
        client.put(
            "/api/interests",
            json={"topics": ["AI"]},
            headers={"origin": "https://foreign.invalid"},
        ).status_code
        == 403
    )
