import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .store import db, initialize, connection, cleanup
from .serving import ContentService
from .snapshots import pack

service = ContentService()


async def periodic_sync():
    from .ingest import sync

    while True:
        await asyncio.sleep(900)
        await asyncio.to_thread(cleanup, time.time())
        try:
            await sync()
            await asyncio.to_thread(service.refresh, True)
        except Exception:
            logging.exception("HN refresh failed; existing catalog retained")


@asynccontextmanager
async def lifespan(app):
    initialize()
    # Keep WAL open across short request transactions, avoiding a checkpoint
    # and WAL recreation on every last-connection close. Retain FULL durability.
    anchor = connection()
    anchor.execute("SELECT count(*) FROM articles").fetchone()
    await asyncio.to_thread(cleanup, time.time())
    task = asyncio.create_task(periodic_sync())
    warm = (
        asyncio.create_task(asyncio.to_thread(service.refresh))
        if os.environ.get("SIGNALFEED_PREWARM") == "1"
        else None
    )
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    if warm is not None:
        try:
            await warm
        except Exception:
            logging.exception("Encoder warmup failed")
    from .experimental import shutdown

    await asyncio.to_thread(shutdown)
    from .cold_worker import shutdown as cold_shutdown

    await asyncio.to_thread(cold_shutdown)
    anchor.close()


app = FastAPI(title="SignalFeed", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def session(request: Request, call_next):
    uid = request.cookies.get("signalfeed_session")
    with db() as c:
        known = (
            c.execute("SELECT 1 FROM users WHERE id=?", (uid,)).fetchone()
            if uid
            else None
        )
        if not known:
            uid = uuid.uuid4().hex
            c.execute("INSERT INTO users(id) VALUES(?)", (uid,))
    request.state.user = uid
    # Same-origin writes, including local deployment. No cross-site cookie use.
    origin = request.headers.get("origin")
    if (
        request.method in {"POST", "DELETE", "PUT"}
        and origin
        and origin.rstrip("/") != str(request.base_url).rstrip("/")
    ):
        return Response("Cross-origin write rejected", status_code=403)
    response = await call_next(request)
    if not known:
        response.set_cookie(
            "signalfeed_session",
            uid,
            httponly=True,
            samesite="strict",
            max_age=31536000,
        )
    return response


@app.get("/api/health")
def health():
    with db() as c:
        count = c.execute("SELECT count(*) FROM articles WHERE deleted=0").fetchone()[0]
    ready = service.vectors is not None and bool(service.items)
    return {
        "status": "ready" if ready else "warming",
        "ready": ready,
        "articles": count,
        "strategy": service.version,
        "catalog_loaded": len(service.items),
        "catalog_refreshed_at": service.refresh_at,
        "catalog_error": service.refresh_error,
        "model_error": service.load_error,
        "test_events": os.environ.get("SIGNALFEED_TEST_EVENTS", "0") == "1",
        "offline_model_promoted": False,
    }


@app.get("/api/ready")
def ready(response: Response):
    state = health()
    if not state["ready"]:
        response.status_code = 503
    return state


def listing(request, query="", limit=30, research=False, cold=False):
    snapshot = {}
    if cold:
        from .cold_serving import recommend

        rows = recommend(service, request.state.user, limit, snapshot)
    else:
        rows = service.recommend(request.state.user, query, limit, snapshot=snapshot)
    rid = uuid.uuid4().hex
    strategy = "hybrid-search-v1" if query else "content-v1"
    model_version = snapshot.get("model_version", service.version)
    if not query:
        strategy = model_version
    if research:
        from .experimental import recommend

        with db() as c:
            history = [
                r[0]
                for r in c.execute(
                    "SELECT item_id FROM events WHERE user_id=? AND kind IN ('click','bookmark') ORDER BY created DESC LIMIT 20",
                    (request.state.user,),
                )
            ][::-1]
        with service.lock:
            payload = (list(service.items), service.vectors.copy(), history)
        ranked, strategy = recommend(payload)
        if ranked:
            model_version = "mind-sid42-ranker42-cross-domain"
            lookup = {
                a["id"]: a
                for a in service.recommend(request.state.user, "", len(service.items))
            }
            rows = [{**lookup[i], "reason": "实验推荐"} for i in ranked if i in lookup][
                :limit
            ]
    with db(durable=False) as c:
        c.execute(
            "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
            (
                rid,
                request.state.user,
                snapshot.get("created", time.time()),
                strategy,
                model_version,
                json.dumps([r["id"] for r in rows]),
                query,
            ),
        )
        if snapshot and not query and not research:
            c.execute(
                "INSERT INTO request_snapshots VALUES(?,?)",
                (rid, pack(snapshot)),
            )
    return dict(
        request_id=rid,
        strategy=strategy,
        model_version=model_version,
        experiment=snapshot.get("coldfuse"),
        items=[{**r, "position": i} for i, r in enumerate(rows)],
    )


@app.get("/api/feed")
def feed(request: Request, research: bool = False, cold: bool = False):
    if research and cold:
        raise HTTPException(422, "Choose one experiment")
    return listing(request, research=research, cold=cold)


@app.get("/api/search")
def search(request: Request, q: str = ""):
    if len(q) > 300:
        raise HTTPException(422, "Query too long")
    return listing(request, q)


class Event(BaseModel):
    event_id: uuid.UUID
    request_id: str
    item_id: str
    kind: Literal["impression", "click", "bookmark", "unbookmark"]
    position: int = Field(ge=0)


@app.post("/api/events")
def event(body: Event, request: Request):
    uid = request.state.user
    now = time.time()
    with db() as c:
        row = c.execute(
            "SELECT * FROM requests WHERE id=? AND user_id=?", (body.request_id, uid)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Unknown request")
        items = json.loads(row["items"])
        if body.position >= len(items) or items[body.position] != body.item_id:
            raise HTTPException(422, "Item/position does not match request")
        # Tests are isolated by explicit server configuration, not client claims.
        is_test = int(os.environ.get("SIGNALFEED_TEST_EVENTS", "0") == "1")
        cur = c.execute(
            "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?)",
            (
                str(body.event_id),
                body.request_id,
                uid,
                body.item_id,
                body.kind,
                body.position,
                now,
                is_test,
            ),
        )
        inserted = bool(cur.rowcount)
        if inserted and body.kind == "bookmark":
            c.execute(
                "INSERT OR IGNORE INTO bookmarks VALUES(?,?,?)",
                (uid, body.item_id, now),
            )
        elif inserted and body.kind == "unbookmark":
            c.execute(
                "DELETE FROM bookmarks WHERE user_id=? AND item_id=?",
                (uid, body.item_id),
            )
    return {"accepted": True, "inserted": inserted}


@app.get("/api/bookmarks")
def bookmarks(request: Request):
    with db() as c:
        rows = [
            dict(r)
            for r in c.execute(
                "SELECT a.* FROM bookmarks b JOIN articles a ON a.id=b.item_id WHERE b.user_id=? AND a.deleted=0 ORDER BY b.created DESC",
                (request.state.user,),
            )
        ]
    rid = uuid.uuid4().hex
    with db(durable=False) as c:
        c.execute(
            "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
            (
                rid,
                request.state.user,
                time.time(),
                "bookmarks",
                service.version,
                json.dumps([r["id"] for r in rows]),
                "",
            ),
        )
    return dict(
        request_id=rid,
        items=[
            {**r, "bookmarked": True, "position": i, "reason": "我的收藏"}
            for i, r in enumerate(rows)
        ],
    )


class Interests(BaseModel):
    topics: list[str] = Field(max_length=12)


@app.get("/api/interests")
def read_interests(request: Request):
    with db() as c:
        row = c.execute(
            "SELECT interests FROM users WHERE id=?", (request.state.user,)
        ).fetchone()
    return {"topics": json.loads(row[0]) if row else []}


@app.put("/api/interests")
def interests(body: Interests, request: Request):
    if any(len(t) > 80 for t in body.topics):
        raise HTTPException(422, "Topic too long")
    with db() as c:
        c.execute(
            "UPDATE users SET interests=? WHERE id=?",
            (json.dumps(body.topics), request.state.user),
        )
    return {"topics": body.topics}


@app.get("/api/stats")
def stats():
    from .online_schema import LABEL_WINDOW

    now = time.time()
    with db() as c:
        groups = [
            dict(r)
            for r in c.execute(
                "WITH exposures AS (SELECT r.id,r.strategy,r.model_version,e.item_id, "
                "MAX(e.kind='click') clicked,MAX(e.kind='impression') viewable "
                "FROM requests r JOIN events e ON e.request_id=r.id AND e.user_id=r.user_id "
                "WHERE r.created+?<=? AND e.created>=r.created AND e.created<r.created+? "
                "AND e.is_test=0 AND e.kind IN ('impression','click') AND r.query='' "
                "AND (r.strategy='content-v1' OR r.strategy LIKE 'online-%' OR r.strategy LIKE 'coldfuse-%') "
                "GROUP BY r.id,e.item_id) "
                "SELECT strategy,model_version,COUNT(*) impressions,SUM(clicked) clicked_impressions, "
                "SUM(viewable) viewable_impressions,SUM(NOT viewable) click_inferred_exposures "
                "FROM exposures GROUP BY strategy,model_version",
                (LABEL_WINDOW, now, LABEL_WINDOW),
            )
        ]
        bookmarks = c.execute("SELECT count(*) FROM bookmarks").fetchone()[0]
    n = sum(g["impressions"] for g in groups)
    clicked = sum(g["clicked_impressions"] for g in groups)
    return {
        "scope": "HN feed only; mature 24h window; viewable OR click-inferred exposure; test events excluded",
        "training_note": "Training additionally requires compatible immutable snapshots",
        "label_window_seconds": LABEL_WINDOW,
        "impressions": n,
        "clicked_impressions": clicked,
        "viewable_impressions": sum(g["viewable_impressions"] for g in groups),
        "click_inferred_exposures": sum(g["click_inferred_exposures"] for g in groups),
        "ctr": clicked / n if n else None,
        "bookmarks": bookmarks,
        "by_policy": groups,
    }


frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
