"""HN adapter: request-time inputs, immediate feedback adaptation, honest fallback."""

import json
import time
import numpy as np
from .coldfuse import ColdFuse, CONTRACT, FEATURE_VERSION, policy_scores
from .serving import ContentService
from .store import db

online = ContentService()


def active_ranker():
    with online.lock:
        return online._active_ranker(contract=CONTRACT, directory="cold-online-models")


def recommend(service, user, limit, snapshot):
    service.refresh()
    with service.lock:
        items, x, index = service.items, service.vectors, service.index
    created = time.time()
    with db() as c:
        c.execute("BEGIN")
        row = c.execute("SELECT interests FROM users WHERE id=?", (user,)).fetchone()
        interests = json.loads(row[0]) if row else []
        history = [
            r[0]
            for r in c.execute(
                "SELECT item_id,MAX(created) stamp FROM events WHERE user_id=? AND created<=? "
                "AND kind IN ('click','bookmark') GROUP BY item_id ORDER BY stamp DESC,item_id LIMIT 20",
                (user, created),
            )
        ][::-1]
        saved = {
            r[0]
            for r in c.execute("SELECT item_id FROM bookmarks WHERE user_id=?", (user,))
        }
        cf = {}
        if history:
            marks = ",".join("?" for _ in history)
            cf = dict(
                c.execute(
                    f"SELECT e2.item_id,COUNT(DISTINCT e2.user_id) FROM events e1 JOIN events e2 ON e1.user_id=e2.user_id WHERE e1.item_id IN ({marks}) "
                    "AND e1.user_id<>? AND e1.kind IN ('click','bookmark') AND e2.kind IN ('click','bookmark') AND e1.is_test=0 AND e2.is_test=0 "
                    "AND e1.created<=? AND e2.created<=? GROUP BY e2.item_id",
                    (*history, user, created, created),
                )
            )
    h = [index[i] for i in history if i in index]
    explicit = (
        service.encode_query(" ".join(interests))
        if interests
        else np.zeros(384, dtype="float32")
    )
    eligible = np.ones(len(items), dtype=bool)
    eligible[h] = False
    popularity = np.log1p([max(0, a["score"]) for a in items])
    if len(items):
        popularity /= max(1, popularity.max())
    # Minute resolution permits cache reuse without future data or stale history.
    freshness = np.array(
        [
            np.exp(-max(0, int(created / 60) * 60 - a["published"]) / 259200)
            for a in items
        ]
    )
    popularity = 0.7 * popularity + 0.3 * freshness
    cf_scores = np.array([cf.get(a["id"], 0) for a in items], dtype="float32")
    core = ColdFuse(x, popularity)
    result = core.recommend(h, eligible, explicit, cf_scores=cf_scores)
    status, model_id = "empty-catalog", None
    if items:
        from .cold_worker import recommend as research

        payload = dict(
            vectors=x,
            ids=[a["id"] for a in items],
            history=h,
            explicit=explicit,
            eligible=eligible,
            popularity=popularity,
            cf_scores=cf_scores,
        )
        completed, status, model_id = research(payload)
        if completed is not None:
            result = completed
    ids, features, base = result["ids"], result["features"], result["scores"]
    with online.lock:
        model = active_ranker()
        online_version = online.online_version
    scores = policy_scores(model, features, base) if ids else np.array([])
    order = np.lexsort((np.asarray(ids), -scores))[:limit]
    selected = [ids[i] for i in order]
    version = "coldfuse-" + (
        "online-" + online_version if model is not None else status
    )
    if model_id:
        version += "-" + model_id[:12]
    snapshot.update(
        created=created,
        feature_version=FEATURE_VERSION,
        model_version=version,
        history=[items[i]["id"] for i in h],
        interests=interests,
        content_scores={items[ids[i]]["id"]: float(base[i]) for i in order},
        features={items[ids[i]]["id"]: features[i].tolist() for i in order},
        coldfuse=dict(
            status=status,
            model_identity=model_id,
            gates=result["gates"],
            reliability=result["state"].reliability,
            anchors=[items[i]["id"] for i in result["channels"]["anchors"]],
        ),
    )
    return [
        {
            **items[i],
            "bookmarked": items[i]["id"] in saved,
            "reason": "兴趣加速 · 实验推荐"
            if "fallback" not in status
            else "兴趣推荐 · 加速准备中",
        }
        for i in selected
    ]
