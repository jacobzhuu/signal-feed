"""Retrain from mature real HN exposures, validate, atomically publish a bundle."""

import hashlib
import json
import os
import time
from collections import defaultdict
import numpy as np
import torch
from .settings import path
from .store import db, initialize
from .ranker import Ranker


def online_matrix(vectors, candidates, history):
    h = history[-20:]
    x = vectors[candidates]
    if h:
        profile = np.average(vectors[h], axis=0, weights=np.linspace(0.5, 1, len(h)))
        profile /= max(np.linalg.norm(profile), 1e-8)
        att = x @ vectors[h].T * 5
        att = np.exp(att - att.max(1, keepdims=True))
        att /= att.sum(1, keepdims=True)
        interest = att @ vectors[h]
        recent = x @ vectors[h[-1]]
    else:
        profile = np.zeros(vectors.shape[1])
        interest = np.zeros_like(x)
        recent = np.zeros(len(x))
    extras = np.stack(
        [
            x @ profile,
            recent,
            np.zeros(len(x)),
            np.full(len(x), len(h) / 20),
            np.zeros(len(x)),
            np.zeros(len(x)),
        ],
        1,
    )
    return np.concatenate([x, interest, x * interest, extras], 1).astype("float32")


FEATURE_VERSION = "online-text-v1"
LABEL_WINDOW = 86400


def feedback_records(now):
    """Read immutable serving features, never reconstruct them from event time.

    Labels cover [request, request + 24h); a click may precede an impression.
    Legacy requests and non-feed surfaces cannot supply comparable features.
    """
    with db() as c:
        c.execute("BEGIN")
        requests = [
            dict(r)
            for r in c.execute(
                "SELECT r.*,s.payload FROM requests r LEFT JOIN request_snapshots s "
                "ON s.request_id=r.id WHERE r.created+?<=? ORDER BY r.created,r.id",
                (LABEL_WINDOW, now),
            )
        ]
        events = defaultdict(list)
        for e in c.execute("SELECT * FROM events WHERE is_test=0 ORDER BY created,id"):
            events[e["request_id"]].append(dict(e))
    records = []
    excluded = defaultdict(int)
    for request in requests:
        rid = request["id"]
        start = request["created"]
        end = start + LABEL_WINDOW
        impressions = [e for e in events[rid] if e["kind"] == "impression"]
        if not impressions:
            continue
        if request["query"] or not (
            request["strategy"] == "content-v1"
            or request["strategy"].startswith("online-")
        ):
            excluded["non_feed"] += len(impressions)
            continue
        if not request["payload"]:
            excluded["missing_snapshot"] += len(impressions)
            continue
        snapshot = json.loads(request["payload"])
        if (
            snapshot.get("feature_version") != FEATURE_VERSION
            or snapshot.get("created") != start
            or snapshot.get("model_version") != request["model_version"]
        ):
            excluded["incompatible_snapshot"] += len(impressions)
            continue
        items = json.loads(request["items"])
        clicks = {
            e["item_id"]
            for e in events[rid]
            if e["kind"] == "click"
            and e["user_id"] == request["user_id"]
            and start <= e["created"] < end
        }
        for e in impressions:
            iid = e["item_id"]
            if not (
                start <= e["created"] < end
                and e["user_id"] == request["user_id"]
                and 0 <= e["position"] < len(items)
                and items[e["position"]] == iid
            ):
                excluded["outside_window_or_invalid_exposure"] += 1
                continue
            feature = np.asarray(
                snapshot.get("features", {}).get(iid, []), dtype="float32"
            )
            score = snapshot.get("content_scores", {}).get(iid)
            if (
                feature.shape != (1158,)
                or not np.isfinite(feature).all()
                or score is None
                or not np.isfinite(score)
            ):
                excluded["invalid_features"] += 1
                continue
            records.append(
                (
                    dict(
                        e,
                        request_created=start,
                        label_end=end,
                        content_score=float(score),
                    ),
                    feature,
                    int(iid in clicks),
                )
            )
    return records, dict(excluded)


def chronological_split(records, incumbent_data_until=None):
    times = {r[0]["request_id"]: r[0]["request_created"] for r in records}
    ordered = sorted(times, key=lambda rid: (times[rid], rid))
    if len(ordered) < 2:
        return [], [], len(records)
    cutoff = times[ordered[int(0.8 * len(ordered))]]
    valid = [
        r
        for r in records
        if r[0]["request_created"] >= cutoff
        and (
            incumbent_data_until is None
            or r[0]["request_created"] > incumbent_data_until
        )
    ]
    if not valid:
        return [], [], len(records)
    cutoff = min(r[0]["request_created"] for r in valid)
    # Purge requests whose label window crosses validation start. Equal request
    # timestamps always stay together; no future outcomes enter training.
    train = [r for r in records if r[0]["label_end"] <= cutoff]
    return train, valid, len(records) - len(train) - len(valid)


def compare_policies(valid, candidate_scores, incumbent_scores):
    from .evaluation import ranking_metrics, average

    groups = defaultdict(list)
    for row, candidate, incumbent in zip(valid, candidate_scores, incumbent_scores):
        groups[row[0]["request_id"]].append((row, float(candidate), float(incumbent)))
    candidate_metrics, incumbent_metrics, predictions = [], [], []
    for rid, group in groups.items():
        # Stable ties follow the original displayed order, not arrival of the
        # delayed impression events.
        group.sort(key=lambda row: row[0][0]["position"])
        labels = [row[0][2] for row in group]
        cs, bs = [row[1] for row in group], [row[2] for row in group]
        predictions.append(
            dict(
                request_id=rid,
                labels=labels,
                items=[row[0][0]["item_id"] for row in group],
                candidate_scores=cs,
                incumbent_scores=bs,
            )
        )
        if len(set(labels)) < 2:
            continue
        candidate_metrics.append(ranking_metrics(labels, cs))
        incumbent_metrics.append(ranking_metrics(labels, bs))
    a, b = average(candidate_metrics), average(incumbent_metrics)
    promote = bool(
        len(candidate_metrics) >= 20
        and a["ndcg10"] > b["ndcg10"]
        and a["auc"] >= b["auc"]
    )
    return a, b, len(candidate_metrics), promote, predictions


def retrain(min_exposures=1000, min_clicks=50):
    from .serving import ContentService

    initialize()

    def finish(result):
        (path("reports") / "online_retrain.json").write_text(
            json.dumps(result, indent=2)
        )
        print(result)
        return result

    now = time.time()
    records, excluded = feedback_records(now)
    summary = dict(
        mature_exposures=len(records),
        positive_exposures=sum(r[2] for r in records),
        required_exposures=min_exposures,
        required_clicks=min_clicks,
        excluded_exposures=excluded,
        evaluated_at=now,
    )
    if len(records) < min_exposures or summary["positive_exposures"] < min_clicks:
        return finish(dict(status="insufficient_real_feedback", **summary))

    pointer = path("online-models") / "active.json"
    pointer_before = pointer.read_bytes() if pointer.exists() else None
    service = ContentService()
    incumbent = service.active_ranker()
    incumbent_manifest = (
        json.loads(pointer_before) if pointer_before and incumbent is not None else {}
    )
    if pointer_before and (
        incumbent is None
        or incumbent_manifest.get("feature_version") != FEATURE_VERSION
        or not isinstance(incumbent_manifest.get("data_until"), (int, float))
    ):
        return finish(dict(status="incumbent_provenance_unavailable", **summary))
    train, valid, purged = chronological_split(
        records, incumbent_manifest.get("data_until")
    )
    if not train or not valid or len({r[2] for r in train}) < 2:
        return finish(
            dict(
                status="insufficient_chronological_feedback",
                **summary,
                train_exposures=len(train),
                valid_exposures=len(valid),
                purged_exposures=purged,
            )
        )
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = Ranker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xx = torch.tensor(np.stack([r[1] for r in train]))
    yy = torch.tensor([r[2] for r in train], dtype=torch.float32)
    losses = []
    for epoch in range(5):
        model.train()
        epoch_losses = []
        for ids in torch.randperm(len(train)).split(256):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xx[ids]), yy[ids]
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    vx = torch.tensor(np.stack([r[1] for r in valid]))
    with torch.inference_mode():
        scores = model(vx).numpy()
        # Evaluate the current deployed ranker, even if requests were served by
        # an older version. The content fallback uses its complete frozen score.
        baseline = (
            incumbent(vx).numpy()
            if incumbent is not None
            else np.array([r[0]["content_score"] for r in valid])
        )
    a, b, nvalid, promoted, predictions = compare_policies(valid, scores, baseline)
    version = str(time.time_ns())
    dest = path(f"online-models/{version}")
    torch.save(model.state_dict(), dest / "model.pt")
    model_hash = hashlib.sha256((dest / "model.pt").read_bytes()).hexdigest()
    pointer_now = pointer.read_bytes() if pointer.exists() else None
    policy_changed = pointer_now != pointer_before
    promoted = promoted and not policy_changed
    report = dict(
        status="promoted" if promoted else "kept_for_research",
        version=version,
        feature_version=FEATURE_VERSION,
        model_sha256=model_hash,
        **summary,
        train_exposures=len(train),
        valid_exposures=len(valid),
        purged_exposures=purged,
        valid_requests=nvalid,
        train_label_end=max(r[0]["label_end"] for r in train),
        valid_request_start=min(r[0]["request_created"] for r in valid),
        data_until=max(r[0]["label_end"] for r in valid),
        dnn=a,
        incumbent=b,
        incumbent_version=service.online_version or "content-v1",
        incumbent_model_sha256=incumbent_manifest.get("model_sha256"),
        policy_changed_during_training=policy_changed,
        train_losses=losses,
        labels="request-time snapshot; fixed 24h click window; test events excluded",
        evaluation_scope="same observed feed exposures per held-out request; not full-catalog or causal CTR",
    )
    (dest / "manifest.json").write_text(json.dumps(report, indent=2))
    (dest / "validation_predictions.json").write_text(json.dumps(predictions, indent=2))
    if promoted:
        temp = pointer.with_suffix(".tmp")
        temp.write_text(json.dumps(report))
        os.replace(temp, pointer)
    return finish(report)
