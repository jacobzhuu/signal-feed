"""Retrain from mature real HN exposures, validate, atomically publish a bundle."""

import hashlib
import json
import time
from collections import defaultdict, deque
import numpy as np
import torch
from .settings import path
from .store import db, initialize
from .ranker import Ranker


from .online_schema import (
    online_matrix as online_matrix,
    policy_scores,
    FEATURE_VERSION,
    LABEL_WINDOW,
    CONTRACT,
)
from .snapshots import unpack
from .publication import publication_lock, atomic_json
from .feedback_policy import load_feedback_policy


def feedback_records(now, max_exposures=50000, *, cold=False, policy=None):
    """Stream only requests with feedback; retain bounded complete request groups.

    A valid click proves exposure even if the one-second view event never arrives.
    Viewable and click-inferred exposure counts remain distinguishable.
    """
    policy = load_feedback_policy() if policy is None else policy
    batches, size = deque(), 0
    excluded = defaultdict(int)
    with db() as c:
        c.execute("BEGIN")
        requests = c.execute(
            "SELECT r.*,s.payload FROM requests r LEFT JOIN request_snapshots s ON s.request_id=r.id "
            "WHERE r.created+?<=? AND EXISTS(SELECT 1 FROM events e WHERE e.request_id=r.id "
            "AND e.is_test=0 AND e.kind IN ('impression','click')) ORDER BY r.created,r.id",
            (LABEL_WINDOW, now),
        )
        for request in requests:
            if request["user_id"] in policy.user_ids:
                excluded["test_user_requests"] += 1
                continue
            rid, start = request["id"], request["created"]
            end = start + LABEL_WINDOW
            events = [
                dict(e)
                for e in c.execute(
                    "SELECT * FROM events WHERE request_id=? AND is_test=0 AND kind IN ('impression','click') ORDER BY created,id",
                    (rid,),
                )
            ]
            feed = (
                request["strategy"].startswith("coldfuse-")
                if cold
                else (
                    request["strategy"] == "content-v1"
                    or request["strategy"].startswith("online-")
                )
            )
            if request["query"] or not feed:
                excluded["non_feed"] += sum(e["kind"] == "impression" for e in events)
                continue
            if not request["payload"]:
                excluded["missing_snapshot"] += sum(
                    e["kind"] == "impression" for e in events
                )
                continue
            try:
                snapshot = unpack(request["payload"])
                from .coldfuse import FEATURE_VERSION as COLD_FEATURE_VERSION

                compatible = (
                    snapshot.get("feature_version")
                    == (COLD_FEATURE_VERSION if cold else FEATURE_VERSION)
                    and snapshot.get("created") == start
                    and snapshot.get("model_version") == request["model_version"]
                )
            except (ValueError, TypeError, KeyError):
                compatible = False
            if not compatible:
                excluded["incompatible_snapshot"] += sum(
                    e["kind"] == "impression" for e in events
                )
                continue
            items = json.loads(request["items"])
            valid_events = []
            for e in events:
                if not (
                    start <= e["created"] < end
                    and e["user_id"] == request["user_id"]
                    and 0 <= e["position"] < len(items)
                    and items[e["position"]] == e["item_id"]
                ):
                    if e["kind"] == "impression":
                        excluded["outside_window_or_invalid_exposure"] += 1
                    continue
                valid_events.append(e)
            clicks = {e["item_id"] for e in valid_events if e["kind"] == "click"}
            exposures = {
                e["item_id"]: dict(e, exposure_kind="viewable")
                for e in valid_events
                if e["kind"] == "impression"
            }
            for e in valid_events:
                if e["kind"] == "click" and e["item_id"] not in exposures:
                    exposures[e["item_id"]] = dict(e, exposure_kind="click_inferred")
            batch = []
            for iid, e in exposures.items():
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
                batch.append(
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
            if batch:
                batches.append(batch)
                size += len(batch)
                while size > max_exposures and batches:
                    removed = len(batches.popleft())
                    size -= removed
                    excluded["capacity_pruned_exposures"] += removed
    return [r for batch in batches for r in batch], dict(excluded)


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


def compare_policies(valid, candidate_scores, incumbent_scores, diagnostics=None):
    from .evaluation import ranking_metrics, average

    groups = defaultdict(list)
    for row, candidate, incumbent in zip(valid, candidate_scores, incumbent_scores):
        groups[row[0]["request_id"]].append((row, float(candidate), float(incumbent)))
    candidate_metrics, incumbent_metrics, predictions = [], [], []
    clusters = defaultdict(list)
    cohorts = defaultdict(list)
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
        cm, bm = ranking_metrics(labels, cs), ranking_metrics(labels, bs)
        candidate_metrics.append(cm)
        incumbent_metrics.append(bm)
        delta = cm["ndcg10"] - bm["ndcg10"]
        user = group[0][0][0].get("user_id", rid)
        clusters[user].append(delta)
        history = round(float(group[0][0][1][-3]) * 20)
        cohort = "empty" if history == 0 else "sparse" if history < 5 else "warm"
        cohorts[cohort].append((delta, cm["auc"] - bm["auc"]))
    a, b = average(candidate_metrics), average(incumbent_metrics)
    rng = np.random.default_rng(42)
    cluster_values = list(clusters.values())
    ci = None
    if cluster_values:
        sums = np.array([sum(v) for v in cluster_values])
        counts = np.array([len(v) for v in cluster_values])
        draws = rng.integers(len(sums), size=(1000, len(sums)))
        means = sums[draws].sum(1) / counts[draws].sum(1)
        ci = np.percentile(means, [2.5, 97.5]).tolist()
    cohort_safe = all(
        len(v) >= 20
        and np.mean([x[0] for x in v]) >= 0
        and np.mean([x[1] for x in v]) >= 0
        for v in cohorts.values()
    )
    if diagnostics is not None:
        diagnostics.update(
            ndcg10_user_bootstrap_ci95=ci,
            independent_users=len(clusters),
            cohorts={
                k: {
                    "requests": len(v),
                    "ndcg10_delta": float(np.mean([x[0] for x in v])),
                    "auc_delta": float(np.mean([x[1] for x in v])),
                }
                for k, v in cohorts.items()
            },
        )
    promote = bool(
        ci is not None
        and ci[0] > 0
        and len(clusters) >= 20
        and cohort_safe
        and len(candidate_metrics) >= 20
        and a["ndcg10"] > b["ndcg10"]
        and a["auc"] >= b["auc"]
    )
    return a, b, len(candidate_metrics), promote, predictions


def retrain(min_exposures=1000, min_clicks=50, *, cold=False):
    with publication_lock(path("cold-online-models" if cold else "online-models")):
        return _retrain(min_exposures, min_clicks, cold=cold)


def _retrain(min_exposures=1000, min_clicks=50, *, cold=False):
    from .serving import ContentService

    policy = load_feedback_policy()
    initialize()
    from . import coldfuse

    contract = coldfuse.CONTRACT if cold else CONTRACT
    scoring = coldfuse.policy_scores if cold else policy_scores
    directory = "cold-online-models" if cold else "online-models"

    def finish(result):
        (
            path("reports")
            / ("cold_online_retrain.json" if cold else "online_retrain.json")
        ).write_text(json.dumps(result, indent=2))
        print(result)
        return result

    now = time.time()
    records, excluded = feedback_records(now, cold=cold, policy=policy)
    summary = dict(
        mature_exposures=len(records),
        positive_exposures=sum(r[2] for r in records),
        required_exposures=min_exposures,
        required_clicks=min_clicks,
        excluded_exposures=excluded,
        feedback_policy=policy.metadata(),
        evaluated_at=now,
    )
    if len(records) < min_exposures or summary["positive_exposures"] < min_clicks:
        return finish(dict(status="insufficient_real_feedback", **summary))

    pointer = path(directory) / "active.json"
    pointer_before = pointer.read_bytes() if pointer.exists() else None
    service = ContentService()
    incumbent = (
        service._active_ranker(contract, directory) if cold else service.active_ranker()
    )
    incumbent_manifest = (
        json.loads(pointer_before) if pointer_before and incumbent is not None else {}
    )
    if pointer_before and (
        incumbent is None
        or incumbent_manifest.get("feature_version") != contract["feature_version"]
        or not isinstance(incumbent_manifest.get("data_until"), (int, float))
    ):
        return finish(dict(status="incumbent_provenance_unavailable", **summary))
    ledger = path(directory) / "validation.json"
    used_until = (
        json.loads(ledger.read_text())["consumed_until"] if ledger.exists() else None
    )
    boundaries = [
        x for x in [used_until, incumbent_manifest.get("data_until")] if x is not None
    ]
    train, valid, purged = chronological_split(
        records, max(boundaries) if boundaries else None
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
    # Consume evaluation data even when the candidate is rejected or training fails.
    atomic_json(
        ledger,
        {"consumed_until": max(r[0]["label_end"] for r in valid), "reserved_at": now},
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
    content = np.array([r[0]["content_score"] for r in valid])
    scores = scoring(model, vx.numpy(), content)
    baseline = scoring(incumbent, vx.numpy(), content)
    diagnostics = {}
    a, b, nvalid, promoted, predictions = compare_policies(
        valid, scores, baseline, diagnostics
    )
    version = str(time.time_ns())
    dest = path(f"{directory}/{version}")
    torch.save(model.state_dict(), dest / "model.pt")
    model_hash = hashlib.sha256((dest / "model.pt").read_bytes()).hexdigest()
    pointer_now = pointer.read_bytes() if pointer.exists() else None
    policy_changed = pointer_now != pointer_before
    promoted = promoted and not policy_changed
    report = dict(
        status="promoted" if promoted else "kept_for_research",
        version=version,
        **contract,
        promotion_diagnostics=diagnostics,
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
        labels="request-time snapshot; mature 24h viewable OR click-inferred exposure; test events excluded",
        click_inferred_exposures=sum(
            r[0]["exposure_kind"] == "click_inferred" for r in records
        ),
        evaluation_scope="same observed feed exposures per held-out request; not full-catalog or causal CTR",
    )
    (dest / "manifest.json").write_text(json.dumps(report, indent=2))
    (dest / "validation_predictions.json").write_text(json.dumps(predictions, indent=2))
    if promoted:
        atomic_json(pointer, report)
    return finish(report)
