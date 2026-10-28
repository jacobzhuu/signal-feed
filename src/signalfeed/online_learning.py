"""Retrain from mature real HN exposures, validate, atomically publish a bundle."""

import hashlib
import json
import os
import time
from collections import defaultdict
import numpy as np
import torch
from .settings import path
from .store import db
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


def retrain(min_exposures=1000, min_clicks=50):
    from .serving import ContentService
    from .evaluation import ranking_metrics, average

    mature = time.time() - 86400
    with db() as c:
        exposures = [
            dict(r)
            for r in c.execute(
                "SELECT * FROM events WHERE kind='impression' AND is_test=0 AND created<? ORDER BY created,id",
                (mature,),
            )
        ]
        clicks = [
            dict(r)
            for r in c.execute(
                "SELECT * FROM events WHERE kind IN ('click','bookmark') AND is_test=0 ORDER BY created,id"
            )
        ]
    click_pairs = {
        (r["request_id"], r["item_id"]) for r in clicks if r["kind"] == "click"
    }
    npositive = sum((r["request_id"], r["item_id"]) in click_pairs for r in exposures)
    if len(exposures) < min_exposures or npositive < min_clicks:
        result = dict(
            status="insufficient_real_feedback",
            mature_exposures=len(exposures),
            positive_exposures=npositive,
            required_exposures=min_exposures,
            required_clicks=min_clicks,
        )
        (path("reports") / "online_retrain.json").write_text(
            json.dumps(result, indent=2)
        )
        print(result)
        return result
    service = ContentService()
    service.refresh()
    index = service.index
    x = service.vectors
    histories = defaultdict(list)
    cursor = 0
    records = []
    for e in exposures:
        while cursor < len(clicks) and clicks[cursor]["created"] < e["created"]:
            click = clicks[cursor]
            if click["item_id"] in index:
                histories[click["user_id"]].append(index[click["item_id"]])
            cursor += 1
        if e["item_id"] not in index:
            continue
        feature = online_matrix(x, [index[e["item_id"]]], histories[e["user_id"]])[0]
        records.append(
            (e, feature, int((e["request_id"], e["item_id"]) in click_pairs))
        )
    # Group whole requests, not individual exposures, across the temporal split.
    request_times = {}
    for e, _, _ in records:
        request_times[e["request_id"]] = min(
            request_times.get(e["request_id"], e["created"]), e["created"]
        )
    reqs = sorted(request_times, key=request_times.get)
    valid_requests = set(reqs[int(0.8 * len(reqs)) :])
    train = [r for r in records if r[0]["request_id"] not in valid_requests]
    valid = [r for r in records if r[0]["request_id"] in valid_requests]
    if not train or not valid:
        raise ValueError("Not enough independent requests for chronological validation")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = Ranker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xx = torch.tensor(np.stack([r[1] for r in train]))
    yy = torch.tensor([r[2] for r in train], dtype=torch.float32)
    for epoch in range(5):
        model.train()
        for ids in torch.randperm(len(train)).split(256):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xx[ids]), yy[ids]
            )
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        scores = model(torch.tensor(np.stack([r[1] for r in valid]))).numpy()
    groups = defaultdict(list)
    for row, score in zip(valid, scores):
        groups[row[0]["request_id"]].append((row[2], float(score), float(row[1][-6])))
    neural = []
    content = []
    for group in groups.values():
        labels = [r[0] for r in group]
        if len(set(labels)) < 2:
            continue
        neural.append(ranking_metrics(labels, [r[1] for r in group]))
        content.append(ranking_metrics(labels, [r[2] for r in group]))
    a = average(neural)
    b = average(content)
    version = str(int(time.time()))
    dest = path(f"online-models/{version}")
    torch.save(model.state_dict(), dest / "model.pt")
    model_hash = hashlib.sha256((dest / "model.pt").read_bytes()).hexdigest()
    promoted = bool(
        len(neural) >= 20 and a["ndcg10"] > b["ndcg10"] and a["auc"] >= b["auc"]
    )
    report = dict(
        status="promoted" if promoted else "kept_for_research",
        version=version,
        feature_version="online-text-v1",
        model_sha256=model_hash,
        train_exposures=len(train),
        valid_exposures=len(valid),
        valid_requests=len(neural),
        dnn=a,
        content=b,
        labels="24h mature real exposures; test events excluded",
    )
    (dest / "manifest.json").write_text(json.dumps(report, indent=2))
    if promoted:
        pointer = path("online-models") / "active.json"
        temp = pointer.with_suffix(".tmp")
        temp.write_text(json.dumps(report))
        os.replace(temp, pointer)
    (path("reports") / "online_retrain.json").write_text(json.dumps(report, indent=2))
    print(report)
    return report
