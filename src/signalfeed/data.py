"""Chronological, impression-preserving MIND preparation."""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
from .settings import path, ENCODER_MODEL, ENCODER_REVISION


def read_behaviors(file):
    rows = []
    with open(file) as f:
        for line in f:
            iid, user, stamp, hist, candidates = line.rstrip("\n").split("\t")
            pairs = [p.rsplit("-", 1) for p in candidates.split()]
            rows.append(
                dict(
                    id=iid,
                    user=user,
                    time=datetime.strptime(stamp, "%m/%d/%Y %I:%M:%S %p")
                    .replace(tzinfo=timezone.utc)
                    .timestamp(),
                    history=hist.split()[-20:],
                    items=[x[0] for x in pairs],
                    labels=[int(x[1]) for x in pairs],
                )
            )
    return sorted(rows, key=lambda x: (x["time"], x["id"]))


def prepare():
    raw = path("raw")
    out = path("mind")
    news = {}
    for split in ["train", "dev"]:
        with open(raw / split / "news.tsv") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                iid, cat, sub, title, abstract, url = fields[:6]
                news[iid] = dict(
                    id=iid,
                    title=title,
                    abstract=abstract,
                    category=cat,
                    subcategory=sub,
                    url=url,
                )
    train = read_behaviors(raw / "train/behaviors.tsv")
    test = read_behaviors(raw / "dev/behaviors.tsv")
    cutoff = train[int(len(train) * 0.9)]["time"]
    parts = {
        "train": [r for r in train if r["time"] < cutoff],
        "valid": [r for r in train if r["time"] >= cutoff],
        "test": test,
    }
    # Catalog availability proxy uses first occurrence, never future click labels.
    first = {}
    for r in train + test:
        for iid in r["history"] + r["items"]:
            first[iid] = min(first.get(iid, float("inf")), r["time"])
    ids = sorted(news)
    for iid in ids:
        news[iid]["first_seen"] = first.get(iid, float("inf"))
    (out / "news.json").write_text(json.dumps([news[i] for i in ids]))
    for split, rows in parts.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    summary = {
        "items": len(ids),
        "split_counts": {s: len(r) for s, r in parts.items()},
        "cutoff": cutoff,
        "train_max_time": parts["train"][-1]["time"],
        "test_min_time": test[0]["time"],
        "availability": "first log appearance proxy, not publication time",
        "sha256": {
            f"{s}/{n}": hashlib.sha256((raw / s / n).read_bytes()).hexdigest()
            for s in ["train", "dev"]
            for n in ["news.tsv", "behaviors.tsv"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(summary)


def load_rows(split, limit=0):
    with open(path("mind") / f"{split}.jsonl") as f:
        rows = [json.loads(line) for line in f]
    if limit and len(rows) > limit:
        # Deterministic uniform sample; not first impressions of a single time block.
        indices = np.random.default_rng(42).choice(len(rows), limit, replace=False)
        rows = [rows[i] for i in sorted(indices)]
    return rows


def encode():
    from sentence_transformers import SentenceTransformer

    out = path("mind")
    news = json.loads((out / "news.json").read_text())
    model = SentenceTransformer(ENCODER_MODEL, revision=ENCODER_REVISION, device="cuda")
    vectors = model.encode(
        [n["title"] + ". " + n["abstract"] for n in news],
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(out / "embeddings.npy", vectors.astype("float32"))
    model.save(str(path("models") / "encoder"))
