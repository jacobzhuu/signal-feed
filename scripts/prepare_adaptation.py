"""Prepare user-disjoint large-MIND research data without fitting to new labels.

Frozen models stay bound to the original small-MIND files. Extended arrays are
an explicit external-catalog adapter, never substituted into those bundles.
"""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from signalfeed.provenance import sha256


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def partition(user):
    slot = int(digest("adaptation-v1/" + user), 16) % 10
    return "select" if slot < 4 else "guard" if slot < 7 else "test"


def fields(file):
    with file.open() as stream:
        for line in stream:
            yield line.rstrip("\n").split("\t")


def prepare(root, baseline):
    out = root / "dataset"
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Prepared data already sealed; use a new directory")
    news = json.loads((baseline / "mind/news.json").read_text())
    old_count = len(news)
    index = {n["id"]: i for i, n in enumerate(news)}
    excluded = set()
    for part in ("train", "valid", "test"):
        with (baseline / f"mind/{part}.jsonl").open() as stream:
            for line in stream:
                excluded.add(json.loads(line)["user"])
    hashes = {}
    for split in ("train", "dev"):
        archive = root / f"raw/MINDlarge_{split}.zip"
        hashes[archive.name] = sha256(archive)
        dest = root / f"raw/{split}"
        with zipfile.ZipFile(archive) as z:
            for name in ("news.tsv", "behaviors.tsv"):
                z.extract(name, dest)
    # Only articles in the frozen catalog or new dev catalog are needed.
    for row in fields(root / "raw/dev/news.tsv"):
        iid, cat, sub, title, abstract, url = row[:6]
        if iid not in index:
            index[iid] = len(news)
            news.append(
                dict(
                    id=iid,
                    category=cat,
                    subcategory=sub,
                    title=title,
                    abstract=abstract,
                    url=url,
                    first_seen=float("inf"),
                )
            )
    first = {n["id"]: n["first_seen"] for n in news}
    train_users, first_dev, main, natural = set(), {}, {}, {}
    stamps = {}
    counts = {}
    for split in ("train", "dev"):
        total = 0
        for iid, user, stamp, hist, candidates in fields(
            root / f"raw/{split}/behaviors.tsv"
        ):
            total += 1
            if stamp not in stamps:
                stamps[stamp] = (
                    datetime.strptime(stamp, "%m/%d/%Y %I:%M:%S %p")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            stamp = stamps[stamp]
            history = hist.split()[-20:]
            # Availability uses IDs only, never positivity.
            pairs = [v.rsplit("-", 1) for v in candidates.split()]
            for item in history + [p[0] for p in pairs]:
                if item in first:
                    first[item] = min(first[item], stamp)
            if split == "train":
                train_users.add(user)
                continue
            key = (stamp, iid)
            first_dev[user] = min(first_dev.get(user, key), key)
            if user in excluded:
                continue
            known_history = [i for i in history if i in index]
            n = len(known_history)
            cohort = main if n >= 15 else natural if n <= 5 else None
            if cohort is None or (
                user in cohort and (cohort[user]["time"], cohort[user]["id"]) <= key
            ):
                continue
            cohort[user] = dict(
                id=iid,
                user=user,
                time=stamp,
                history=history,
                items=[v[0] for v in pairs],
                labels=[int(v[1]) for v in pairs],
                known_history=n,
                raw_history=len(hist.split()),
                known_large_train=user in train_users,
            )
        counts[split] = total
        print(dict(scanned=split, rows=total), flush=True)
    # Resolve overlap using metadata before any label-based diagnostics.
    natural = {u: r for u, r in natural.items() if u not in main}
    files, users = {}, {}
    for split in ("select", "guard", "test"):
        for name, pool in (("main", main), ("natural", natural)):
            limit = 1200 if split == "test" and name == "main" else 600
            rows = [r for u, r in pool.items() if partition(u) == split]
            rows.sort(key=lambda r: digest("sample-v1/" + r["user"]))
            rows = rows[:limit]
            for row in rows:
                row["first_dev_request"] = (row["time"], row["id"]) == first_dev[
                    row["user"]
                ]
            file = out / f"{split}-{name}.jsonl"
            with file.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            files[file.name] = dict(sha256=sha256(file), users=len(rows))
            users[split + "-" + name] = {r["user"] for r in rows}
    keys = list(users)
    assert all(
        not users[a] & users[b] for j, a in enumerate(keys) for b in keys[j + 1 :]
    )
    for row in news:
        row["first_seen"] = first[row["id"]]
    (out / "news.json").write_text(json.dumps(news))
    print(
        dict(
            catalog=len(news),
            frozen=old_count,
            new=len(news) - old_count,
            cohorts=files,
        ),
        flush=True,
    )
    import torch
    from sentence_transformers import SentenceTransformer
    from signalfeed.settings import ENCODER_REVISION
    from signalfeed.sid import ResidualCodes

    torch.set_num_threads(4)
    cache = (
        baseline
        / "hf/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        / ENCODER_REVISION
    )
    model = SentenceTransformer(str(cache), device="cpu")
    fresh = model.encode(
        [n["title"] + ". " + n["abstract"] for n in news[old_count:]],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.concatenate(
        [np.load(baseline / "mind/embeddings.npy"), fresh.astype("float32")]
    )
    np.save(out / "embeddings.npy", vectors)
    quant = ResidualCodes.load(baseline / "mind/quantizer.npz")
    codes = np.concatenate([np.load(baseline / "mind/codes.npy"), quant.encode(fresh)])
    np.save(out / "codes.npy", codes)
    for name in ("news.json", "embeddings.npy", "codes.npy"):
        files[name] = dict(sha256=sha256(out / name))
    manifest = dict(
        protocol="adaptation-v1",
        baseline_commit="b747819",
        baseline=str(baseline),
        encoder_revision=ENCODER_REVISION,
        archive_sha256=hashes,
        files=files,
        old_items=old_count,
        items=len(news),
        excluded_small_users=len(excluded),
        rows_scanned=counts,
        labels_used_for_sampling=False,
        note="User-disjoint, same historical period; not a future-time or real signup test",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    args = p.parse_args()
    prepare(args.run, args.baseline)
