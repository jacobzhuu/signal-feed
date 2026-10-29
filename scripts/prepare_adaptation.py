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


def news_mapping(old_news, large_news):
    """Numeric IDs are independently anonymized in small and large releases."""
    keys = ("category", "subcategory", "title", "abstract", "url")
    content = {tuple(n[k] for k in keys): n["id"] for n in old_news}
    if len(content) != len(old_news):
        raise ValueError("Ambiguous frozen article identity")
    mapping, texts = {}, {}
    for n in large_news:
        identity = tuple(n[k] for k in keys)
        if n["id"] in texts and texts[n["id"]] != identity:
            raise ValueError("Large train/dev article identity conflict")
        texts[n["id"]] = identity
        mapping[n["id"]] = content.get(identity, "large:" + n["id"])
    if not set(content.values()).issubset(mapping.values()):
        raise ValueError("Cannot reconcile all frozen articles by exact content")
    return mapping


def event_signature(stamp, history, items):
    # Labels and local impression/user identifiers deliberately do not enter.
    return digest(json.dumps([stamp, history[-20:], sorted(items)]))


def canonical_exclusions(root, baseline, mapping):
    signatures = set()
    for split in ("train", "valid", "test"):
        with (baseline / f"mind/{split}.jsonl").open() as stream:
            for line in stream:
                r = json.loads(line)
                signatures.add(event_signature(r["time"], r["history"], r["items"]))
    excluded, matched, stamps = set(), set(), {}
    for split in ("train", "dev"):
        for _, user, stamp, hist, candidates in fields(
            root / f"raw/{split}/behaviors.tsv"
        ):
            if stamp not in stamps:
                stamps[stamp] = (
                    datetime.strptime(stamp, "%m/%d/%Y %I:%M:%S %p")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            history = [mapping[i] for i in hist.split()[-20:]]
            items = [mapping[p.rsplit("-", 1)[0]] for p in candidates.split()]
            signature = event_signature(stamps[stamp], history, items)
            if signature in signatures:
                matched.add(signature)
                excluded.add(user)
    if matched != signatures:
        raise ValueError(
            f"Only {len(matched)}/{len(signatures)} exposed events reconciled; independence unproven"
        )
    audit = dict(
        exposed_event_signatures=len(signatures),
        matched_event_signatures=len(matched),
        excluded_large_users=len(excluded),
        labels_used=False,
        identity="exact category/subcategory/title/abstract/url; time + last20 canonical history + unordered exposure IDs",
        duplicate_event_policy="exclude every matching large user conservatively",
    )
    return excluded, audit


def prepare(root, baseline):
    out = root / "dataset"
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Prepared data already sealed; use a new directory")
    news = json.loads((baseline / "mind/news.json").read_text())
    old_count = len(news)
    index = {n["id"]: i for i, n in enumerate(news)}
    hashes = {}
    for split in ("train", "dev"):
        archive = root / f"raw/MINDlarge_{split}.zip"
        hashes[archive.name] = sha256(archive)
        dest = root / f"raw/{split}"
        with zipfile.ZipFile(archive) as z:
            for name in ("news.tsv", "behaviors.tsv"):
                z.extract(name, dest)
    large_news = []
    for split in ("train", "dev"):
        for row in fields(root / f"raw/{split}/news.tsv"):
            large_news.append(
                dict(
                    zip(
                        ("id", "category", "subcategory", "title", "abstract", "url"),
                        row[:6],
                    )
                )
            )
    mapping = news_mapping(news, large_news)
    excluded, identity_audit = canonical_exclusions(root, baseline, mapping)
    print(dict(identity_audit=identity_audit), flush=True)
    # Only articles in the frozen catalog or new dev catalog are needed.
    for row in fields(root / "raw/dev/news.tsv"):
        iid, cat, sub, title, abstract, url = row[:6]
        iid = mapping[iid]
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
            history = [mapping[i] for i in hist.split()[-20:]]
            # Availability uses IDs only, never positivity.
            pairs = [v.rsplit("-", 1) for v in candidates.split()]
            pairs = [(mapping[i], label) for i, label in pairs]
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
    for name, obj in (
        ("identity_audit.json", identity_audit),
        ("large_to_canonical.json", mapping),
        ("excluded_large_users.json", sorted(excluded)),
    ):
        (out / name).write_text(json.dumps(obj, indent=2))
        files[name] = dict(sha256=sha256(out / name))
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
        protocol="adaptation-v2-canonical-identity",
        baseline_commit="b747819",
        baseline=str(baseline),
        encoder_revision=ENCODER_REVISION,
        archive_sha256=hashes,
        files=files,
        old_items=old_count,
        items=len(news),
        excluded_large_users=len(excluded),
        identity_audit=identity_audit,
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
