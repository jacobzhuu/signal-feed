"""Label-blind natural-history sampling excluding all previous research users."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from signalfeed.provenance import sha256

LENGTHS = (1, 2, 3, 4, 5, 10)
LIMITS = dict(select=100, guard=200, test=400)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def partition(user):
    slot = int(digest("interest-v3/split/" + user), 16) % 10
    return "select" if slot < 3 else "guard" if slot < 6 else "test"


def period(stamp):
    return "early" if datetime.fromtimestamp(stamp, timezone.utc).hour < 12 else "late"


def prepare(run, previous, failed, baseline):
    out = run / "dataset"
    out.mkdir()  # Partial preparation is never silently reused.
    previous, failed, baseline = (
        previous.resolve(),
        failed.resolve(),
        baseline.resolve(),
    )
    old_manifest = json.loads((previous / "dataset/manifest.json").read_text())
    if old_manifest["protocol"] != "adaptation-v2-canonical-identity":
        raise ValueError("Require canonical identity")
    for name, meta in old_manifest["files"].items():
        if sha256(previous / "dataset" / name) != meta["sha256"]:
            raise ValueError("Changed previous dataset: " + name)
    excluded = set(
        json.loads((previous / "dataset/excluded_large_users.json").read_text())
    )
    old_small_count = len(excluded)
    sources = {}
    prior_users = set()
    for parent in (previous, failed):
        for f in sorted((parent / "dataset").glob("*-*.jsonl")):
            sources[str(f)] = sha256(f)
            prior_users.update(json.loads(line)["user"] for line in f.open())
    excluded.update(prior_users)
    (out / "excluded_users.json").write_text(json.dumps(sorted(excluded)))
    mapping = json.loads((previous / "dataset/large_to_canonical.json").read_text())
    news = json.loads((previous / "dataset/news.json").read_text())
    catalog = {n["id"] for n in news}
    raw = previous / "raw"
    for split in ("train", "dev"):
        sources[str(raw / split / "behaviors.tsv")] = sha256(
            raw / split / "behaviors.tsv"
        )
    train_users = set()
    for line in (raw / "train/behaviors.tsv").open():
        train_users.add(line.split("\t", 2)[1])
    earliest, first_dev, stamps = {}, {}, {}
    skipped = Counter()
    for line in (raw / "dev/behaviors.tsv").open():
        iid, user, stamp, history, impressions = line.rstrip("\n").split("\t")
        if user in excluded:
            continue
        if stamp not in stamps:
            stamps[stamp] = (
                datetime.strptime(stamp, "%m/%d/%Y %I:%M:%S %p")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        time = stamps[stamp]
        key = (time, iid)
        first_dev[user] = min(first_dev.get(user, key), key)
        raw_h = history.split()
        if len(raw_h) not in LENGTHS:
            continue
        h = [mapping[i] for i in raw_h]
        items = [mapping[p.rsplit("-", 1)[0]] for p in impressions.split()]
        if not set(h + items) <= catalog:
            skipped["unknown_catalog"] += 1
            continue
        if user in earliest and (earliest[user]["time"], earliest[user]["id"]) <= key:
            continue
        earliest[user] = dict(
            id=iid,
            user=user,
            time=time,
            history=h,
            items=items,
            raw_history=len(raw_h),
            unique_history=len(set(h)),
            known_large_train=user in train_users,
            raw_impressions=impressions,
            period=period(time),
        )
    pools = defaultdict(list)
    for user, row in earliest.items():
        pools[partition(user), row["raw_history"], row["period"]].append(row)
    files, counts, selected = {}, {}, set()
    for split, limit in LIMITS.items():
        rows = []
        for k in LENGTHS:
            for segment in ("early", "late"):
                pool = sorted(
                    pools[split, k, segment],
                    key=lambda r: digest("interest-v3/sample/" + r["user"]),
                )
                counts[f"{split}/{k}/{segment}"] = dict(
                    available=len(pool), selected=min(limit, len(pool))
                )
                rows += pool[:limit]
        rows.sort(key=lambda r: digest("interest-v3/sample/" + r["user"]))
        with (out / f"{split}.jsonl").open("x") as stream:
            for row in rows:
                assert row["user"] not in selected and row["user"] not in excluded
                selected.add(row["user"])
                row["first_dev_request"] = (row["time"], row["id"]) == first_dev[
                    row["user"]
                ]
                row["labels"] = [
                    int(p.rsplit("-", 1)[1]) for p in row.pop("raw_impressions").split()
                ]
                stream.write(json.dumps(row) + "\n")
        files[f"{split}.jsonl"] = sha256(out / f"{split}.jsonl")
    files["excluded_users.json"] = sha256(out / "excluded_users.json")
    for name in ("news.json", "embeddings.npy"):
        (out / name).symlink_to(previous / "dataset" / name)
        files[name] = sha256(out / name)
    baseline_files = {
        name: sha256(baseline / "mind" / name) for name in ("pop.npy", "cf.npz")
    }
    manifest = dict(
        protocol="interest-v3",
        previous=str(previous),
        baseline=str(baseline),
        previous_manifest_sha256=sha256(previous / "dataset/manifest.json"),
        frozen_previous_sha256=sha256(run / "frozen_previous.json"),
        protocol_sha256=sha256(Path("docs/INTEREST_V3_PROTOCOL.md")),
        files=files,
        baseline_files=baseline_files,
        prior_sample_files=sources,
        old_small_excluded=old_small_count,
        prior_sample_users=len(prior_users),
        all_excluded=len(excluded),
        selected_users=len(selected),
        counts=counts,
        skipped=dict(skipped),
        labels_used_for_sampling=False,
        old_items=old_manifest["old_items"],
        items=len(news),
        dates=sorted(
            {
                datetime.fromtimestamp(r["time"], timezone.utc).date().isoformat()
                for r in earliest.values()
            }
        ),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("all_excluded", "selected_users", "counts", "dates")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("run", "previous", "failed", "baseline"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.run, a.previous, a.failed, a.baseline)
