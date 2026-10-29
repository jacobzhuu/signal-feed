"""Sealed natural-user selection, admission and final evaluation; no serving changes."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from .adaptation import AdaptConfig, signals, interest_scores
from .adaptation_experiments import paired_test, holm
from .coldfuse import state, itemcf_scores
from .stream_interest import InterestConfig, InterestState, GRID, VERSION
from .evaluation import recall_metrics, ranking_metrics, average
from .provenance import sha256, code_identity
from .retrieval import top

METHODS = (
    "content_safe",
    "old_interest",
    "interest",
    "no_multi",
    "no_reliability",
    "no_recency",
    "popularity",
    "itemcf",
)
FROZEN = (
    "stream_interest.py",
    "interest_experiments.py",
    "adaptation.py",
    "adaptation_experiments.py",
    "coldfuse.py",
    "evaluation.py",
    "retrieval.py",
    "provenance.py",
)


def identity():
    base = Path(__file__).parent
    result = {name: sha256(base / name) for name in FROZEN}
    root = base.parents[1]
    for name in ("scripts/prepare_interest.py", "docs/INTEREST_V3_PROTOCOL.md"):
        result[name] = sha256(root / name)
    return result


def write_json(file, value):
    with file.open("x") as stream:
        json.dump(value, stream, indent=2)


def load_catalog(run):
    dataset = run / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["protocol"] != "interest-v3":
        raise ValueError("Wrong protocol")
    for name, checksum in manifest["files"].items():
        if sha256(dataset / name) != checksum:
            raise ValueError("Changed dataset " + name)
    baseline = Path(manifest["baseline"]) / "mind"
    for name, checksum in manifest["baseline_files"].items():
        if sha256(baseline / name) != checksum:
            raise ValueError("Changed baseline " + name)
    news = json.loads((dataset / "news.json").read_text())
    x = np.load(dataset / "embeddings.npy")
    pop = np.pad(np.load(baseline / "pop.npy"), (0, len(news) - manifest["old_items"]))
    old = sparse.load_npz(baseline / "cf.npz").tocoo()
    cf = sparse.csr_matrix((old.data, (old.row, old.col)), shape=(len(news), len(news)))
    return SimpleNamespace(
        x=x,
        pop=pop,
        cf=cf,
        index={n["id"]: i for i, n in enumerate(news)},
        first=np.array([n["first_seen"] for n in news]),
        manifest=manifest,
    )


def read_rows(run, split):
    return [json.loads(line) for line in (run / f"dataset/{split}.jsonl").open()]


def inputs(catalog, row):
    h = [catalog.index[i] for i in row["history"]]
    items = [catalog.index[i] for i in row["items"]]
    targets = [i for i, y in zip(items, row["labels"]) if y]
    eligible = catalog.first < row["time"]
    eligible[h] = False
    return h, items, targets, eligible


def method_scores(catalog, history, config, method):
    """Independent invocation for latency; shares no computed request scores."""
    if method == "popularity":
        return catalog.pop
    if method == "itemcf":
        return itemcf_scores(catalog.cf, history)
    if method in ("content_safe", "no_multi"):
        s = state(catalog.x, history)
        return catalog.x @ s.profile if s.history else catalog.pop
    if method == "old_interest":
        return (
            interest_scores(
                signals(catalog.x, history), AdaptConfig(multi=1, recent=0.25, side=0)
            )
            if history
            else catalog.pop
        )
    s = InterestState.from_history(
        catalog.x, history, decay=1 if method == "no_recency" else 0.85
    )
    return s.scores(s.project(), config, method) if s.n else catalog.pop


def candidates(score, eligible, method):
    return top(score, eligible & (score > 0) if method == "itemcf" else eligible, 100)


def evaluate(run, split):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
    frozen = identity()
    start = time.perf_counter()
    catalog = load_catalog(run)
    selected, decision = None, None
    if split != "select":
        selected = json.loads((run / "selection.json").read_text())
        if selected["algorithm"] != frozen or selected["dataset_sha256"] != sha256(
            run / "dataset/manifest.json"
        ):
            raise ValueError("Selection binding changed")
    if split == "test":
        decision = json.loads((run / "guard.json").read_text())
        if (
            decision["selection_sha256"] != sha256(run / "selection.json")
            or decision["algorithm"] != frozen
        ):
            raise ValueError("Admission binding changed")
    configs = GRID if selected is None else (InterestConfig(**selected["config"]),)
    rows = read_rows(run, split)
    dest = run / "reports" / split
    dest.mkdir(parents=True, exist_ok=False)
    counts = defaultdict(list)
    with (dest / "predictions.jsonl").open("x") as stream:
        for n, row in enumerate(rows):
            h, items, targets, eligible = inputs(catalog, row)
            s = InterestState.from_history(catalog.x, h)
            projection = s.project()
            scores = {
                method: method_scores(catalog, h, configs[0], method)
                for method in (
                    "content_safe",
                    "old_interest",
                    "no_recency",
                    "popularity",
                    "itemcf",
                )
            }
            scores["no_multi"] = scores["content_safe"]
            scores["no_reliability"] = s.scores(
                projection, configs[0], "no_reliability"
            )
            trials = []
            for cfg in configs:
                score = s.scores(projection, cfg)
                pred = top(score, eligible, 100)
                trials.append(
                    dict(
                        config=asdict(cfg),
                        ids=pred,
                        scores=score[items].tolist(),
                        metrics={
                            **recall_metrics(pred, targets),
                            **ranking_metrics(row["labels"], score[items]),
                        },
                    )
                )
            scores["interest"] = s.scores(projection, configs[0])
            variants = {
                m: candidates(score, eligible, m) for m, score in scores.items()
            }
            band = "1-5" if row["raw_history"] <= 5 else "10"
            if decision:
                src = (
                    "interest"
                    if decision["groups"][band]["accepted"]
                    else "content_safe"
                )
                variants["guarded"] = variants[src]
                scores["guarded"] = scores[src]
            metrics = {
                m: {
                    **recall_metrics(pred, targets),
                    **ranking_metrics(row["labels"], scores[m][items]),
                }
                for m, pred in variants.items()
            }
            for m, metric in metrics.items():
                counts[band, m].append(metric)
            record = dict(
                user=row["user"],
                id=row["id"],
                time=row["time"],
                k=row["raw_history"],
                unique_history=s.n,
                history=h,
                period=row["period"],
                band=band,
                known_large_train=row["known_large_train"],
                first_dev_request=row["first_dev_request"],
                targets=targets,
                reachable=[i for i in targets if eligible[i]],
                new_targets=[i for i in targets if i >= catalog.manifest["old_items"]],
                items=items,
                labels=row["labels"],
                variants=variants,
                scores={m: score[items].tolist() for m, score in scores.items()},
                metrics=metrics,
                trials=trials if split == "select" else None,
                clusters=s.clusters,
                support=(s.mass - s.largest).tolist(),
            )
            stream.write(json.dumps(record) + "\n")
            if (n + 1) % 100 == 0:
                print(
                    dict(
                        split=split,
                        requests=n + 1,
                        seconds=round(time.perf_counter() - start, 2),
                    ),
                    flush=True,
                )
    if identity() != frozen:
        raise ValueError("Algorithm changed during run; preserve rejected predictions")
    write_json(
        dest / "metrics.json",
        dict(
            split=split,
            algorithm=frozen,
            provenance=code_identity(),
            dataset_sha256=sha256(run / "dataset/manifest.json"),
            selection_sha256=sha256(run / "selection.json") if selected else None,
            guard_sha256=sha256(run / "guard.json") if decision else None,
            predictions_sha256=sha256(dest / "predictions.jsonl"),
            seconds=time.perf_counter() - start,
            users=len(rows),
            groups={
                b: {m: average(counts[b, m]) for m in variants} for b in ("1-5", "10")
            },
        ),
    )


def records(run, split):
    dest = run / "reports" / split
    meta = json.loads((dest / "metrics.json").read_text())
    if (
        meta["algorithm"] != identity()
        or meta["predictions_sha256"] != sha256(dest / "predictions.jsonl")
        or meta["dataset_sha256"] != sha256(run / "dataset/manifest.json")
    ):
        raise ValueError("Prediction evidence changed")
    rows = [json.loads(line) for line in (dest / "predictions.jsonl").open()]
    if len({r["user"] for r in rows}) != len(rows):
        raise ValueError("Repeated users")
    return rows


def select(run):
    rows = records(run, "select")
    primary = [r for r in rows if r["band"] == "1-5"]
    means = [
        float(np.mean([r["trials"][i]["metrics"]["recall50"] for r in primary]))
        for i in range(len(GRID))
    ]
    best = min(range(len(GRID)), key=lambda i: (-means[i], GRID[i].alpha, -GRID[i].tau))
    result = dict(
        version=VERSION,
        algorithm=identity(),
        config=asdict(GRID[best]),
        dataset_sha256=sha256(run / "dataset/manifest.json"),
        predictions_sha256=sha256(run / "reports/select/predictions.jsonl"),
        trials=[dict(config=asdict(c), recall50=means[i]) for i, c in enumerate(GRID)],
    )
    write_json(run / "selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


def admission(rows):
    groups = {}
    for band in ("1-5", "10"):
        subset = [r for r in rows if r["band"] == band]
        deltas = [
            r["metrics"]["interest"]["recall50"]
            - r["metrics"]["content_safe"]["recall50"]
            for r in subset
        ]
        test = paired_test(deltas)
        test["periods"] = {
            p: dict(
                n=sum(r["period"] == p for r in subset),
                mean=float(
                    np.mean([d for r, d in zip(subset, deltas) if r["period"] == p])
                )
                if any(r["period"] == p for r in subset)
                else None,
            )
            for p in ("early", "late")
        }
        test["ndcg10_delta"] = (
            float(
                np.mean(
                    [
                        r["metrics"]["interest"]["end_to_end_ndcg10"]
                        - r["metrics"]["content_safe"]["end_to_end_ndcg10"]
                        for r in subset
                    ]
                )
            )
            if subset
            else None
        )
        groups[band] = test
    holm(groups)
    for test in groups.values():
        test["accepted"] = bool(
            test["n"] >= 200
            and test["ci"][0] > 0
            and test["p_holm"] <= 0.05
            and test["ndcg10_delta"] >= 0
            and all(v["n"] >= 100 and v["mean"] >= 0 for v in test["periods"].values())
        )
    return groups


def guard(run):
    result = dict(
        version=VERSION,
        domain="mind-natural-v3",
        algorithm=identity(),
        selection_sha256=sha256(run / "selection.json"),
        predictions_sha256=sha256(run / "reports/guard/predictions.jsonl"),
        groups=admission(records(run, "guard")),
    )
    write_json(run / "guard.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("evaluate", "select", "guard"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--split", choices=("select", "guard", "test"))
    a = p.parse_args()
    if a.action == "evaluate":
        evaluate(a.run, a.split)
    elif a.action == "select":
        select(a.run)
    else:
        guard(a.run)
