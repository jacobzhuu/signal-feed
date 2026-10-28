"""Aggregate a single compatible run; paired uncertainty clusters by user."""

import json
import os
from pathlib import Path
import numpy as np
from signalfeed.evaluation import recall_metrics
from signalfeed.retrieval import POLICY_VERSION
from signalfeed.provenance import sha256


def bootstrap(pairs, left, right):
    clusters = {}
    differences = []
    for r in pairs:
        delta = (
            recall_metrics(r["variants"][left], r["positives"])["recall50"]
            - recall_metrics(r["variants"][right], r["positives"])["recall50"]
        )
        clusters.setdefault(r["user"], []).append(delta)
        differences.append(delta)
    if not pairs:
        return {"impressions": 0, "users": 0, "mean": None, "ci95": None}
    sums = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    rng = np.random.default_rng(42)
    values = []
    for _ in range(2000):
        ids = rng.integers(len(sums), size=len(sums))
        values.append(sums[ids].sum() / counts[ids].sum())
    return {
        "metric": "Recall@50 absolute difference",
        "mean": float(np.mean(differences)),
        "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
        "impressions": len(pairs),
        "users": len(clusters),
        "seed": 42,
        "resampling_unit": "user, preserving all sampled requests",
    }


def summarize(root):
    reports = {}
    for seed in [42, 43, 44]:
        file = root / f"reports/test-{seed}/metrics.json"
        if file.exists():
            reports[str(seed)] = json.loads(file.read_text())
    if not reports:
        raise ValueError("No completed evaluations in this run")
    reference = next(iter(reports.values()))
    for seed, report in reports.items():
        if report.get("policy_version") != POLICY_VERSION:
            raise ValueError("Stale policy results cannot be summarized as current")
        for key in ["source_sha256", "data", "test_sha256"]:
            if report["provenance"][key] != reference["provenance"][key]:
                raise ValueError("Mixed code/data runs: " + key)
        if report["fusion_gate"] != reference["fusion_gate"]:
            raise ValueError("Mixed fusion configurations")
    result = {
        "run_id": root.name,
        "provenance": reference["provenance"],
        "policy_version": POLICY_VERSION,
        "completed_seeds": list(reports),
        "missing_seeds": [s for s in ["42", "43", "44"] if s not in reports],
        "ranking": {},
        "retrieval": {},
        "input_sha256": {},
    }
    sample_ids = None
    for seed in reports:
        file = root / f"reports/test-{seed}/retrieval_predictions.jsonl"
        result["input_sha256"][str(file.relative_to(root))] = sha256(file)
        ids = [json.loads(line)["id"] for line in file.open()]
        if sample_ids is not None and ids != sample_ids:
            raise ValueError("Seeds evaluated different requests")
        sample_ids = ids
    for section in ["ranking", "retrieval"]:
        for name in sorted(set().union(*(r[section] for r in reports.values()))):
            metrics = {}
            keys = set().union(*(r[section].get(name, {}) for r in reports.values()))
            for key in keys:
                values = [
                    r[section][name][key]
                    for r in reports.values()
                    if name in r[section] and r[section][name][key] is not None
                ]
                if values:
                    metrics[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "n": len(values),
                    }
            result[section][name] = metrics
    result["cohorts_by_seed"] = {
        s: {
            k: r[k]
            for k in [
                "ranking_cohorts",
                "retrieval_cohorts",
                "positive_availability_rate",
            ]
        }
        for s, r in reports.items()
    }
    pairs = [
        json.loads(line)
        for line in (root / "reports/test-42/retrieval_predictions.jsonl").open()
    ]
    comparisons = [
        ("full_dnn", "no_gen_dnn"),
        ("generative_prefix", "generative"),
        ("adaptive_fusion", "fixed_fusion"),
    ]
    result["paired_bootstrap"] = {
        a + "_minus_" + b: bootstrap(pairs, a, b) for a, b in comparisons
    }
    result["cold_bootstrap"] = {
        c: bootstrap([r for r in pairs if c in r["cohorts"]], "full_dnn", "no_gen_dnn")
        for c in [
            "history_0",
            "history_1_5",
            "history_6_20",
            "unseen_user",
            "new_item",
            "history_0/new_item",
            "history_1_5/new_item",
        ]
    }
    (root / "reports/summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(Path(os.environ["SIGNALFEED_HOME"])), indent=2))
