"""Publish measured results only. Missing runs remain explicitly absent."""

import json
from pathlib import Path
import numpy as np
import os

root = Path(os.environ.get("SIGNALFEED_HOME", "/data1/zhuzy/signalfeed"))
reports = {}
for seed in [42, 43, 44]:
    p = root / f"reports/test-{seed}/metrics.json"
    if p.exists():
        reports[str(seed)] = json.loads(p.read_text())
summary = {
    "completed_seeds": list(reports),
    "missing_seeds": [s for s in ["42", "43", "44"] if s not in reports],
    "ranking": {},
    "retrieval": {},
}
for section in ["ranking", "retrieval"]:
    names = (
        set().union(*(r[section].keys() for r in reports.values()))
        if reports
        else set()
    )
    for name in sorted(names):
        metrics = {}
        keys = set().union(*(r[section].get(name, {}).keys() for r in reports.values()))
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
        summary[section][name] = metrics
(root / "reports/summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

# Paired bootstrap across the SAME 1,000 impressions (seed 42), never treat
# repeated model seeds as independent users. This is uncertainty, not online lift.
from signalfeed.evaluation import recall_metrics

prediction_file = root / "reports/test-42/retrieval_predictions.jsonl"
if prediction_file.exists() and "42" in reports:
    pairs = [json.loads(line) for line in prediction_file.open()]
    comparisons = {}
    rng = np.random.default_rng(42)
    for left, right in [
        ("full_dnn", "no_gen_dnn"),
        ("generative_prefix", "generative"),
        ("adaptive_fusion", "fixed_fusion"),
    ]:
        differences = np.array(
            [
                recall_metrics(r["variants"][left], r["positives"])["recall50"]
                - recall_metrics(r["variants"][right], r["positives"])["recall50"]
                for r in pairs
            ]
        )
        boot = np.array(
            [
                rng.choice(differences, len(differences), replace=True).mean()
                for _ in range(2000)
            ]
        )
        comparisons[f"{left}_minus_{right}"] = {
            "metric": "Recall@50 absolute difference",
            "mean": float(differences.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "impressions": len(pairs),
            "seed": 42,
        }
    summary["paired_bootstrap"] = comparisons
    (root / "reports/summary.json").write_text(json.dumps(summary, indent=2))
