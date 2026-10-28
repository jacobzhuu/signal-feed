"""Verify raw predictions and report every group/method, including null effects."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from signalfeed.provenance import sha256
from signalfeed.evaluation import ranking_metrics


def interval(rows):
    clusters = defaultdict(list)
    for user, delta in rows:
        clusters[user].append(delta)
    sums = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    rng = np.random.default_rng(42)
    draws = rng.integers(len(sums), size=(1000, len(sums)))
    values = sums[draws].sum(1) / counts[draws].sum(1)
    return dict(
        delta=float(sums.sum() / counts.sum()),
        ci95=np.percentile(values, [2.5, 97.5]).tolist(),
        users=len(sums),
        requests=int(counts.sum()),
    )


def summarize(root):
    reports, differences = [], defaultdict(lambda: defaultdict(list))
    identities, sample_ids = {}, {}
    max_error = 0.0
    for natural in (False, True):
        for seed in (42, 43, 44):
            directory = (
                root / "reports" / f"cold-{'natural' if natural else 'test'}-{seed}"
            )
            report = json.loads((directory / "metrics.json").read_text())
            pred = directory / "predictions.jsonl"
            if sha256(pred) != report["predictions_sha256"]:
                raise ValueError("Changed raw predictions")
            identity = report["provenance"]
            for key in ("source_sha256", "data", "test_sha256"):
                if key in identities and identities[key] != identity[key]:
                    raise ValueError("Mixed experiment sources")
                identities[key] = identity[key]
            for key in ("config", "selection_sha256", "baseline_gate_sha256"):
                if key in identities and identities[key] != report[key]:
                    raise ValueError("Mixed fusion configurations")
                identities[key] = report[key]
            if (
                sha256(root / f"models/coldfuse-{seed}/model.pt")
                != report["cold_model_sha256"]
            ):
                raise ValueError("Changed evaluated checkpoint")
            if natural in sample_ids and sample_ids[natural] != report["ids"]:
                raise ValueError("Seed samples differ")
            sample_ids[natural] = report["ids"]
            reports.append(report)
            sums, n = defaultdict(lambda: defaultdict(list)), 0
            with pred.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    n += 1
                    if set(r["observed"]) & set(r["proxy_source"]):
                        raise ValueError("Proxy and observations overlap")
                    pos = set(r["targets"])
                    for method, ids in r["variants"].items():
                        if len(ids) != len(set(ids)) or set(ids) & set(r["observed"]):
                            raise ValueError(
                                "Duplicate or already observed recommendations"
                            )
                        recomputed = ranking_metrics(r["labels"], r["scores"][method])
                        for k in (20, 50):
                            recomputed[f"recall{k}"] = (
                                sum(i in pos for i in ids[:k]) / len(pos) if pos else 0
                            )
                        for k in (5, 10):
                            dcg = sum(
                                (i in pos) / np.log2(j + 2)
                                for j, i in enumerate(ids[:k])
                            )
                            ideal = sum(
                                1 / np.log2(j + 2) for j in range(min(k, len(pos)))
                            )
                            recomputed[f"end_to_end_ndcg{k}"] = (
                                dcg / ideal if ideal else 0
                            )
                        for metric, value in recomputed.items():
                            if value is not None:
                                max_error = max(
                                    max_error, abs(value - r["metrics"][method][metric])
                                )
                                sums[r["group"], method][metric].append(value)
                        if method != "coldfuse":
                            for metric in (
                                "recall50",
                                "end_to_end_ndcg10",
                                "auc",
                                "ndcg10",
                            ):
                                a, b = (
                                    r["metrics"]["coldfuse"][metric],
                                    r["metrics"][method][metric],
                                )
                                if a is not None and b is not None:
                                    differences[r["group"], method, metric][
                                        r["id"], r["user"]
                                    ].append(a - b)
            expected = report["requests"] * (1 if natural else 10)
            if n != expected:
                raise ValueError("Incomplete prediction file")
            for (group, method), metrics in sums.items():
                for name, values in metrics.items():
                    max_error = max(
                        max_error,
                        abs(
                            float(np.mean(values))
                            - report["groups"][group][method][name]
                        ),
                    )
    if max_error > 1e-10:
        raise ValueError(f"Metric mismatch {max_error}")
    groups = defaultdict(lambda: defaultdict(dict))
    combined = defaultdict(list)
    for report in reports:
        for group, methods in report["groups"].items():
            for method, metrics in methods.items():
                for metric, value in metrics.items():
                    if value is not None:
                        combined[group, method, metric].append(value)
    for (group, method, metric), values in combined.items():
        groups[group][method][metric] = dict(
            mean=float(np.mean(values)), std=float(np.std(values)), seeds=len(values)
        )
    cis = defaultdict(lambda: defaultdict(dict))
    for (group, method, metric), pairs in differences.items():
        if any(len(v) != 3 for v in pairs.values()):
            raise ValueError("Missing paired seed predictions")
        cis[group][method][metric] = interval(
            [(user, float(np.mean(v))) for (_, user), v in pairs.items()]
        )
    result = dict(
        run_id=root.name,
        identity=identities,
        groups=groups,
        comparisons=cis,
        config=reports[0]["config"],
        maximum_recomputed_error=max_error,
        costs=[
            dict(seed=r["seed"], natural=r["natural"], groups=r["costs"])
            for r in reports
        ],
        uncertainty="Average seed deltas within each impression, then bootstrap users; exploratory unadjusted 95% intervals; no causal CTR inference",
        runs=[
            dict(
                seed=r["seed"],
                natural=r["natural"],
                provenance=r["provenance"],
                cold_model_sha256=r["cold_model_sha256"],
                predictions_sha256=r["predictions_sha256"],
            )
            for r in reports
        ],
    )
    (root / "reports/cold_summary.json").write_text(json.dumps(result, indent=2))
    with (root / "reports/cold_metrics.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "method", "metric", "mean", "std", "seeds"])
        for group, methods in sorted(groups.items()):
            for method, metrics in sorted(methods.items()):
                for metric, v in sorted(metrics.items()):
                    writer.writerow(
                        [group, method, metric, v["mean"], v["std"], v["seeds"]]
                    )
    print(
        json.dumps(
            dict(run=root.name, maximum_recomputed_error=max_error, groups=len(groups))
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    summarize(parser.parse_args().run)
