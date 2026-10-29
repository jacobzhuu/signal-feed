"""Use already-consumed predictions only to diagnose coverage, never select a method."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import norm


def checksum(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def diagnose(run, output):
    output.mkdir(parents=True, exist_ok=False)
    file = run / "reports/test/predictions.jsonl"
    meta = json.loads(file.with_name("metrics.json").read_text())
    if checksum(file) != meta["predictions_sha256"]:
        raise ValueError("Old predictions changed")
    groups = defaultdict(list)
    for line in file.open():
        row = json.loads(line)
        groups[row["band"]].append(row)
    result = {}
    for band, rows in groups.items():
        methods = {}
        targets = sum(len(set(r["targets"])) for r in rows)
        reachable = sum(len(set(r["reachable"])) for r in rows)
        for method in (
            "content_safe",
            "interest",
            "old_interest",
            "no_reliability",
            "no_recency",
        ):
            hits = {
                str(k): sum(
                    len(set(r["variants"][method][:k]) & set(r["targets"]))
                    for r in rows
                )
                for k in (10, 20, 50, 100)
            }
            user_delta = np.array(
                [
                    r["metrics"][method]["recall50"]
                    - r["metrics"]["content_safe"]["recall50"]
                    for r in rows
                ]
            )
            methods[method] = dict(
                target_hits=hits,
                hit_users50=sum(r["metrics"][method]["recall50"] > 0 for r in rows),
                reachable_targets_outside_top100=reachable - hits["100"],
                targets_in_100_but_outside_50=hits["100"] - hits["50"],
                targets_in_50_but_outside_10=hits["50"] - hits["10"],
                changed_lists=sum(
                    r["variants"][method] != r["variants"]["content_safe"] for r in rows
                ),
                nonzero_paired_recall_users=int(np.count_nonzero(user_delta)),
                paired_recall_variance=float(np.var(user_delta, ddof=1)),
                mean_exposure_auc=float(
                    np.mean(
                        [
                            r["metrics"][method]["auc"]
                            for r in rows
                            if r["metrics"][method]["auc"] is not None
                        ]
                    )
                ),
            )
        result[band] = dict(
            users=len(rows),
            positive_targets=targets,
            reachable_targets=reachable,
            unavailable_targets=targets - reachable,
            methods=methods,
        )
    # Planning sensitivity only; no threshold or sample-size selection from test.
    alpha = 0.05 / 2
    constant = float((norm.ppf(1 - alpha / 2) + norm.ppf(0.8)) ** 2)
    sensitivity = [
        dict(
            assumed_paired_variance=v,
            minimum_effect=delta,
            approximate_users_per_band_per_stage=math.ceil(constant * v / delta**2),
        )
        for v in (0.005, 0.01, 0.02)
        for delta in (0.001, 0.002, 0.005)
    ]
    report = dict(
        status="diagnosis_of_previously_used_data_not_new_evidence",
        source_sha256=checksum(file),
        source_meta_sha256=checksum(file.with_name("metrics.json")),
        script_sha256=checksum(Path(__file__)),
        groups=result,
        power_sensitivity=dict(
            formula="n ~= (z_(1-alpha/2)+z_0.8)^2 * Var(paired delta) / minimum_effect^2",
            alpha_per_comparison=alpha,
            comparisons=2,
            power=0.8,
            rows=sensitivity,
            limitations="Normal approximation; variance assumptions, not estimates for HN. Needs sparse-outcome simulation and date clustering before an efficacy protocol. Zero observed v3 variance is not zero future variance.",
        ),
    )
    with (output / "diagnosis.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    diagnose(a.run, a.output)
