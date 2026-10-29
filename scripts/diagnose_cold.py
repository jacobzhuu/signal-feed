"""Read-only diagnosis of the already exposed baseline; never selects a model."""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from signalfeed.provenance import sha256


def diagnose(root, output):
    groups = defaultdict(lambda: defaultdict(list))
    sources = {}
    names = ("content", "itemcf", "sid_prefix")
    for seed in (42, 43, 44):
        for cohort in ("test", "natural"):
            file = root / f"reports/cold-{cohort}-{seed}/predictions.jsonl"
            sources[str(file)] = sha256(file)
            with file.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    v, p = r["variants"], set(r["targets"])
                    n = max(len(p), 1)
                    g = groups[r["group"]]
                    for name, pred in v.items():
                        g[name + "/recall50"].append(len(set(pred[:50]) & p) / n)
                    channels = {name: set(v[name]) for name in names}
                    union = set().union(*channels.values())
                    g["union100_recall"].append(len(union & p) / n)
                    full_union = union | set(v["popularity"])
                    fused = set(v["coldfuse"])
                    g["union_with_pop100_recall"].append(len(full_union & p) / n)
                    g["fusion100_recall"].append(len(fused & p) / n)
                    g["lost_at_fusion_recall"].append(len((full_union - fused) & p) / n)
                    g["lost_at_ranking50_recall"].append(
                        len((fused - set(v["coldfuse"][:50])) & p) / n
                    )
                    for a, b in itertools.combinations(names, 2):
                        g[f"jaccard/{a}/{b}"].append(
                            len(channels[a] & channels[b])
                            / max(1, len(channels[a] | channels[b]))
                        )
                    for a in names:
                        others = set().union(
                            *(s for b, s in channels.items() if b != a)
                        )
                        g[f"exclusive_recall/{a}"].append(
                            len((channels[a] - others) & p) / n
                        )
                        g[f"size/{a}"].append(len(channels[a]))
                        g[f"survival100/{a}"].append(
                            len(channels[a] & fused) / max(1, len(channels[a]))
                        )
                    for name, weight in r["gates"].items():
                        g[f"gate/{name}"].append(weight)
                    g["beam_concentration"].append(r["beam_concentration"])
    report = dict(
        scope="Exposed baseline diagnosis, not independent confirmation or model selection",
        sources=sources,
        groups={
            g: {k: float(np.mean(v)) for k, v in metrics.items()}
            for g, metrics in groups.items()
        },
    )
    output.write_text(json.dumps(report, indent=2))
    for group, metrics in report["groups"].items():
        print(
            group,
            {
                k: round(metrics[k], 5)
                for k in (
                    "content/recall50",
                    "coldfuse/recall50",
                    "union100_recall",
                    "lost_at_fusion_recall",
                    "lost_at_ranking50_recall",
                    "gate/popular",
                    "gate/content",
                )
            },
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    diagnose(args.baseline, args.output)
