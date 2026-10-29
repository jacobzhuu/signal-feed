"""Archive previous evidence hashes before creating a fresh interest run."""

import argparse
import json
from pathlib import Path
import subprocess
from signalfeed.provenance import sha256


def freeze(run):
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    root = Path(".")
    paths = list(root.glob("docs/*ADAPTATION*")) + list(root.glob("docs/adaptation*"))
    for name in ("adaptation-v1", "adaptation-v2"):
        parent = root / "artifacts" / name
        paths += [
            f
            for directory in ("dataset", "reports", "logs", "engineering-completed")
            for f in (parent / directory).rglob("*")
            if f.is_file()
        ]
        paths += [f for f in parent.glob("*.json") if f.is_file()]
    with (run / "frozen_previous.json").open("x") as stream:
        json.dump(
            dict(
                commit="3426a00", files={str(p): sha256(p) for p in sorted(set(paths))}
            ),
            stream,
            indent=2,
        )


def freeze_baseline(run):
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "3426a00", "docs"], text=True
    ).splitlines()
    paths = [Path(f) for f in files]
    for name in ("audit-v2", "review-fix", "coldfuse-v1", "coldfuse-v2", "reports"):
        parent = Path("artifacts") / name
        directory = parent if name == "reports" else parent / "reports"
        paths += [p for p in directory.rglob("*") if p.is_file()]
    with (run / "frozen_baseline_results.json").open("x") as stream:
        json.dump(
            dict(
                commit="3426a00", files={str(p): sha256(p) for p in sorted(set(paths))}
            ),
            stream,
            indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline-only", action="store_true")
    a = p.parse_args()
    if not a.baseline_only:
        freeze(a.run)
    freeze_baseline(a.run)
