"""Reproduce the early-click leak using isolated synthetic feedback only.

Usage: python scripts/review_online.py --output /tmp/unique-run
       [--legacy-source /path/to/old/online_learning.py]
The output directory must not exist. This never writes real traffic or publishes
models to the application's model directory. Results are regression evidence,
not a HN effectiveness experiment.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["SIGNALFEED_HOME"] = str(root)
    os.environ["SIGNALFEED_DB"] = str(root / "fixture.sqlite")

    import numpy as np
    from signalfeed import online_learning as fixed
    from signalfeed import serving
    from signalfeed.store import db, initialize

    initialize()
    vectors = np.eye(384, dtype="float32")[:2]
    features = fixed.online_matrix(vectors, [0, 1], [])
    # 150 independent cold users, balanced positives. No article or prior history
    # can predict the alternating label. A click occurs .1s after recommendation,
    # followed by impressions at 1s. Each request is one day apart.
    with db() as c:
        for i in range(150):
            stamp = 1000000 + i * 86400
            uid, rid = f"u{i}", f"r{i}"
            c.execute(
                "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
                (rid, uid, stamp, "content-v1", "content-v1", '["0","1"]', ""),
            )
            snapshot = dict(
                created=stamp,
                feature_version=fixed.FEATURE_VERSION,
                model_version="content-v1",
                history=[],
                interests=[],
                features={"0": features[0].tolist(), "1": features[1].tolist()},
                content_scores={"0": 0.0, "1": 1.0},
            )
            c.execute(
                "INSERT INTO request_snapshots VALUES(?,?)", (rid, json.dumps(snapshot))
            )
            c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                (f"c{i}", rid, uid, str(i % 2), "click", i % 2, stamp + 0.1, 0),
            )
            for j in range(2):
                c.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                    (f"e{i}-{j}", rid, uid, str(j), "impression", j, stamp + 1, 0),
                )

    if args.legacy_source:
        spec = importlib.util.spec_from_file_location(
            "signalfeed.legacy_online", args.legacy_source
        )
        learning = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(learning)

        # Old retraining reads today's catalog. Freeze it to the exact same vectors
        # so feature differences come only from the delayed-event history bug.
        def refresh(service):
            service.index = {"0": 0, "1": 1}
            service.vectors = vectors.copy()

        serving.ContentService.refresh = refresh
    else:
        learning = fixed

    report = learning.retrain(min_exposures=100, min_clicks=50)
    result = dict(
        scope="synthetic regression only; no HN quality or CTR claim",
        implementation="legacy" if args.legacy_source else "fixed",
        requests=150,
        exposures=300,
        early_clicks=150,
        positive_self_similarity=1.0 if args.legacy_source else float(features[0, -6]),
        report=report,
    )
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
