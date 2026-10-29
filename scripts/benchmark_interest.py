"""Independent uncached requests plus update-only cost; bind every measured list."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from signalfeed.interest_experiments import (
    METHODS,
    load_catalog,
    read_rows,
    inputs,
    records,
    method_scores,
    candidates,
    identity,
    write_json,
)
from signalfeed.stream_interest import InterestConfig, InterestState
from signalfeed.provenance import sha256, code_identity


def summary(values):
    return dict(
        n=len(values),
        p50_ms=float(np.percentile(values, 50) * 1000),
        p95_ms=float(np.percentile(values, 95) * 1000),
    )


def benchmark(run):
    threadpool_limits(limits=1, user_api="blas")
    start = time.perf_counter()
    frozen = identity()
    catalog = load_catalog(run)
    initialization = time.perf_counter() - start
    cfg = InterestConfig(**json.loads((run / "selection.json").read_text())["config"])
    expected = {r["user"]: r for r in records(run, "test")}
    counts, selected = defaultdict(int), []
    for row in read_rows(run, "test"):
        key = row["raw_history"], row["period"]
        if counts[key] < 20:
            selected.append(row)
            counts[key] += 1
    out = run / "cost"
    out.mkdir(exist_ok=False)
    h, _, _, eligible = inputs(catalog, selected[0])
    for method in METHODS:
        candidates(method_scores(catalog, h, cfg, method), eligible, method)
    values, updates, batch = defaultdict(list), defaultdict(list), defaultdict(list)
    state_bytes = 0
    with (out / "calls.jsonl").open("x") as stream:
        for i, row in enumerate(selected):
            h, _, _, eligible = inputs(catalog, row)
            order = METHODS[i % len(METHODS) :] + METHODS[: i % len(METHODS)]
            for method in order:
                tick = time.perf_counter()
                pred = candidates(
                    method_scores(catalog, h, cfg, method), eligible, method
                )
                seconds = time.perf_counter() - tick
                if pred != expected[row["user"]]["variants"][method]:
                    raise ValueError(
                        f"Independent prediction mismatch: {method}/{row['user']}"
                    )
                values[method, str(row["raw_history"])].append(seconds)
                values[method, "all"].append(seconds)
                stream.write(
                    json.dumps(
                        dict(
                            user=row["user"],
                            k=row["raw_history"],
                            method=method,
                            seconds=seconds,
                            ids=pred,
                        )
                    )
                    + "\n"
                )
            current = InterestState.from_history(catalog.x, h[:-1])
            tick = time.perf_counter()
            rebuilt = current.update(h[-1])
            update_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            rebuilt_state = InterestState.from_history(catalog.x, h)
            batch_seconds = time.perf_counter() - tick
            for name in ("total", "position", "sums", "mass", "largest"):
                np.testing.assert_allclose(
                    getattr(current, name),
                    getattr(rebuilt_state, name),
                    rtol=1e-12,
                    atol=1e-12,
                )
            for j in range(1, len(h) + 1):
                # Validate the entire real prefix trajectory, not only its last event.
                online = InterestState.from_history(catalog.x, h[: j - 1])
                online.update(h[j - 1])
                reference = InterestState.from_history(catalog.x, h[:j])
                np.testing.assert_allclose(
                    online.sums, reference.sums, rtol=1e-12, atol=1e-12
                )
            incremental_pred = candidates(
                current.scores(current.project(), cfg), eligible, "interest"
            )
            if incremental_pred != expected[row["user"]]["variants"]["interest"]:
                raise ValueError("Incremental candidate mismatch")
            updates[str(row["raw_history"])].append(update_seconds)
            batch[str(row["raw_history"])].append(batch_seconds)
            state_bytes = max(
                state_bytes,
                sum(
                    getattr(current, name).nbytes
                    for name in ("total", "position", "sums", "mass", "largest")
                ),
            )
            stream.write(
                json.dumps(
                    dict(
                        user=row["user"],
                        k=row["raw_history"],
                        method="incremental_update",
                        seconds=update_seconds,
                        batch_seconds=batch_seconds,
                        rebuilt=rebuilt,
                    )
                )
                + "\n"
            )
    if frozen != identity():
        raise ValueError("Algorithm changed during benchmark")
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    report = dict(
        algorithm=frozen,
        provenance=code_identity(),
        config=asdict(cfg),
        benchmark_sha256=sha256(Path(__file__)),
        predictions_sha256=sha256(run / "reports/test/predictions.jsonl"),
        selection_sha256=sha256(run / "selection.json"),
        dataset_sha256=sha256(run / "dataset/manifest.json"),
        raw_sha256=sha256(out / "calls.jsonl"),
        users=len(selected),
        calls=len(selected) * len(METHODS),
        all_predictions_exact=True,
        initialization_seconds=initialization,
        total_seconds=time.perf_counter() - start,
        cpu=cpu,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        load_average=os.getloadavg(),
        threadpools=threadpool_info(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        catalog_shape=list(catalog.x.shape),
        catalog_bytes=catalog.x.nbytes,
        state_numeric_bytes=state_bytes,
        methods={
            m: {k: summary(v) for (mm, k), v in values.items() if mm == m}
            for m in METHODS
        },
        incremental={
            k: dict(update=summary(v), batch=summary(batch[k]))
            for k, v in updates.items()
        },
        notes=[
            "Preencoded resident catalog, no result cache, one warmup per method, rotating order",
            "Does not include HTTP, database, text encoding or cold model startup",
            "Peak RSS includes all baselines, CF and held quality predictions; shared host",
            "State bytes exclude Python objects and shared catalog; update does not retrieve",
        ],
    )
    write_json(out / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    benchmark(p.parse_args().run)
