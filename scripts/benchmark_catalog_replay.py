"""Bounded isolated engineering benchmark; synthetic users are not efficacy evidence."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ.update(
        SIGNALFEED_HOME=str(out / "home"),
        SIGNALFEED_DB=str(out / "isolated.sqlite"),
        SIGNALFEED_TEST_EVENTS="1",
        HF_HOME=str(args.hf_home.resolve()),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    import numpy as np
    from signalfeed.catalog_replay import ReplayStore, ReplayReader, file_digest
    from signalfeed.provenance import code_identity
    from signalfeed.store import db
    from signalfeed.temporal_audit import backup
    import signalfeed.serving as serving

    identity = code_identity()
    backup(args.db, out / "isolated.sqlite")
    # Erase user data only in the private copy; the production source is read-only.
    with db() as c:
        for table in ("request_snapshots", "bookmarks", "events", "requests", "users"):
            c.execute(f"DELETE FROM {table}")
        c.execute("PRAGMA secure_delete=ON")
    with db() as c:
        c.execute("VACUUM")
    backup(out / "isolated.sqlite", out / "catalog-input.sqlite")

    start = time.perf_counter()
    s = serving.ContentService()
    s.refresh()
    initialization_ms = (time.perf_counter() - start) * 1000
    if len(s.items) < 100 or s.refresh_error:
        raise ValueError("Benchmark requires at least 100 valid catalog items")
    # Fix the serving timestamp to the source catalog's latest update for reproducibility.
    now = max(a["updated"] for a in s.items) + 60.0
    s.refresh_at = now - 1
    s.refresh = lambda *a, **k: None
    clock = SimpleNamespace(time=lambda: now)
    serving.time = clock

    original = subprocess.check_output(
        ["git", "show", "cc144ad:src/signalfeed/serving.py"]
    )
    (out / "baseline-serving.py").write_bytes(original)
    baseline_module = ModuleType("signalfeed._replay_baseline")
    baseline_module.__package__ = "signalfeed"
    exec(
        compile(original, str(out / "baseline-serving.py"), "exec"),
        baseline_module.__dict__,
    )
    baseline_module.time = clock
    old = baseline_module.ContentService()
    old.items, old.vectors, old.index = s.items, s.vectors, s.index
    old.model, old.refresh_at = s.model, s.refresh_at
    old.refresh = lambda *a, **k: None

    samples = []
    with db() as c:
        for n in (0, 1, 3, 5, 10, 25):
            for j in range(20):
                user = f"synthetic-{n}-{j}"
                interests = (
                    ["machine learning", "software engineering"] if j % 2 else []
                )
                c.execute(
                    "INSERT INTO users VALUES(?,?)", (user, json.dumps(interests))
                )
                ids = [s.items[(j * 13 + k * 7) % len(s.items)]["id"] for k in range(n)]
                for k, iid in enumerate(ids):
                    event = f"{user}-{k}"
                    c.execute(
                        "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                        (event, event, user, iid, "click", 0, now - 100 + k, 1),
                    )
                # Synthetic CF peer events are flag 0 ONLY inside this isolated database.
                if ids:
                    for k, iid in enumerate(
                        (
                            ids[0],
                            s.items[(j + 60) % len(s.items)]["id"],
                            "outside-catalog",
                        )
                    ):
                        event = f"peer-{user}-{k}"
                        c.execute(
                            "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                            (
                                event,
                                event,
                                "peer-" + user,
                                iid,
                                "click",
                                0,
                                now - 100,
                                0,
                            ),
                        )
                samples.append(dict(user=user, history=n, explicit=bool(interests)))

    store = ReplayStore(out / "journal")
    start = time.perf_counter()
    initial_version = store.register(s.items, s.vectors)
    registration_ms = (time.perf_counter() - start) * 1000
    reader = ReplayReader(store.root)
    # Encode the same fixed explicit input once, as normal serving's query cache does.
    for service in (old, s):
        service.encode_query("machine learning software engineering")
    s.recommend(samples[0]["user"], replay=store)
    reader.replay(store.last_request)
    raw = []
    for i, sample in enumerate(samples):
        user = sample["user"]
        reference_snapshot = {}
        reference = old.recommend(user, limit=100, snapshot=reference_snapshot)
        reference_ids = [a["id"] for a in reference]
        # Prepared request allows replay to precede timed capture in rotated orders.
        s.recommend(user, limit=30, replay=store)
        prepared = store.last_request
        methods = ["plain", "recorded", "replay"]
        order = methods[i % 3 :] + methods[: i % 3]
        row = {**sample, "order": order, "prepared_request": prepared, "ms": {}}
        for method in order:
            start = time.perf_counter()
            if method == "replay":
                result = reader.replay(prepared)
            else:
                result = s.recommend(
                    user, limit=30, replay=store if method == "recorded" else None
                )
            row["ms"][method] = (time.perf_counter() - start) * 1000
            if method == "replay":
                assert result["top100"] == reference_ids
                assert [
                    float(result["scores"][s.index[iid]]) for iid in reference_ids
                ] == [
                    reference_snapshot["content_scores"][iid] for iid in reference_ids
                ]
            else:
                assert [a["id"] for a in result] == reference_ids[:30]
                if method == "recorded":
                    row["measured_request"] = store.last_request
        measured = reader.replay(row["measured_request"])
        assert measured["top100"] == reference_ids
        row["request_bytes"] = (
            (store.root / "requests" / f"{row['measured_request']}.json").stat().st_size
        )
        row["exact_baseline_match"] = True
        raw.append(row)
        with (out / "timings.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    start = time.perf_counter()
    ReplayReader(store.root).replay(raw[0]["measured_request"])
    cold_read_ms = (time.perf_counter() - start) * 1000
    unchanged = {
        r["measured_request"]: file_digest(
            store.root / "requests" / f"{r['measured_request']}.json"
        )
        for r in raw
    }
    # A later catalog removes a row, changes text/popularity/vectors and adds an item.
    old_items = s.items
    s.items = [dict(a) for a in old_items[:-1]]
    s.items[0].update(title="Synthetic future title", score=12345)
    s.items.append(
        dict(old_items[-1], id="synthetic-new-article", title="Future catalog addition")
    )
    s.vectors = s.vectors.copy()
    s.vectors[[0, -1]] = s.vectors[[1, 2]]
    s.vectors.setflags(write=False)
    s.index = {a["id"]: i for i, a in enumerate(s.items)}
    now += 86400
    s.refresh_at = now
    with db() as c:
        c.execute("UPDATE users SET interests=?", ('["future preference"]',))
        for i, sample in enumerate(samples):
            e = "future-" + str(i)
            c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                (e, e, sample["user"], s.items[-1]["id"], "click", 0, now, 1),
            )
    s.recommend(samples[0]["user"], replay=store)
    future_version = store._catalog.version
    assert future_version != initial_version
    fresh_reader = ReplayReader(store.root)
    for r in raw:
        request = r["measured_request"]
        assert fresh_reader.replay(request)["catalog_version"] == initial_version
        assert unchanged[request] == file_digest(
            store.root / "requests" / f"{request}.json"
        )

    def quantiles(values):
        return dict(
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
            maximum=float(max(values)),
        )

    summaries = {}
    for key in ("all", 0, 1, 3, 5, 10, 25):
        rows = raw if key == "all" else [r for r in raw if r["history"] == key]
        summaries[str(key)] = {
            m: quantiles([r["ms"][m] for r in rows]) for m in methods
        }
        summaries[str(key)]["capture_overhead"] = quantiles(
            [r["ms"]["recorded"] - r["ms"]["plain"] for r in rows]
        )
    overhead = summaries["all"]["capture_overhead"]["p95"]
    small_sizes = [r["request_bytes"] for r in raw if r["history"] <= 20]
    catalog_sizes = {
        p.name: sum(f.stat().st_size for f in p.iterdir())
        for p in (store.root / "catalogs").iterdir()
    }
    report = dict(
        provenance=identity,
        synthetic_only=True,
        efficacy_authorized=False,
        deployed=False,
        baseline_commit="cc144ad",
        baseline_sha256=file_digest(out / "baseline-serving.py"),
        input_db_sha256=file_digest(out / "catalog-input.sqlite"),
        catalog_items=len(old_items),
        requests=len(raw),
        catalog_versions=catalog_sizes,
        initial_catalog_version=initial_version,
        future_catalog_version=future_version,
        initialization_ms=initialization_ms,
        catalog_registration_ms=registration_ms,
        cold_reader_ms=cold_read_ms,
        latency_ms=summaries,
        request_bytes=quantiles([r["request_bytes"] for r in raw]),
        small_history_request_bytes=quantiles(small_sizes),
        catalog_reuse=dict(
            measured_requests=120,
            initial_versions=1,
            bytes_without_sharing=120 * catalog_sizes[initial_version],
            bytes_with_sharing=catalog_sizes[initial_version],
        ),
        gates=dict(
            capture_overhead_p95_limit_ms=10,
            capture_overhead_pass=overhead <= 10,
            small_request_limit_bytes=32768,
            small_request_pass=max(small_sizes) <= 32768,
        ),
        exact_baseline_matches=120,
        preserved_after_future_changes=120,
        environment=dict(
            python=sys.version,
            numpy=np.__version__,
            platform=platform.platform(),
            cpu=next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
            torch_threads=2,
            blas_threads=1,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        ),
        limitations=[
            "Shared host; 120 sequential synthetic requests, no concurrency capacity claim",
            "Warm encoder/explicit-query cache; no recommendation-result cache",
            "Measured service includes SQLite reads and durable journal fsync, excludes HTTP and API snapshot writes",
            "Cold reader means a new reader object, OS page cache not evicted",
            "Exact scores are tied to source and numerical environment; not cross-hardware bitwise guarantee",
        ],
    )
    if identity != code_identity():
        raise ValueError("Source changed during measurement")
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
