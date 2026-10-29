"""One frozen four-way storage comparison on v5 synthetic requests and catalog."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def frozen_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--previous", type=Path, default=Path("artifacts/replay-v5/benchmark")
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ.update(
        SIGNALFEED_HOME=str(out / "home"),
        SIGNALFEED_DB=str(out / "isolated.sqlite"),
        SIGNALFEED_TEST_EVENTS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    import numpy as np
    import sqlite3
    from signalfeed.catalog_replay import (
        ReplayStore,
        ReplayReader,
        SQLiteReplayStore,
        SQLiteReplayReader,
        load_catalog,
        file_digest,
    )
    from signalfeed.provenance import code_identity
    from signalfeed.store import db
    from signalfeed.temporal_audit import backup
    import signalfeed.serving as serving

    identity = code_identity()
    previous = args.previous.resolve()
    old_results = json.loads((previous / "results.json").read_text())
    samples = [
        json.loads(line)
        for line in (previous / "timings.jsonl").read_text().splitlines()
    ]
    assert len(samples) == 120
    records = [
        json.loads(
            (
                previous / "journal/requests" / f"{s['measured_request']}.json"
            ).read_text()
        )["record"]
        for s in samples
    ]
    catalog = load_catalog(previous / "journal", old_results["initial_catalog_version"])
    # Reconstitute exactly the old synthetic inputs; no production DB is opened.
    backup(previous / "isolated.sqlite", out / "isolated.sqlite")
    now = records[0]["created"]
    with db() as connection:
        connection.execute("DELETE FROM events WHERE created>?", (now,))
        for sample, record in zip(samples, records):
            connection.execute(
                "UPDATE users SET interests=? WHERE id=?",
                (json.dumps(record["interests"]), sample["user"]),
            )
    explicit = {
        " ".join(r["interests"]): np.array(r["explicit_vector"], dtype="float32")
        for r in records
    }
    s = serving.ContentService()
    s.items, s.vectors = catalog.rows, catalog.vectors
    s.index = {a["id"]: i for i, a in enumerate(s.items)}
    s.refresh_at = records[0]["catalog_refreshed_at"]
    s.refresh = lambda *a, **kw: None
    s.encode_query = lambda text: explicit[text].copy()
    clock = SimpleNamespace(time=lambda: now)
    serving.time = clock

    frozen = out / "frozen-v5"
    frozen.mkdir()
    for name in ("serving.py", "catalog_replay.py"):
        (frozen / name).write_bytes(
            subprocess.check_output(["git", "show", f"c1d6296:src/signalfeed/{name}"])
        )
    legacy = frozen_module("signalfeed._frozen_v5_replay", frozen / "catalog_replay.py")
    old_serving = frozen_module("signalfeed._frozen_v5_serving", frozen / "serving.py")
    old_serving.time = clock
    reference = old_serving.ContentService()
    reference.items, reference.vectors, reference.index = s.items, s.vectors, s.index
    reference.refresh_at, reference.refresh, reference.encode_query = (
        s.refresh_at,
        s.refresh,
        s.encode_query,
    )
    stores = dict(
        file_v5=legacy.ReplayStore(out / "file-v5"),
        file_batched=ReplayStore(out / "file-batched"),
        sqlite=SQLiteReplayStore(out / "sqlite"),
    )
    readers = dict(
        file_v5=legacy.ReplayReader(stores["file_v5"].root),
        file_batched=ReplayReader(stores["file_batched"].root),
        sqlite=SQLiteReplayReader(stores["sqlite"].root),
    )
    registration = {}
    for name, store in stores.items():
        start = time.perf_counter()
        store.register(s.items, s.vectors)
        registration[name] = (time.perf_counter() - start) * 1000

    def storage():
        return {
            p.name: p.stat().st_size
            for p in stores["sqlite"].root.glob("requests.sqlite*")
        }

    def allocated_storage():
        return {
            p.name: p.stat().st_blocks * 512
            for p in stores["sqlite"].root.glob("requests.sqlite*")
        }

    initial_storage = storage()
    initial_allocated = allocated_storage()
    # One unmeasured warm request per recorder, all later results retained.
    for name, store in stores.items():
        s.recommend(samples[0]["user"], replay=store)
        readers[name].replay(store.last_request)
    methods = ["plain", "file_v5", "file_batched", "sqlite"]
    raw = []
    for i, sample in enumerate(samples):
        snapshot = {}
        baseline = reference.recommend(sample["user"], limit=100, snapshot=snapshot)
        baseline_ids = [a["id"] for a in baseline]
        order = methods[i % 4 :] + methods[: i % 4]
        row = {
            "user": sample["user"],
            "history": sample["history"],
            "order": order,
            "ms": {},
            "receipts": {},
            "request_bytes": {},
            "physical_growth_bytes": {},
            "allocated_growth_bytes": {},
        }
        for name in order:
            before = storage() if name == "sqlite" else {}
            before_allocated = allocated_storage() if name == "sqlite" else {}
            start = time.perf_counter()
            result = s.recommend(sample["user"], replay=stores.get(name))
            row["ms"][name] = (time.perf_counter() - start) * 1000
            after = storage() if name == "sqlite" else {}
            after_allocated = allocated_storage() if name == "sqlite" else {}
            assert [a["id"] for a in result] == baseline_ids[:30]
            if name != "plain":
                request = stores[name].last_request
                row["receipts"][name] = request
                replay = readers[name].replay(request)
                assert replay["top100"] == baseline_ids
                assert [
                    float(replay["scores"][s.index[iid]]) for iid in baseline_ids
                ] == [snapshot["content_scores"][iid] for iid in baseline_ids]
                if name == "sqlite":
                    size = (
                        stores[name]
                        .connection.execute(
                            "SELECT length(envelope) FROM records WHERE id=?",
                            (request,),
                        )
                        .fetchone()[0]
                    )
                    row["physical_growth_bytes"][name] = sum(after.values()) - sum(
                        before.values()
                    )
                    row["allocated_growth_bytes"][name] = sum(
                        after_allocated.values()
                    ) - sum(before_allocated.values())
                else:
                    size = (
                        (stores[name].root / "requests" / f"{request}.json")
                        .stat()
                        .st_size
                    )
                    row["physical_growth_bytes"][name] = size
                row["request_bytes"][name] = size
        raw.append(row)
        with (out / "timings.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    # Warm replay cost separate from capture, never substituted for write latency.
    replay_ms = []
    for row in raw:
        start = time.perf_counter()
        readers["sqlite"].replay(row["receipts"]["sqlite"])
        replay_ms.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    SQLiteReplayReader(stores["sqlite"].root).replay(raw[0]["receipts"]["sqlite"])
    cold_reader_ms = (time.perf_counter() - start) * 1000
    before_checkpoint = storage()
    before_checkpoint_allocated = allocated_storage()
    start = time.perf_counter()
    checkpoint = list(
        stores["sqlite"]
        .connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        .fetchone()
    )
    checkpoint_ms = (time.perf_counter() - start) * 1000
    after_checkpoint = storage()
    after_checkpoint_allocated = allocated_storage()
    assert checkpoint[0] == 0
    # Future mutation is outside timing. Recheck every past record with new readers.
    s.items = [dict(a) for a in s.items[:-1]]
    s.items[0].update(title="future mutation", score=10000)
    s.items.append(dict(s.items[-1], id="future-new", title="future addition"))
    s.vectors = s.vectors.copy()
    s.vectors[[0, -1]] = s.vectors[[1, 2]]
    s.vectors.setflags(write=False)
    s.index = {a["id"]: i for i, a in enumerate(s.items)}
    now += 86400
    s.refresh_at = now
    with db() as c:
        c.execute("UPDATE users SET interests='[]'")
        c.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
            (
                "v6-future",
                "v6-future",
                samples[0]["user"],
                "future-new",
                "click",
                0,
                now,
                1,
            ),
        )
    for name, store in stores.items():
        s.recommend(samples[0]["user"], replay=store)
        reader = type(readers[name])(store.root)
        for row in raw:
            assert (
                reader.replay(row["receipts"][name])["catalog_version"]
                == old_results["initial_catalog_version"]
            )
    stores["sqlite"].close()

    def quantiles(values):
        return dict(
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
            maximum=float(max(values)),
        )

    latency = {}
    for group in ("all", 0, 1, 3, 5, 10, 25):
        rows = raw if group == "all" else [r for r in raw if r["history"] == group]
        latency[str(group)] = {
            name: quantiles([r["ms"][name] for r in rows]) for name in methods
        }
        for name in stores:
            latency[str(group)][name + "_overhead"] = quantiles(
                [r["ms"][name] - r["ms"]["plain"] for r in rows]
            )
    small = [r for r in raw if r["history"] <= 20]
    sizes = {
        name: quantiles([r["request_bytes"][name] for r in small]) for name in stores
    }
    physical = quantiles([r["physical_growth_bytes"]["sqlite"] for r in small])
    report = dict(
        provenance=identity,
        synthetic_only=True,
        efficacy_authorized=False,
        deployed=False,
        baseline="c1d6296",
        frozen_files={p.name: file_digest(p) for p in frozen.iterdir()},
        source_inputs={
            name: file_digest(previous / name)
            for name in ["results.json", "timings.jsonl", "isolated.sqlite"]
        },
        catalog_version=old_results["initial_catalog_version"],
        catalog_items=len(catalog.rows),
        requests=120,
        latency_ms=latency,
        small_request_bytes=sizes,
        small_sqlite_physical_growth_bytes=physical,
        storage=dict(
            initial=initial_storage,
            initial_allocated=initial_allocated,
            before_checkpoint=before_checkpoint,
            before_checkpoint_allocated=before_checkpoint_allocated,
            after_checkpoint=after_checkpoint,
            after_checkpoint_allocated=after_checkpoint_allocated,
            checkpoint_ms=checkpoint_ms,
            checkpoint_result=checkpoint,
            acknowledged_records_before_future=121,
            amortized_before_checkpoint_bytes=(
                sum(before_checkpoint.values()) - sum(initial_storage.values())
            )
            / 121,
        ),
        catalog_registration_ms=registration,
        sqlite_replay_ms=quantiles(replay_ms),
        sqlite_cold_reader_ms=cold_reader_ms,
        exact_matches_per_recorder=120,
        old_records_preserved_after_future_per_recorder=120,
        environment=dict(
            sqlite=sqlite3.sqlite_version,
            numpy=np.__version__,
            python=sys.version,
            blas_threads=1,
        ),
        gates=dict(
            capture_overhead_p95_limit_ms=10,
            capture_overhead_pass=latency["all"]["sqlite_overhead"]["p95"] <= 10,
            small_request_limit_bytes=32768,
            small_request_pass=sizes["sqlite"]["maximum"] <= 32768
            and physical["maximum"] <= 32768,
        ),
        limitations=[
            "Frozen actual vectors and explicit vectors; encoding excluded as in warm v5 serving",
            "Sequential 120 synthetic requests on shared CPU; not concurrency capacity evidence",
            "FULL synchronous commits, default 1000-page autocheckpoint retained; explicit checkpoint timed separately",
            "Storage includes WAL and SHM; allocated filesystem blocks also reported separately",
            "Same numerical environment only; new reader does not evict OS page cache",
        ],
    )
    if identity != code_identity():
        raise ValueError("Source changed during measurement")
    with (out / "results.json").open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
