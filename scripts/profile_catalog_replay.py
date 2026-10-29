"""Profile the frozen v5 benchmark once; instrumented timing is not admission evidence."""

import argparse
import cProfile
import importlib.util
import json
import os
from pathlib import Path
import pstats
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hf-home", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    # Imports must see the same isolated environment as the wrapped benchmark.
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
    # settings creates HOME at import; reserve output through benchmark instead.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="replay-profile-import-") as temporary:
        os.environ["SIGNALFEED_HOME"] = temporary
        import signalfeed.serving as serving
        import signalfeed.settings as settings
        import signalfeed.store as store

    settings.ROOT = out / "home"
    store.ROOT = settings.ROOT
    os.environ["SIGNALFEED_HOME"] = str(settings.ROOT)
    original = serving.ContentService.recommend
    profiles = {}

    def measured(self, *a, **kw):
        recorder = kw.get("replay")
        if recorder is None:
            return original(self, *a, **kw)
        profile = cProfile.Profile()
        result = profile.runcall(original, self, *a, **kw)
        profiles[recorder.last_request] = [
            dict(
                file=k[0],
                line=k[1],
                function=k[2],
                calls=v[1],
                exclusive_ms=v[2] * 1000,
                cumulative_ms=v[3] * 1000,
            )
            for k, v in pstats.Stats(profile).stats.items()
        ]
        return result

    serving.ContentService.recommend = measured
    benchmark = Path(__file__).with_name("benchmark_catalog_replay.py")
    spec = importlib.util.spec_from_file_location("replay_benchmark", benchmark)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [
        str(benchmark),
        "--db",
        str(args.db),
        "--output",
        str(out),
        "--hf-home",
        str(args.hf_home),
    ]
    try:
        module.main()
    finally:
        serving.ContentService.recommend = original
    samples = [json.loads(s) for s in (out / "timings.jsonl").read_text().splitlines()]
    selected = {s["measured_request"]: profiles[s["measured_request"]] for s in samples}
    aggregate = {}
    for stats in selected.values():
        for r in stats:
            key = (r["file"], r["line"], r["function"])
            if key not in aggregate:
                aggregate[key] = {
                    **r,
                    "calls": 0,
                    "exclusive_ms": 0,
                    "cumulative_ms": 0,
                }
            for name in ("calls", "exclusive_ms", "cumulative_ms"):
                aggregate[key][name] += r[name]
    report = dict(
        diagnostic_only=True,
        requests=len(selected),
        note="cProfile adds overhead; cumulative times overlap and must not be added. Formal admission requires uninstrumented measurement.",
        functions=sorted(aggregate.values(), key=lambda r: -r["exclusive_ms"]),
        per_request=selected,
    )
    with (out / "profile.json").open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["functions"][:15], indent=2))


if __name__ == "__main__":
    main()
