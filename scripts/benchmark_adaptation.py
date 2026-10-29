"""Standalone uncached inference costs on preregistered metadata-only samples."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from signalfeed.adaptation import (
    GuardedAdapter,
    safe_content,
    signals,
    interest_scores,
    fuse_interest,
)
from signalfeed.adaptation_experiments import (
    extended,
    inputs,
    load_rows,
    algorithm_identity,
)
from signalfeed.coldfuse import ColdFuse, Config, state
from signalfeed.cold_training import load
from signalfeed.cold_experiments import eligibility
from signalfeed.generator import Generator
from signalfeed.provenance import sha256, code_identity
from signalfeed.settings import path
from signalfeed.retrieval import top, fuse


def benchmark(root):
    target = root / "reports/inference_cost.json"
    if target.exists():
        raise FileExistsError("Preserve previous cost measurements")
    threadpool_limits(limits=1, user_api="blas")
    torch.set_num_threads(2)
    start = time.perf_counter()
    f, pv, _ = extended(root)
    model, generator = load(42), Generator(42)
    generation_calls = 0
    generate = generator.generate

    def counted_generate(*args, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        return generate(*args, **kwargs)

    generator.generate = counted_generate
    core = ColdFuse(f.x, f.pop, f.cf, f.codes, generator)
    config = Config(
        **json.loads((path("models") / "coldfuse.json").read_text())["config"]
    )
    decision = json.loads((root / "guard.json").read_text())
    assert decision["algorithm"] == algorithm_identity()
    guarded = GuardedAdapter(decision, domain="mind-large-independent-users")
    always = dict(decision, groups={g: dict(accepted=True) for g in decision["groups"]})
    unguarded = GuardedAdapter(always, domain="mind-large-independent-users")
    load_seconds = time.perf_counter() - start
    rows = load_rows(root, "test")
    chosen, counts = [], defaultdict(int)
    for cohort, row in rows:
        if counts[cohort] < 100:
            chosen.append((cohort, row))
            counts[cohort] += 1
    records, summaries = [], defaultdict(lambda: defaultdict(list))
    names = (
        "popularity",
        "content",
        "content_safe",
        "itemcf",
        "sid_prefix",
        "fixed_fusion",
        "coldfuse",
        "adaptation",
        "interest_only",
        "no_multi",
        "no_reliability",
        "no_side",
        "no_gen",
        "fixed_side",
        "guarded",
    )
    for number, (cohort, row) in enumerate(chosen):
        for scenario, k, h, prior in inputs(f, pv, cohort, row):
            mask = eligibility(f, row, h)
            provider_calls = 0

            def provider():
                nonlocal provider_calls
                provider_calls += 1
                return core.channels(state(f.x, h, prior), mask)["lists"]

            def content_only():
                s = state(f.x, h, prior)
                return dict(
                    ids=top(f.x @ s.profile, mask, 100) if s.profile.any() else []
                )

            def cf_only():
                s = state(f.x, h, prior)
                scores = (
                    np.asarray(f.cf[s.history].sum(0)).ravel()
                    if s.history
                    else np.zeros(len(f.x))
                )
                return dict(ids=top(scores, mask & (scores > 0), 100))

            def adaptation_variant(ablation):
                sig = signals(f.x, h, prior)
                personal = bool(sig["state"].history or sig["state"].explicit.any())
                scores = (
                    interest_scores(sig, unguarded.config, ablation)
                    if personal
                    else f.pop
                )
                content = top(scores, mask, 100)
                if ablation in ("interest_only", "no_side") or not personal:
                    return dict(ids=content)
                channels = (
                    core.channels(state(f.x, h, prior), mask, beams=[])["lists"]
                    if ablation == "no_gen"
                    else provider()
                )
                ids, _, _ = fuse_interest(
                    sig, content, channels, unguarded.config, ablation
                )
                return dict(ids=ids)

            functions = dict(
                popularity=lambda: dict(ids=top(f.pop, mask, 100)),
                content=content_only,
                content_safe=lambda: dict(ids=safe_content(f.x, h, prior, mask, f.pop)),
                itemcf=cf_only,
                sid_prefix=lambda: dict(ids=provider()["generated"]),
                fixed_fusion=lambda: dict(ids=fuse(provider())),
                coldfuse=lambda: core.recommend(h, mask, prior, config, model),
                adaptation=lambda: unguarded.recommend(
                    f.x, f.pop, h, prior, mask, provider
                ),
                guarded=lambda: guarded.recommend(f.x, f.pop, h, prior, mask, provider),
            )
            for ablation in (
                "interest_only",
                "no_multi",
                "no_reliability",
                "no_side",
                "no_gen",
                "fixed_side",
            ):
                functions[ablation] = lambda ablation=ablation: adaptation_variant(
                    ablation
                )
            ordered = names[number % len(names) :] + names[: number % len(names)]
            for name in ordered:
                before = provider_calls
                before_generation = generation_calls
                tick = time.perf_counter()
                result = functions[name]()
                seconds = time.perf_counter() - tick
                group = f"{scenario}/{k}"
                summaries[group][name].append(seconds)
                records.append(
                    dict(
                        user=row["user"],
                        id=row["id"],
                        group=group,
                        method=name,
                        seconds=seconds,
                        ids=result["ids"],
                        status=result.get("status"),
                        provider_calls=provider_calls - before,
                        generation_calls=generation_calls - before_generation,
                    )
                )
        if number % 25 == 0:
            print(
                dict(cost_requests=number + 1, seconds=time.perf_counter() - start),
                flush=True,
            )
    raw = root / "reports/inference_cost_predictions.jsonl"
    with raw.open("x") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    report = dict(
        provenance=code_identity(),
        algorithm=algorithm_identity(),
        dataset_sha256=sha256(root / "dataset/manifest.json"),
        guard_sha256=sha256(root / "guard.json"),
        raw_sha256=sha256(raw),
        model_and_data_load_seconds=load_seconds,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_threads=2,
        blas_threads=1,
        cached_results=False,
        catalog_items=len(f.news),
        sample=dict(counts),
        records=len(records),
        groups={
            g: {
                name: dict(
                    n=len(v),
                    p50_ms=float(np.percentile(v, 50) * 1000),
                    p95_ms=float(np.percentile(v, 95) * 1000),
                )
                for name, v in m.items()
            }
            for g, m in summaries.items()
        },
        provider_calls={
            name: sum(r["provider_calls"] for r in records if r["method"] == name)
            for name in ("adaptation", "guarded")
        },
        generation_calls={
            name: sum(r["generation_calls"] for r in records if r["method"] == name)
            for name in names
        },
        notes=[
            "First100 hash-selected users per cohort, all scenarios/budgets; fixed rotating method order",
            "No HTTP, database or text encoder request cost; catalog size recorded explicitly",
            "Resident models and prebuilt SID index; index construction included only in initialization time",
            "SID-only uses the existing shared channel builder, including its auxiliary content/CF work",
            "Shared host, CPU only; peak RSS includes all loaded baselines, not isolated adapter memory",
            "Generation/DNN bypassed entirely for rejected guard groups; no future feedback used",
        ],
    )
    target.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    benchmark(args.run)
