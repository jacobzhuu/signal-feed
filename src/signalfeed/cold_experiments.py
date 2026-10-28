"""Paired cold evaluation, validation-only grid selection, raw predictions retained."""

import argparse
import itertools
import json
import time
from collections import defaultdict
from dataclasses import asdict
import numpy as np
import torch
from .coldfuse import ColdFuse, Config, BUDGETS, state, matrix, weights, fusion, blend
from .cold_protocol import prototypes, simulate, sample
from .cold_training import load
from .features import Features
from .data import load_rows
from .ranker import load_ranker
from .generator import Generator
from .retrieval import adaptive, fuse, load_gate
from .evaluation import ranking_metrics, recall_metrics, average
from .provenance import run_identity, sha256, code_identity
from .settings import path

SCENARIOS = ("no_explicit", "proxy_explicit")
ABLATIONS = ("no_prior", "no_gen", "fixed_gates", "no_shrinkage", "no_dnn")


def context(
    core, h, prior, eligible, config, model, impression_ids, *, beams=None, shrink=True
):
    start = time.perf_counter()
    s = state(core.x, h, prior, config.tau, shrink)
    ch = core.channels(s, eligible, beams=beams)
    channel_seconds = time.perf_counter() - start
    ids = sorted(set(impression_ids).union(*(set(v) for v in ch["lists"].values())))
    x = matrix(core.x, ids, s)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x)).numpy()
    return dict(
        state=s,
        channels=ch,
        features=dict(zip(ids, x)),
        logits=dict(zip(ids, logits)),
        costs=dict(
            channels_seconds=channel_seconds,
            features_dnn_seconds=time.perf_counter() - start - channel_seconds,
        ),
    )


def compose(ctx, config, impression_ids, ablation=None):
    s, ch = ctx["state"], ctx["channels"]
    gate = weights(
        s,
        ch["cf_support"],
        ch["concentration"],
        config,
        fixed=ablation == "fixed_gates",
    )
    lists = {
        k: v for k, v in ch["lists"].items() if ablation != "no_gen" or k != "generated"
    }
    ids, base = fusion(lists, gate)
    base_map = dict(zip(ids, base))
    all_ids = list(dict.fromkeys(ids + impression_ids))
    x = np.array([ctx["features"][i] for i in all_ids], dtype="float32")
    scores = blend(
        x,
        [base_map.get(i, 0) for i in all_ids],
        None if ablation == "no_dnn" else np.array([ctx["logits"][i] for i in all_ids]),
        config.dnn,
    )
    score_map = dict(zip(all_ids, scores))
    pred = sorted(ids, key=lambda i: (-score_map[i], i))
    return pred, [float(score_map[i]) for i in impression_ids]


def baselines(f, core, ctx, h, impression_ids, old_ranker, gate):
    ch, s = ctx["channels"], ctx["state"]
    lists = ch["lists"]
    variants = dict(
        popularity=lists["popular"],
        content=lists["content"],
        itemcf=lists["itemcf"],
        sid_exact=ch["exact"],
        sid_prefix=lists["generated"],
        fixed_fusion=fuse(lists),
        repaired_adaptive=adaptive(lists, len(h), gate),
    )
    r = dict(history=[f.news[i]["id"] for i in h])
    ids = list(dict.fromkeys(variants["repaired_adaptive"] + impression_ids))
    with torch.inference_mode():
        logits = old_ranker(torch.from_numpy(f.matrix(r, ids, stats=False))).numpy()
    scores = dict(zip(ids, logits))
    variants["repaired_dnn"] = sorted(
        variants["repaired_adaptive"], key=lambda i: (-scores[i], i)
    )
    rank = {}
    for name, pred in variants.items():
        # Impression scoring for sparse retrievers: absent candidates tie at 0.
        score = {i: 1 / (61 + p) for p, i in enumerate(pred)}
        rank[name] = [score.get(i, 0) for i in impression_ids]
    rank["popularity"] = f.pop[impression_ids].tolist()
    rank["content"] = (f.x[impression_ids] @ s.profile).tolist()
    rank["itemcf"] = (
        np.asarray(f.cf[h][:, impression_ids].sum(0)).ravel().tolist()
        if h
        else [0.0] * len(impression_ids)
    )
    rank["repaired_dnn"] = [float(scores[i]) for i in impression_ids]
    return variants, rank


def setup(seed):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
    torch.set_num_threads(2)
    f = Features().load()
    return (
        f,
        ColdFuse(f.x, f.pop, f.cf, f.codes, Generator(seed)),
        load(seed),
        prototypes(f),
    )


def eligibility(f, row, h):
    mask = f.first < row["time"]
    mask[h] = False
    return mask


def tune():
    start = time.time()
    source = run_identity(42, "valid")
    f, core, model, pv = setup(42)
    rows = sample(load_rows("valid"), f, 300)
    configs = [
        Config(*v)
        for v in itertools.product((1, 3, 5), (0.25, 0.75, 1.25), (0, 0.25, 0.5))
    ]
    values = [defaultdict(list) for _ in configs]
    dest = path("reports/cold-valid")
    if (dest / "trials.json").exists():
        raise FileExistsError("Validation already selected; use a new run")
    for n, row in enumerate(rows):
        targets = [f.index[i] for i, y in zip(row["items"], row["labels"]) if y]
        ids = [f.index[i] for i in row["items"]]
        for scenario in SCENARIOS:
            for k in BUDGETS:
                h, prior, _ = simulate(f, row, k, scenario, pv)
                eligible = eligibility(f, row, h)
                contexts, beams = {}, None
                for tau in (1, 3, 5):
                    ctx = context(
                        core,
                        h,
                        prior,
                        eligible,
                        Config(tau=tau),
                        model,
                        ids,
                        beams=beams,
                    )
                    contexts[tau] = ctx
                    beams = ctx["channels"]["beams"]
                for j, config in enumerate(configs):
                    pred, _ = compose(contexts[config.tau], config, ids)
                    values[j][f"{scenario}/{k}"].append(
                        recall_metrics(pred, targets)["recall50"]
                    )
        if n % 25 == 0:
            print(dict(validation=n + 1, seconds=time.time() - start), flush=True)
    trials = []
    for config, groups in zip(configs, values):
        groups = {key: float(np.mean(v)) for key, v in groups.items()}
        score = float(
            np.mean([v for key, v in groups.items() if not key.endswith("/10")])
        )
        trials.append(dict(config=asdict(config), recall50=score, groups=groups))
    chosen = max(trials, key=lambda t: t["recall50"])
    if source["source_sha256"] != code_identity()["source_sha256"]:
        raise RuntimeError("Source changed during validation")
    result = dict(
        **chosen,
        trials=trials,
        selection="equal scenario/budget validation Recall50; budgets 0/1/3/5 only",
        requests=len(rows),
        ids=[r["id"] for r in rows],
        provenance=source,
        cold_model_sha256=sha256(path("models/coldfuse-42") / "model.pt"),
        seconds=time.time() - start,
    )
    (dest / "trials.json").write_text(json.dumps(result, indent=2))
    (dest / "per_request_recall.json").write_text(json.dumps(values))
    (path("models") / "coldfuse.json").write_text(json.dumps(result, indent=2))
    print(dict(selected=chosen, seconds=result["seconds"]), flush=True)
    return result


def evaluate(seed=42, natural=False):
    start, source = time.time(), run_identity(seed, "test")
    f, core, model, pv = setup(seed)
    selection = json.loads((path("models") / "coldfuse.json").read_text())
    if selection["provenance"]["split"] != "valid" or len(selection["trials"]) != 81:
        raise ValueError("ColdFuse requires the complete validation grid")
    for key in ("source_sha256", "data"):
        if selection["provenance"][key] != source[key]:
            raise ValueError("Validation and test sources differ; use a fresh run")
    config = Config(**selection["config"])
    old_ranker, gate = load_ranker(seed), load_gate()
    rows = sample(load_rows("test"), f, 300 if natural else 1000, natural=natural)
    dest = path(f"reports/cold-{'natural' if natural else 'test'}-{seed}")
    file = dest / "predictions.jsonl"
    if file.exists():
        raise FileExistsError("Never overwrite test predictions")
    aggregates, coverage, costs = defaultdict(list), defaultdict(set), defaultdict(list)
    with file.open("w") as out:
        for n, row in enumerate(rows):
            targets = [f.index[i] for i, y in zip(row["items"], row["labels"]) if y]
            ids = [f.index[i] for i in row["items"]]
            scenarios = ("natural",) if natural else SCENARIOS
            for scenario in scenarios:
                for k in (len(f.history(row)),) if natural else BUDGETS:
                    tick = time.perf_counter()
                    h, prior, older = (
                        (f.history(row), np.zeros(f.x.shape[1]), [])
                        if natural
                        else simulate(f, row, k, scenario, pv)
                    )
                    eligible = eligibility(f, row, h)
                    ctx = context(core, h, prior, eligible, config, model, ids)
                    cold_pred, cold_rank = compose(ctx, config, ids)
                    ctx["costs"]["coldfuse_seconds"] = time.perf_counter() - tick
                    variants, ranking = baselines(
                        f, core, ctx, h, ids, old_ranker, gate
                    )
                    variants["coldfuse"], ranking["coldfuse"] = cold_pred, cold_rank
                    for ablation in ABLATIONS:
                        ac = ctx
                        if ablation == "no_prior" and prior.any():
                            ac = context(
                                core,
                                h,
                                np.zeros_like(prior),
                                eligible,
                                config,
                                model,
                                ids,
                            )
                        elif ablation == "no_shrinkage":
                            ac = context(
                                core,
                                h,
                                prior,
                                eligible,
                                config,
                                model,
                                ids,
                                beams=ctx["channels"]["beams"],
                                shrink=False,
                            )
                        variants[ablation], ranking[ablation] = compose(
                            ac, config, ids, ablation
                        )
                    group = (
                        f"{scenario}/{k}"
                        if not natural
                        else ("natural/0" if k == 0 else "natural/1-5")
                    )
                    reachable = [i for i in targets if eligible[i]]
                    metric = {}
                    for name, pred in variants.items():
                        metric[name] = {
                            **recall_metrics(pred, targets),
                            **ranking_metrics(row["labels"], ranking[name]),
                            "reachable_recall50": len(set(pred[:50]) & set(reachable))
                            / len(set(reachable))
                            if reachable
                            else None,
                            "reachable_rate": len(set(reachable)) / len(set(targets))
                            if targets
                            else 0,
                            "candidates": len(pred),
                        }
                        aggregates[group, name].append(metric[name])
                        coverage[group, name].update(pred[:50])
                    elapsed = time.perf_counter() - tick
                    costs[group].append(
                        dict(**ctx["costs"], all_baselines_ablations_seconds=elapsed)
                    )
                    record = dict(
                        id=row["id"],
                        user=row["user"],
                        group=group,
                        k=k,
                        observed=h,
                        proxy_source=older,
                        prior_available=bool(prior.any()),
                        targets=targets,
                        reachable=reachable,
                        items=ids,
                        labels=row["labels"],
                        variants=variants,
                        scores=ranking,
                        metrics=metric,
                        gates=weights(
                            ctx["state"],
                            ctx["channels"]["cf_support"],
                            ctx["channels"]["concentration"],
                            config,
                        ),
                        beam_concentration=ctx["channels"]["concentration"],
                        costs=costs[group][-1],
                    )
                    out.write(json.dumps(record) + "\n")
            if n % 25 == 0:
                print(
                    dict(
                        seed=seed,
                        natural=natural,
                        test=n + 1,
                        seconds=time.time() - start,
                    ),
                    flush=True,
                )
    if source["source_sha256"] != code_identity()["source_sha256"]:
        raise RuntimeError("Source changed during evaluation")
    report = dict(
        seed=seed,
        natural=natural,
        requests=len(rows),
        ids=[r["id"] for r in rows],
        config=asdict(config),
        provenance=source,
        cold_model_sha256=sha256(path(f"models/coldfuse-{seed}") / "model.pt"),
        selection_sha256=sha256(path("models") / "coldfuse.json"),
        predictions_sha256=sha256(file),
        seconds=time.time() - start,
        groups={
            group: {
                name: dict(
                    **average(metrics),
                    requests=len(metrics),
                    catalog_coverage=len(coverage[group, name]) / len(f.news),
                )
                for (g, name), metrics in aggregates.items()
                if g == group
            }
            for group in sorted(costs)
        },
        costs={
            group: {
                key: dict(
                    p50=float(np.percentile([r[key] for r in values], 50)),
                    p95=float(np.percentile([r[key] for r in values], 95)),
                )
                for key in values[0]
            }
            for group, values in costs.items()
        },
        notes=[
            "MIND MRR convention: average reciprocal rank of all positives per impression",
            "Proxy interests from disjoint older truncated history; not real onboarding",
            "Sparse retriever impression scores tie at zero outside retrieved candidates",
            "All positive targets retained; catalog uses first appearance proxy strictly before request",
            "Original DNN has no explicit-interest feature; same prior enters its candidate channels",
            "No trained score is calibrated CTR; natural cohort is separate from mature-user truncation",
        ],
    )
    (dest / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("tune", "evaluate"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--natural", action="store_true")
    args = parser.parse_args()
    tune() if args.action == "tune" else evaluate(args.seed, args.natural)
