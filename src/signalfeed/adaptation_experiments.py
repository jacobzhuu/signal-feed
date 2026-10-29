"""Independent-user experiments. Selection, guard and test are separate actions."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy import sparse
from .adaptation import (
    VERSION,
    GRID,
    AdaptConfig,
    signals,
    interest_scores,
    fuse_interest,
    cohort_key,
)
from .coldfuse import ColdFuse, Config, BUDGETS, weights, itemcf_scores
from .cold_protocol import simulate, prototypes
from .cold_training import load
from .cold_experiments import context, compose, eligibility
from .features import Features
from .generator import Generator
from .retrieval import top, fuse
from .evaluation import recall_metrics, ranking_metrics, average
from .provenance import sha256, code_identity, data_identity
from .settings import path

ABLATIONS = ("no_multi", "no_reliability", "no_side", "no_gen", "fixed_side")
FROZEN_FILES = (
    "adaptation.py",
    "adaptation_experiments.py",
    "coldfuse.py",
    "cold_protocol.py",
    "cold_experiments.py",
    "cold_training.py",
    "features.py",
    "generator.py",
    "retrieval.py",
    "evaluation.py",
    "sid.py",
)


def algorithm_identity():
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in FROZEN_FILES}


def extended(root):
    dataset = root / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("protocol") != "adaptation-v2-canonical-identity":
        raise ValueError(
            "Cross-release identities must be reconciled before evaluation"
        )
    audit = manifest["identity_audit"]
    if audit["matched_event_signatures"] != audit["exposed_event_signatures"]:
        raise ValueError("Exposed-event exclusion is incomplete")
    for name, meta in manifest["files"].items():
        if sha256(dataset / name) != meta["sha256"]:
            raise ValueError(f"Dataset changed: {name}")
    f = Features().load()
    pv = prototypes(f)
    original = len(f.news)
    names = sorted({n["category"] for n in f.news})
    f.news = json.loads((dataset / "news.json").read_text())
    names += sorted({n["category"] for n in f.news} - set(names))
    categories = {c: i for i, c in enumerate(names)}
    f.categories = np.array([categories[n["category"]] for n in f.news])
    f.index = {n["id"]: i for i, n in enumerate(f.news)}
    f.first = np.array([n["first_seen"] for n in f.news])
    f.x = np.load(dataset / "embeddings.npy")
    f.codes = np.load(dataset / "codes.npy")
    size = len(f.news)
    old = f.cf.tocoo()
    f.cf = sparse.csr_matrix((old.data, (old.row, old.col)), shape=(size, size))
    f.pop = np.pad(f.pop, (0, size - original))
    return f, pv, manifest


def load_rows(root, split):
    rows = []
    for cohort in ("main", "natural"):
        with (root / f"dataset/{split}-{cohort}.jsonl").open() as stream:
            rows += [(cohort, json.loads(line)) for line in stream]
    return rows


def inputs(f, pv, cohort, row):
    if cohort == "natural":
        h = f.history(row)
        yield "natural", len(h), h, np.zeros(f.x.shape[1], dtype="float32")
    else:
        for scenario in ("no_explicit", "proxy_explicit"):
            for k in BUDGETS:
                h, prior, _ = simulate(f, row, k, scenario, pv)
                yield scenario, k, h, prior


def new_predictions(
    sig, eligible, pop, channels, config, impression_ids, ablation=None
):
    personal = bool(sig["state"].history or sig["state"].explicit.any())
    scores = interest_scores(sig, config, ablation) if personal else pop
    content = top(scores, eligible, 100)
    pred, fused, gates = fuse_interest(sig, content, channels, config, ablation)
    # Sparse fusion scoring is explicit, while dense interest has its own metric.
    return (
        pred,
        [float(fused.get(i, 0.0)) for i in impression_ids],
        gates,
        scores[impression_ids].tolist(),
    )


def collect(root, split, seed):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
    torch.set_num_threads(2)
    identity = algorithm_identity()
    provenance = code_identity()
    dest = root / f"reports/{split}-{seed}"
    dest.mkdir(exist_ok=True)
    if (dest / "predictions.jsonl").exists():
        raise FileExistsError(
            "Predictions already exist; never overwrite an evaluated run"
        )
    selection, decision = None, None
    if split != "select":
        selection = json.loads((root / "selection.json").read_text())
        if selection["algorithm"] != identity:
            raise ValueError("Algorithm changed after validation selection")
    if split == "test":
        decision = json.loads((root / "guard.json").read_text())
        if decision["selection_sha256"] != sha256(root / "selection.json"):
            raise ValueError("Guard belongs to another selection")
        if decision["algorithm"] != identity:
            raise ValueError("Guard algorithm changed")
    f, pv, manifest = extended(root)
    if selection and selection["dataset_sha256"] != sha256(
        root / "dataset/manifest.json"
    ):
        raise ValueError("Dataset changed since selection")
    model, gen = load(seed), Generator(seed)
    core = ColdFuse(f.x, f.pop, f.cf, f.codes, gen)
    config = Config(
        **json.loads((path("models") / "coldfuse.json").read_text())["config"]
    )
    chosen = AdaptConfig(**selection["config"]) if selection else None
    candidates = GRID if split == "select" else (chosen,)
    rows = load_rows(root, split)
    aggregate = defaultdict(list)
    start = time.perf_counter()
    with (dest / "predictions.jsonl").open("x") as output:
        for n, (cohort, row) in enumerate(rows):
            impression = [f.index[i] for i in row["items"]]
            targets = [i for i, y in zip(impression, row["labels"]) if y]
            for scenario, k, h, prior in inputs(f, pv, cohort, row):
                eligible = eligibility(f, row, h)
                tick = time.perf_counter()
                ctx = context(core, h, prior, eligible, config, model, impression)
                pred, rank = compose(ctx, config, impression)
                cold_seconds = time.perf_counter() - tick
                channels = ctx["channels"]["lists"]
                tick = time.perf_counter()
                sig = signals(f.x, h, prior)
                signal_seconds = time.perf_counter() - tick
                variants = dict(
                    popularity=channels["popular"],
                    content=channels["content"],
                    content_safe=channels["content"] or channels["popular"],
                    itemcf=channels["itemcf"],
                    sid_prefix=channels["generated"],
                    fixed_fusion=fuse(channels),
                    coldfuse=pred,
                )
                scoring = {}
                for name, v in variants.items():
                    score_map = {i: 1 / (61 + j) for j, i in enumerate(v)}
                    scoring[name] = [score_map.get(i, 0) for i in impression]
                scoring["content"] = (f.x[impression] @ ctx["state"].profile).tolist()
                scoring["popularity"] = f.pop[impression].tolist()
                scoring["content_safe"] = (
                    scoring["content"] if channels["content"] else scoring["popularity"]
                )
                scoring["coldfuse"] = rank
                scoring["itemcf"] = itemcf_scores(
                    f.cf, ctx["state"].history, impression
                ).tolist()
                costs = dict(
                    coldfuse=cold_seconds, signal=signal_seconds, **ctx["costs"]
                )
                gates = {}
                trial_recalls = []
                for cfg in candidates:
                    tick = time.perf_counter()
                    p, scores, gate, dense = new_predictions(
                        sig, eligible, f.pop, channels, cfg, impression
                    )
                    trial_recalls.append(recall_metrics(p, targets)["recall50"])
                    if chosen is not None:
                        variants["adaptation"], scoring["adaptation"] = p, scores
                        costs["adaptation_after_channels"] = (
                            signal_seconds + time.perf_counter() - tick
                        )
                        gates = gate
                        # Dense-interest-only baseline uses identical learned representation.
                        raw_score = (
                            interest_scores(sig, cfg) if (h or prior.any()) else f.pop
                        )
                        variants["interest_only"] = top(raw_score, eligible, 100)
                        scoring["interest_only"] = dense
                if chosen is not None:
                    for ablation in ABLATIONS:
                        p, score, _, _ = new_predictions(
                            sig, eligible, f.pop, channels, chosen, impression, ablation
                        )
                        variants[ablation], scoring[ablation] = p, score
                    if decision:
                        accepted = decision["groups"][cohort_key(h, prior)]["accepted"]
                        name = "adaptation" if accepted else "content_safe"
                        variants["guarded"], scoring["guarded"] = (
                            variants[name],
                            scoring[name],
                        )
                metrics = {
                    name: {
                        **recall_metrics(v, targets),
                        **ranking_metrics(row["labels"], scoring[name]),
                    }
                    for name, v in variants.items()
                }
                group = f"{scenario}/{k}"
                for name, m in metrics.items():
                    aggregate[group, name].append(m)
                record = dict(
                    id=row["id"],
                    user=row["user"],
                    time=row["time"],
                    group=group,
                    cohort=cohort,
                    k=k,
                    observed=h,
                    prior_available=bool(prior.any()),
                    guard_group=cohort_key(h, prior),
                    targets=targets,
                    reachable=[i for i in targets if eligible[i]],
                    new_targets=[i for i in targets if i >= manifest["old_items"]],
                    known_large_train=row["known_large_train"],
                    first_dev_request=row["first_dev_request"],
                    coherence=sig["coherence"],
                    prior_agreement=sig["agreement"],
                    variants=variants,
                    items=impression,
                    labels=row["labels"],
                    scores=scoring,
                    metrics=metrics,
                    trials=trial_recalls if split == "select" else None,
                    gates=gates,
                    cold_gates=weights(
                        ctx["state"],
                        ctx["channels"]["cf_support"],
                        ctx["channels"]["concentration"],
                        config,
                    ),
                    beam_concentration=ctx["channels"]["concentration"],
                    costs=costs,
                )
                output.write(json.dumps(record) + "\n")
            if n % 25 == 0:
                print(
                    dict(
                        split=split,
                        seed=seed,
                        requests=n + 1,
                        seconds=time.perf_counter() - start,
                    ),
                    flush=True,
                )
    if identity != algorithm_identity():
        raise RuntimeError("Algorithm changed during collection; preserve rejected run")
    report = dict(
        split=split,
        seed=seed,
        requests=len(rows),
        algorithm=identity,
        provenance=provenance,
        dataset_sha256=sha256(root / "dataset/manifest.json"),
        data=data_identity(),
        models={
            kind: sha256(path(f"models/{kind}-{seed}") / "model.pt")
            for kind in ("sid", "coldfuse")
        },
        predictions_sha256=sha256(dest / "predictions.jsonl"),
        selection_sha256=sha256(root / "selection.json") if selection else None,
        guard_sha256=sha256(root / "guard.json") if decision else None,
        seconds=time.perf_counter() - start,
        groups={
            g: {name: average(ms) for (gg, name), ms in aggregate.items() if gg == g}
            for g in sorted({g for g, _ in aggregate})
        },
    )
    (dest / "metrics.json").write_text(json.dumps(report, indent=2))


def read_records(root, split, seeds):
    for seed in seeds:
        file = root / f"reports/{split}-{seed}/predictions.jsonl"
        meta = json.loads(file.with_name("metrics.json").read_text())
        if (
            sha256(file) != meta["predictions_sha256"]
            or meta["algorithm"] != algorithm_identity()
        ):
            raise ValueError("Changed predictions/algorithm")
        with file.open() as stream:
            for line in stream:
                yield seed, json.loads(line)


def select(root):
    dest = root / "selection.json"
    if dest.exists():
        raise FileExistsError("Selection sealed")
    values = []
    groups = defaultdict(list)
    for _, r in read_records(root, "select", (42,)):
        if r["cohort"] == "main" and r["k"] != 10:
            values.append(r["trials"])
            groups[r["group"]].append(r["trials"])
    means = np.mean(values, axis=0)
    # Ties favor lower side cost, then simpler interests; order is preregistered.
    best = min(
        range(len(GRID)),
        key=lambda i: (-means[i], GRID[i].side, GRID[i].multi, GRID[i].recent),
    )
    result = dict(
        version=VERSION,
        algorithm=algorithm_identity(),
        dataset_sha256=sha256(root / "dataset/manifest.json"),
        config=asdict(GRID[best]),
        selection="main 0/1/3/5, two scenarios, Recall50; seed42 only",
        trials=[
            dict(
                config=asdict(c),
                recall50=float(means[i]),
                groups={g: float(np.mean(v, axis=0)[i]) for g, v in groups.items()},
            )
            for i, c in enumerate(GRID)
        ],
        predictions_sha256=sha256(root / "reports/select-42/predictions.jsonl"),
    )
    dest.write_text(json.dumps(result, indent=2))
    print(dict(selected=result["config"], score=float(means[best])), flush=True)


def paired_test(values, seed=20261029, repeats=10000):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(n=0, mean=0.0, ci=[0.0, 0.0], p=1.0)
    rng = np.random.default_rng(seed)
    boot, extreme = [], 0
    observed = abs(x.mean())
    for start in range(0, repeats, 100):
        count = min(100, repeats - start)
        boot.extend(x[rng.integers(len(x), size=(count, len(x)))].mean(1).tolist())
        flipped = (x * rng.choice([-1, 1], size=(count, len(x)))).mean(1)
        extreme += int((np.abs(flipped) >= observed - 1e-15).sum())
    return dict(
        n=len(x),
        mean=float(x.mean()),
        ci=np.quantile(boot, [0.025, 0.975]).tolist(),
        p=float((extreme + 1) / (repeats + 1)),
    )


def holm(tests):
    keys = sorted(tests, key=lambda k: tests[k]["p"])
    running = 0.0
    for j, key in enumerate(keys):
        running = max(running, min(1.0, tests[key]["p"] * (len(keys) - j)))
        tests[key]["p_holm"] = running


def guard(root):
    if (root / "guard.json").exists():
        raise FileExistsError("Guard decision sealed")
    values = defaultdict(lambda: defaultdict(list))
    # Seed repetitions and multiple budgets of a user never increase sample size.
    for _, r in read_records(root, "guard", (42, 43, 44)):
        delta = (
            r["metrics"]["adaptation"]["recall50"]
            - r["metrics"]["content_safe"]["recall50"]
        )
        values[r["guard_group"]][r["user"]].append(delta)
    tests = {
        f"{s}/{b}": paired_test([np.mean(v) for v in values[f"{s}/{b}"].values()])
        for s in ("prior", "no_prior")
        for b in ("0", "1", "2-5", "6+")
    }
    holm(tests)
    for result in tests.values():
        result["accepted"] = bool(
            result["n"] >= 100 and result["ci"][0] > 0 and result["p_holm"] <= 0.05
        )
    selection = json.loads((root / "selection.json").read_text())
    result = dict(
        version=VERSION,
        domain="mind-large-independent-users",
        config=selection["config"],
        selection_sha256=sha256(root / "selection.json"),
        algorithm=algorithm_identity(),
        groups=tests,
        rule="n>=100, user bootstrap lower95>0, two-sided sign-flip Holm8<=0.05",
        sources={
            str(s): sha256(root / f"reports/guard-{s}/predictions.jsonl")
            for s in (42, 43, 44)
        },
    )
    (root / "guard.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "select", "guard"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("select", "guard", "test"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.action == "collect":
        collect(args.run, args.split, args.seed)
    elif args.action == "select":
        select(args.run)
    else:
        guard(args.run)
