"""Separate impression ranking from full-catalog retrieval; persist predictions."""

import json
import time
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .settings import path
from .features import Features
from .data import load_rows
from .ranker import load_ranker
from .generator import Generator
from .retrieval import adaptive, RetrievalPolicy, load_gate, POLICY_VERSION
from .provenance import run_identity


def ranking_metrics(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores)
    order = np.argsort(-s, kind="stable")
    ranked = y[order]
    n = int(y.sum())
    result = {
        "auc": float(roc_auc_score(y, s)) if 0 < n < len(y) else None,
        "mrr": float((ranked / np.arange(1, len(y) + 1)).sum() / n) if n else None,
    }
    for k in [5, 10]:
        denom = (np.sort(y)[::-1][:k] / np.log2(np.arange(min(k, len(y))) + 2)).sum()
        result[f"ndcg{k}"] = (
            float((ranked[:k] / np.log2(np.arange(min(k, len(y))) + 2)).sum() / denom)
            if denom
            else None
        )
    return result


def average(rows):
    return (
        {
            k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows)
            else None
            for k in rows[0]
        }
        if rows
        else {}
    )


def recall_metrics(pred, positive):
    pred = list(dict.fromkeys(pred))
    pos = set(positive)
    n = len(pos)
    result = {f"recall{k}": len(set(pred[:k]) & pos) / n if n else 0 for k in [20, 50]}
    for k in [5, 10]:
        dcg = sum((i in pos) / np.log2(j + 2) for j, i in enumerate(pred[:k]))
        ideal = sum(1 / np.log2(j + 2) for j in range(min(k, n)))
        result[f"end_to_end_ndcg{k}"] = dcg / ideal if ideal else 0
    return result


def evaluate(seed=42, ranking_limit=0, retrieval_limit=1000, split="test"):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")
    torch.set_num_threads(4)
    start = time.time()
    f = Features().load()
    ranker = load_ranker(seed)
    dest = path(f"reports/{split}-{seed}")
    rows = load_rows(split, ranking_limit)
    ablation_path = path("models") / f"ranker-no-personal-{seed}" / "model.pt"
    no_personal = load_ranker(seed, False) if ablation_path.exists() else None
    rank_reports = defaultdict(list)
    rank_cohorts = defaultdict(lambda: defaultdict(list))
    train_users = {r["user"] for r in load_rows("train")}

    def cohorts(r):
        n = len(f.history(r))
        result = [
            "all",
            "history_0" if n == 0 else "history_1_5" if n <= 5 else "history_6_20",
            "unseen_user" if r["user"] not in train_users else "seen_user",
        ]
        positives = [f.index[i] for i, y in zip(r["items"], r["labels"]) if y]
        if any(f.first[i] >= f.cutoff for i in positives):
            result += ["new_item", result[1] + "/new_item"]
        return result

    with (
        open(dest / "ranking_predictions.jsonl", "w") as predictions,
        torch.inference_mode(),
    ):
        for r in rows:
            ids = [f.index[i] for i in r["items"]]
            x = f.matrix(r, ids, stats=False)
            xp = f.matrix(r, ids, personal=False, stats=False)
            scores = {
                "popularity": f.pop[ids],
                "content": f.x[ids] @ f.vector(r),
                "dnn": ranker(torch.from_numpy(x)).numpy(),
                "dnn_no_personalization": ranker(torch.from_numpy(xp)).numpy(),
            }
            if no_personal is not None:
                scores["dnn_no_personalization_retrained"] = no_personal(
                    torch.from_numpy(xp)
                ).numpy()
            for name, s in scores.items():
                metric = ranking_metrics(r["labels"], s)
                rank_reports[name].append(metric)
                for cohort in cohorts(r):
                    rank_cohorts[cohort][name].append(metric)
            predictions.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "user": r["user"],
                        "cohorts": cohorts(r),
                        "labels": r["labels"],
                        "items": r["items"],
                        "scores": {k: v.tolist() for k, v in scores.items()},
                    }
                )
                + "\n"
            )
    print({"ranking_done": len(rows), "seconds": time.time() - start}, flush=True)
    gen = Generator(seed)
    retrieval = defaultdict(list)
    coverage = defaultdict(set)
    overlaps = []
    unique_hits = []
    new_hits = defaultdict(list)
    latencies = []
    rrows = load_rows(split, retrieval_limit)
    gate = (
        {
            "threshold": 5,
            "weight": 1.5,
            "source": "untuned validation default",
            "policy_version": POLICY_VERSION,
        }
        if split == "valid"
        else load_gate()
    )
    policy = RetrievalPolicy(f, gen, gate)
    retrieval_cohorts = defaultdict(lambda: defaultdict(list))
    availability = []
    # Strict availability is first appearance < impression time, excluding
    # candidates whose only known appearance is this very impression.
    with (
        open(dest / "retrieval_predictions.jsonl", "w") as predictions,
        torch.inference_mode(),
    ):
        for idx, r in enumerate(rrows):
            tick = time.perf_counter()
            eligible = f.first < r["time"]
            eligible[f.history(r)] = False
            variants = policy.candidates(r, eligible)
            full = variants["adaptive_fusion"]
            no_gen = variants["adaptive_no_gen"]
            prefix, traditional = variants["generative_prefix"], variants["traditional"]
            for name, candidates in [("full_dnn", full), ("no_gen_dnn", no_gen)]:
                if candidates:
                    scores = ranker(
                        torch.from_numpy(f.matrix(r, candidates, stats=False))
                    ).numpy()
                    variants[name] = [
                        candidates[i] for i in np.argsort(-scores, kind="stable")
                    ]
                else:
                    variants[name] = []
            positives = [f.index[i] for i, y in zip(r["items"], r["labels"]) if y]
            new = [i for i in positives if f.first[i] >= f.cutoff]
            reachable = [i for i in positives if eligible[i]]
            availability.append(len(reachable) / len(positives) if positives else 0)
            for name, pred in variants.items():
                metric = recall_metrics(pred, positives)
                metric["reachable_recall50"] = (
                    len(set(pred[:50]) & set(reachable)) / len(set(reachable))
                    if reachable
                    else None
                )
                retrieval[name].append(metric)
                for cohort in cohorts(r):
                    retrieval_cohorts[cohort][name].append(metric)
                coverage[name].update(pred)
                if new:
                    new_hits[name].append(len(set(pred[:50]) & set(new)) / len(new))
            overlaps.append(
                len(set(prefix) & set(traditional))
                / max(1, len(set(prefix) | set(traditional)))
            )
            unique_hits.append(len((set(prefix) - set(traditional)) & set(positives)))
            latencies.append(time.perf_counter() - tick)
            predictions.write(
                json.dumps(
                    dict(
                        id=r["id"],
                        user=r["user"],
                        time=r["time"],
                        cohorts=cohorts(r),
                        positives=positives,
                        available_positives=reachable,
                        available_positive_count=int(
                            sum(eligible[i] for i in positives)
                        ),
                        variants=variants,
                        history_length=len(f.history(r)),
                    )
                )
                + "\n"
            )
            if idx % 100 == 0:
                print(
                    {"retrieval_done": idx + 1, "elapsed": time.time() - start},
                    flush=True,
                )
    report = dict(
        provenance=run_identity(seed, split),
        policy_version=POLICY_VERSION,
        dataset="MIND-small historical logs; not HN online effects",
        ranking_cohorts={
            c: {n: {"impressions": len(v), **average(v)} for n, v in models.items()}
            for c, models in rank_cohorts.items()
        },
        retrieval_cohorts={
            c: {n: {"impressions": len(v), **average(v)} for n, v in models.items()}
            for c, models in retrieval_cohorts.items()
        },
        positive_availability_rate=float(np.mean(availability)),
        seed=seed,
        ranking_impressions=len(rows),
        retrieval_impressions=len(rrows),
        ranking={k: average(v) for k, v in rank_reports.items()},
        retrieval={
            k: {
                **average(v),
                "catalog_coverage": len(coverage[k]) / len(f.news),
                "new_item_recall50": float(np.mean(new_hits[k]))
                if new_hits[k]
                else None,
            }
            for k, v in retrieval.items()
        },
        complementarity=dict(
            mean_jaccard=float(np.mean(overlaps)),
            unique_generated_positive_hits=int(sum(unique_hits)),
        ),
        retrieval_seconds_p50=float(np.percentile(latencies, 50)),
        retrieval_seconds_p95=float(np.percentile(latencies, 95)),
        seconds=time.time() - start,
        fusion_gate=gate,
        notes=[
            "DNN scores trained with sampled negatives are not calibrated CTR",
            "No-personalization is inference masking, not a retrained ablation",
            "All positive targets remain in recall denominator, including unavailable first-seen items",
            "Generated exact codes map collisions by train popularity; prefix expansion uses content similarity",
        ],
    )
    (dest / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def tune_fusion(seed=42):
    source = path(f"reports/valid-{seed}") / "retrieval_predictions.jsonl"
    rows = [json.loads(line) for line in source.open()]
    if len(rows) < 2000:
        raise ValueError(
            "Fusion selection requires at least 2,000 internal-validation requests"
        )
    metrics = json.loads(source.with_name("metrics.json").read_text())
    if (
        metrics.get("policy_version") != POLICY_VERSION
        or metrics["provenance"]["split"] != "valid"
    ):
        raise ValueError(
            "Fusion tuning requires current-policy internal validation predictions"
        )
    trials = []
    for threshold in [1, 5, 10]:
        for weight in [0.5, 1.0, 1.5, 2.0]:
            values = []
            for r in rows:
                v = r["variants"]
                pred = adaptive(
                    {
                        "popular": v["popularity"],
                        "content": v["content"],
                        "itemcf": v["itemcf"],
                        "generated": v["generative_prefix"],
                    },
                    r["history_length"],
                    {"threshold": threshold, "weight": weight},
                )
                values.append(recall_metrics(pred, r["positives"])["recall50"])
            trials.append(
                dict(
                    threshold=threshold, weight=weight, recall50=float(np.mean(values))
                )
            )
    best = max(trials, key=lambda x: x["recall50"])
    result = {
        **best,
        "source": str(source),
        "selection": "internal-validation recall50 only",
        "policy_version": POLICY_VERSION,
        "validation_requests": len(rows),
        "provenance": metrics["provenance"],
        "trials": trials,
    }
    (path("models") / "fusion.json").write_text(json.dumps(result, indent=2))
    print(result)
