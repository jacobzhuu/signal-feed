"""Recompute raw predictions, paired user tests, cohorts and standalone figures."""

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from signalfeed.adaptation_experiments import paired_test, holm, algorithm_identity
from signalfeed.adaptation import GRID
from signalfeed.provenance import sha256

METHODS = (
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
MAIN = ("content_safe", "fixed_fusion", "coldfuse", "adaptation", "guarded")


def independent_metrics(pred, targets, labels, scores):
    if len(pred) != len(set(pred)) or len(pred) > 100:
        raise ValueError("Invalid candidate budget/duplicates")
    p = set(targets)
    result = {
        f"recall{k}": len(p.intersection(pred[:k])) / len(p) if p else 0.0
        for k in (20, 50)
    }
    y, score = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(score).all() or len(y) != len(score):
        raise ValueError("Invalid scores")
    positive = int(y.sum())
    order = np.argsort(-score, kind="stable")
    result["mrr"] = (
        float(np.sum(y[order] / np.arange(1, len(y) + 1)) / positive)
        if positive
        else None
    )
    ranks = rankdata(score)
    result["auc"] = (
        float(
            (ranks[y == 1].sum() - positive * (positive + 1) / 2)
            / (positive * (len(y) - positive))
        )
        if 0 < positive < len(y)
        else None
    )
    for k in (5, 10):
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(p))))
        result[f"end_to_end_ndcg{k}"] = (
            sum((i in p) / np.log2(j + 2) for j, i in enumerate(pred[:k])) / ideal
            if ideal
            else 0.0
        )
        denom = sum(1 / np.log2(i + 2) for i in range(min(k, positive)))
        result[f"ndcg{k}"] = (
            float(
                sum(y[order[j]] / np.log2(j + 2) for j in range(min(k, len(y)))) / denom
            )
            if denom
            else None
        )
    return result


def confidence(values, repeats=2000):
    x = np.asarray(values)
    rng = np.random.default_rng(20261029)
    boot = []
    for start in range(0, repeats, 100):
        boot.extend(
            x[rng.integers(len(x), size=(min(100, repeats - start), len(x)))].mean(1)
        )
    return np.quantile(boot, [0.025, 0.975]).tolist() if len(x) else [None, None]


def decision_audit(root, selection, guard):
    """Reconstruct all selection means and guard decisions from recorded trials."""
    trials, groups = [], defaultdict(list)
    sources, bindings = {}, {}
    file = root / "reports/select-42/predictions.jsonl"
    assert sha256(file) == selection["predictions_sha256"]
    with file.open() as stream:
        for line in stream:
            r = json.loads(line)
            assert len(r["trials"]) == len(GRID)
            groups[r["group"]].append(r["trials"])
            if r["cohort"] == "main" and r["k"] != 10:
                trials.append(r["trials"])
    means = np.mean(trials, axis=0)
    for i, trial in enumerate(selection["trials"]):
        assert abs(trial["recall50"] - means[i]) < 1e-12
    winner = min(
        range(len(GRID)),
        key=lambda i: (-means[i], GRID[i].side, GRID[i].multi, GRID[i].recent),
    )
    from dataclasses import asdict

    assert selection["config"] == asdict(GRID[winner])
    users = defaultdict(lambda: defaultdict(list))
    for split, seeds in (
        ("select", (42,)),
        ("guard", (42, 43, 44)),
        ("test", (42, 43, 44)),
    ):
        for seed in seeds:
            file = root / f"reports/{split}-{seed}/predictions.jsonl"
            meta = json.loads(file.with_name("metrics.json").read_text())
            assert meta["algorithm"] == algorithm_identity()
            assert meta["dataset_sha256"] == sha256(root / "dataset/manifest.json")
            assert meta["predictions_sha256"] == sha256(file)
            sources[f"{split}-{seed}"] = meta["predictions_sha256"]
            binding = dict(models=meta["models"], data=meta["data"])
            if seed in bindings:
                assert binding == bindings[seed]
            bindings[seed] = binding
            if split != "guard":
                continue
            assert guard["sources"][str(seed)] == meta["predictions_sha256"]
            assert meta["selection_sha256"] == sha256(root / "selection.json")
            with file.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    p = set(r["targets"])
                    # Match the original separate-ratio subtraction exactly.
                    delta = (
                        (
                            len(set(r["variants"]["adaptation"][:50]) & p) / len(p)
                            - len(set(r["variants"]["content_safe"][:50]) & p) / len(p)
                        )
                        if p
                        else 0.0
                    )
                    users[r["guard_group"]][r["user"]].append(delta)
    tests = {
        key: paired_test([np.mean(v) for v in users[key].values()])
        for key in guard["groups"]
    }
    holm(tests)
    for key, test in tests.items():
        expected = guard["groups"][key]
        assert test["n"] == expected["n"]
        assert np.allclose(test["ci"], expected["ci"], atol=1e-12, rtol=0)
        assert abs(test["mean"] - expected["mean"]) < 1e-12
        assert test["p"] == expected["p"] and test["p_holm"] == expected["p_holm"]
        assert (
            bool(test["n"] >= 100 and test["ci"][0] > 0 and test["p_holm"] <= 0.05)
            == expected["accepted"]
        )
    return dict(
        selection_recomputed=True,
        guard_recomputed=True,
        models_and_bound_data_unchanged_across_phases=True,
        sources=sources,
        selection_all_groups={
            g: np.mean(v, axis=0).tolist() for g, v in groups.items()
        },
    )


def report(root, output):
    output.mkdir(exist_ok=True, parents=True)
    dataset = root / "dataset"
    data = json.loads((dataset / "manifest.json").read_text())
    if data.get("protocol") != "adaptation-v2-canonical-identity":
        raise ValueError("Unreconciled cross-release identity")
    for name, meta in data["files"].items():
        assert sha256(dataset / name) == meta["sha256"], name
    selection = json.loads((root / "selection.json").read_text())
    guard = json.loads((root / "guard.json").read_text())
    assert selection["algorithm"] == guard["algorithm"] == algorithm_identity()
    assert guard["selection_sha256"] == sha256(root / "selection.json")
    decisions = decision_audit(root, selection, guard)
    cost = json.loads((root / "reports/inference_cost.json").read_text())
    assert cost["algorithm"] == algorithm_identity()
    assert cost["guard_sha256"] == sha256(root / "guard.json")
    assert cost["dataset_sha256"] == sha256(dataset / "manifest.json")
    cost_raw = root / "reports/inference_cost_predictions.jsonl"
    assert cost["raw_sha256"] == sha256(cost_raw)
    cost_predictions = {}
    with cost_raw.open() as stream:
        for line in stream:
            r = json.loads(line)
            key = (r["user"], r["group"], r["method"])
            assert key not in cost_predictions
            cost_predictions[key] = r
    parity = {
        method: dict(checked=0, full_order_differences=0, max_recall50_difference=0.0)
        for method in METHODS
    }
    users, sample = {}, {}
    excluded = set(json.loads((dataset / "excluded_large_users.json").read_text()))
    for split in ("select", "guard", "test"):
        users[split] = set()
        for cohort in ("main", "natural"):
            with (dataset / f"{split}-{cohort}.jsonl").open() as stream:
                for line in stream:
                    r = json.loads(line)
                    assert r["user"] not in users[split]
                    users[split].add(r["user"])
                    assert r["user"] not in excluded
                    if split == "test":
                        sample[cohort, r["id"], r["user"]] = r
    assert (
        not users["select"] & users["guard"]
        and not users["select"] & users["test"]
        and not users["guard"] & users["test"]
    )
    news = json.loads((dataset / "news.json").read_text())
    first = np.array([n["first_seen"] for n in news])
    index = {n["id"]: i for i, n in enumerate(news)}
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_seed = defaultdict(lambda: defaultdict(list))
    primary = defaultdict(lambda: defaultdict(list))
    cohort_values = defaultdict(lambda: defaultdict(list))
    diagnostic = defaultdict(lambda: defaultdict(list))
    costs, provenance, maximum_error = defaultdict(lambda: defaultdict(list)), {}, 0.0
    medtime = float(
        np.median([r["time"] for (c, _, _), r in sample.items() if c == "main"])
    )
    for seed in (42, 43, 44):
        file = root / f"reports/test-{seed}/predictions.jsonl"
        meta = json.loads(file.with_name("metrics.json").read_text())
        assert meta["algorithm"] == algorithm_identity()
        assert meta["predictions_sha256"] == sha256(file)
        assert meta["dataset_sha256"] == sha256(dataset / "manifest.json")
        assert meta["selection_sha256"] == sha256(root / "selection.json")
        assert meta["guard_sha256"] == sha256(root / "guard.json")
        provenance[str(seed)] = meta
        seen = set()
        with file.open() as stream:
            for line in stream:
                r = json.loads(line)
                row = sample[r["cohort"], r["id"], r["user"]]
                assert r["labels"] == row["labels"]
                assert r["items"] == [index[i] for i in row["items"]]
                assert set(r["targets"]) == {
                    i for i, y in zip(r["items"], r["labels"]) if y
                }
                key = (r["group"], r["user"])
                assert key not in seen
                seen.add(key)
                assert set(r["variants"]) == set(METHODS)
                g = r["group"]
                aliases = [g]
                if r["cohort"] == "natural":
                    aliases.append("natural/all")
                    if r["k"]:
                        aliases.append("natural/1-5")
                    kind = (
                        "natural_known" if r["known_large_train"] else "natural_unseen"
                    )
                    aliases.extend([f"{kind}/{r['k']}", f"{kind}/all"])
                expected = (
                    "adaptation"
                    if guard["groups"][r["guard_group"]]["accepted"]
                    else "content_safe"
                )
                assert r["variants"]["guarded"] == r["variants"][expected]
                p, h = set(r["targets"]), set(r["observed"])
                reachable = {i for i in p if first[i] < r["time"] and i not in h}
                assert reachable == set(r["reachable"])
                record_metrics = {}
                for method in METHODS:
                    pred = r["variants"][method]
                    measured = (
                        cost_predictions.get((r["user"], r["group"], method))
                        if seed == 42
                        else None
                    )
                    if measured:
                        check = parity[method]
                        check["checked"] += 1
                        check["full_order_differences"] += measured["ids"] != pred
                        difference = abs(
                            len(set(measured["ids"][:50]) & p) - len(set(pred[:50]) & p)
                        ) / max(1, len(p))
                        check["max_recall50_difference"] = max(
                            check["max_recall50_difference"], difference
                        )
                        if method != "coldfuse":
                            assert measured["ids"] == pred, (
                                method,
                                r["user"],
                                r["group"],
                            )
                    assert all(first[i] < r["time"] and i not in h for i in pred)
                    metrics = independent_metrics(
                        pred, r["targets"], r["labels"], r["scores"][method]
                    )
                    record_metrics[method] = metrics
                    for key, value in metrics.items():
                        stored = r["metrics"][method][key]
                        if value is None:
                            assert stored is None
                        else:
                            maximum_error = max(maximum_error, abs(value - stored))
                    metrics["reachable_recall50"] = (
                        len(set(pred[:50]) & reachable) / len(reachable)
                        if reachable
                        else None
                    )
                    metrics["reachable_rate"] = len(reachable) / len(p) if p else 0.0
                    metrics["candidates"] = len(pred)
                    for group in aliases:
                        for metric, value in metrics.items():
                            if value is not None:
                                values[group, method][metric][r["user"]].append(value)
                        per_seed[group, method][seed].append(metrics["recall50"])
                if r["cohort"] == "natural" or r["k"] != 10:
                    family = "natural" if r["cohort"] == "natural" else g.split("/")[0]
                    for method in ("adaptation", "guarded", "coldfuse"):
                        d = (
                            record_metrics[method]["recall50"]
                            - record_metrics["content_safe"]["recall50"]
                        )
                        primary[family, method][r["user"]].append(d)
                # Predefined slices, no use in model selection or guard decisions.
                slices = [
                    "known_train" if r["known_large_train"] else "unseen_train",
                    "first_request" if r["first_dev_request"] else "later_request",
                    "no_history"
                    if not r["observed"]
                    else "coherent"
                    if r["coherence"] >= 0.5
                    else "mixed",
                    "early_time" if r["time"] <= medtime else "late_time",
                ]
                for slice_name in slices:
                    for method in ("coldfuse", "adaptation", "guarded"):
                        for group in aliases:
                            delta = (
                                record_metrics[method]["recall50"]
                                - record_metrics["content_safe"]["recall50"]
                            )
                            cohort_values[group, slice_name, method][r["user"]].append(
                                delta
                            )
                for slice_name, positive in (
                    ("old_items", {i for i in p if i < data["old_items"]}),
                    ("new_items", set(r["new_targets"])),
                ):
                    if positive:
                        for method in ("coldfuse", "adaptation", "guarded"):
                            delta = (
                                len(set(r["variants"][method][:50]) & positive)
                                - len(
                                    set(r["variants"]["content_safe"][:50]) & positive
                                )
                            ) / len(positive)
                            cohort_values[g, slice_name, method][r["user"]].append(
                                delta
                            )
                channels = {
                    name: set(r["variants"][name])
                    for name in ("content", "itemcf", "sid_prefix")
                }
                union = set().union(*channels.values())
                full_union = union | set(r["variants"]["popularity"])
                diag = dict(
                    union100_recall=len(union & p) / max(1, len(p)),
                    fusion_loss=len((full_union - set(r["variants"]["coldfuse"])) & p)
                    / max(1, len(p)),
                    coherence=r["coherence"],
                    prior_agreement=r["prior_agreement"],
                    effective_history=len(h),
                )
                new_content = set(r["variants"]["interest_only"])
                side_unique = channels["sid_prefix"] - new_content - channels["itemcf"]
                diag["sid_exclusive_of_new_content_recall"] = len(
                    side_unique & p
                ) / max(1, len(p))
                diag["new_content_sid_jaccard"] = len(
                    new_content & channels["sid_prefix"]
                ) / max(1, len(new_content | channels["sid_prefix"]))
                diag["new_representation_recall50_gain"] = (
                    record_metrics["interest_only"]["recall50"]
                    - record_metrics["content_safe"]["recall50"]
                )
                diag["residual_fusion_recall50_gain"] = (
                    record_metrics["adaptation"]["recall50"]
                    - record_metrics["interest_only"]["recall50"]
                )
                diag["adaptation_pool_loss"] = len(
                    (
                        (new_content | channels["itemcf"] | channels["sid_prefix"])
                        - set(r["variants"]["adaptation"])
                    )
                    & p
                ) / max(1, len(p))
                for a, b in itertools.combinations(channels, 2):
                    diag[f"jaccard/{a}/{b}"] = len(channels[a] & channels[b]) / max(
                        1, len(channels[a] | channels[b])
                    )
                for a in channels:
                    others = set().union(*(v for b, v in channels.items() if b != a))
                    diag[f"exclusive_recall/{a}"] = len(
                        (channels[a] - others) & p
                    ) / max(1, len(p))
                    diag[f"size/{a}"] = len(channels[a])
                for group in aliases + [g + "|" + s for s in slices]:
                    for name, value in diag.items():
                        diagnostic[group][name].append(value)
                for group in aliases:
                    for name, value in r["costs"].items():
                        costs[group][name].append(value)
        expected_count = sum(10 if c == "main" else 1 for c, _, _ in sample)
        assert len(seen) == expected_count
        print(
            dict(verified_seed=seed, instances=len(seen), max_error=maximum_error),
            flush=True,
        )
    assert maximum_error < 1e-12
    assert sum(v["checked"] for v in parity.values()) == len(cost_predictions)
    assert all(v["max_recall50_difference"] == 0 for v in parity.values())
    summary = {}
    csv_rows = []
    for (group, method), metrics in sorted(values.items()):
        means = {
            metric: float(np.mean([np.mean(v) for v in users.values()]))
            for metric, users in metrics.items()
        }
        seeds = {str(s): float(np.mean(v)) for s, v in per_seed[group, method].items()}
        recall = [np.mean(v) for v in metrics["recall50"].values()]
        result = dict(
            **means, users=len(recall), recall50_ci=confidence(recall), seeds=seeds
        )
        summary.setdefault(group, {})[method] = result
        csv_rows.append(
            dict(
                group=group,
                method=method,
                **means,
                users=len(recall),
                recall50_low=result["recall50_ci"][0],
                recall50_high=result["recall50_ci"][1],
                **{f"seed{s}": v for s, v in seeds.items()},
            )
        )
    tests = {
        "/".join(key): paired_test([np.mean(v) for v in users.values()])
        for key, users in primary.items()
    }
    holm(tests)
    comparisons = {}
    for group in summary:
        for method, base in (
            ("coldfuse", "content_safe"),
            ("adaptation", "content_safe"),
            ("guarded", "content_safe"),
            ("adaptation", "no_multi"),
            ("adaptation", "no_reliability"),
            ("adaptation", "no_side"),
            ("adaptation", "no_gen"),
            ("adaptation", "fixed_side"),
        ):
            left, right = (
                values[group, method]["recall50"],
                values[group, base]["recall50"],
            )
            delta = [float(np.mean(left[u]) - np.mean(right[u])) for u in sorted(left)]
            comparisons[group + "/" + method + "-" + base] = dict(
                n=len(delta), mean=float(np.mean(delta)), ci=confidence(delta)
            )
    slices = {
        "/".join(key): dict(
            n=len(users),
            mean=float(np.mean([np.mean(v) for v in users.values()])),
            ci=confidence([np.mean(v) for v in users.values()]),
        )
        for key, users in sorted(cohort_values.items())
    }
    result = dict(
        protocol=data["protocol"],
        dataset=data,
        selection=selection,
        guard=guard,
        decision_audit=decisions,
        standalone_inference_cost=cost,
        standalone_prediction_parity=parity,
        maximum_recomputed_error=maximum_error,
        provenance=provenance,
        groups=summary,
        sample_counts={
            cohort: dict(
                users=sum(c == cohort for c, _, _ in sample),
                unseen_large_train=sum(
                    not row["known_large_train"]
                    for (c, _, _), row in sample.items()
                    if c == cohort
                ),
                first_dev_request=sum(
                    row["first_dev_request"]
                    for (c, _, _), row in sample.items()
                    if c == cohort
                ),
            )
            for cohort in ("main", "natural")
        },
        primary_tests=tests,
        exploratory_comparisons=comparisons,
        exploratory_slices=slices,
        complementarity={
            g: dict(
                users=len(d["coherence"]) // 3,
                **{k: float(np.mean(v)) for k, v in d.items()},
            )
            for g, d in diagnostic.items()
        },
        shared_research_costs={
            g: {
                k: dict(
                    p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95))
                )
                for k, v in d.items()
            }
            for g, d in costs.items()
        },
        notes=[
            "User-disjoint same-period evaluation, not a future temporal holdout or HN signup test",
            "Sparse fused AUC and dense content/DNN AUC have different support; primary endpoint is full-catalog Recall50",
            "Three model seeds are averaged within user, not counted as independent users",
            "Primary nine tests Holm-corrected; per-budget and cohort intervals exploratory",
            "Union100 is an oracle coverage diagnostic with up to 300 candidates, not a fair-budget recommender",
            "new_items means added to the frozen small catalog, not a verified publication-date cohort",
        ],
    )
    (output / "adaptation_results.json").write_text(json.dumps(result, indent=2))
    with (output / "ADAPTATION_METRICS.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    plots(summary, output)
    markdown(result, output)


def plots(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(
        content_safe="#1b7a57",
        fixed_fusion="#8b8b8b",
        coldfuse="#b94f49",
        adaptation="#3157b7",
        guarded="#8b52a1",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, scenario in zip(axes, ("no_explicit", "proxy_explicit")):
        for method in MAIN:
            rows = [summary[f"{scenario}/{k}"][method] for k in (0, 1, 3, 5, 10)]
            mean = np.array([r["recall50"] for r in rows])
            ci = np.array([r["recall50_ci"] for r in rows])
            ax.plot(
                [0, 1, 3, 5, 10],
                mean,
                "o--" if method == "guarded" else "o-",
                label=method,
                color=colors[method],
                linewidth=1.5,
            )
            ax.fill_between(
                [0, 1, 3, 5, 10], ci[:, 0], ci[:, 1], alpha=0.07, color=colors[method]
            )
        ax.set_title(scenario + " (same 1,200 users)")
        ax.set_xlabel("Observed history budget (truncated simulation)")
        ax.set_xticks([0, 1, 3, 5, 10])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Full-catalog Recall@50")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Independent users: mean of three frozen seeds; user-bootstrap 95% bands"
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"adaptation_learning_curves.{ext}", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = [k for k in range(6) if f"natural/{k}" in summary]
    for method in MAIN:
        rows = [summary[f"natural/{k}"][method] for k in ks]
        y = np.array([r["recall50"] for r in rows])
        ci = np.array([r["recall50_ci"] for r in rows])
        ax.plot(
            ks,
            y,
            "o--" if method == "guarded" else "o-",
            label=method,
            color=colors[method],
        )
        ax.fill_between(ks, ci[:, 0], ci[:, 1], alpha=0.07, color=colors[method])
    ax.set_xticks(
        ks, [f"{k}\n(n={summary[f'natural/{k}']['content_safe']['users']})" for k in ks]
    )
    ax.set_xlabel("Actual logged history length (different users at each point)")
    ax.set_ylabel("Full-catalog Recall@50")
    ax.set_title(
        "Natural sparse-history cohort; not a signup or longitudinal learning curve"
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"adaptation_natural_cohorts.{ext}", dpi=160)
    plt.close(fig)


def markdown(result, output):
    lines = [
        "# 冷启动兴趣适应：独立用户实验",
        "",
        "本文件从完整逐请求预测生成。解释与部署决定另见 ADAPTATION_ANALYSIS.md；协议见 ADAPTATION_PROTOCOL.md。",
        "",
        f"原始指标独立复算最大误差：{result['maximum_recomputed_error']}。参数选择：`{result['selection']['config']}`。",
        "",
        "## 完整学习曲线",
        "",
        "![模拟冷启动](adaptation_learning_curves.png)",
        "",
        "![自然少历史](adaptation_natural_cohorts.png)",
        "",
        "模拟是同一批用户截断；自然曲线各点是不同用户，不是上线连续学习。真实HN新注册用户没有质量样本。",
        "",
        "| 场景/历史 | 用户 | 内容回退 | 固定融合 | 旧ColdFuse | 新适应 | 准入回退后 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, methods in result["groups"].items():
        lines.append(
            "| "
            + group
            + " | "
            + str(methods["content_safe"]["users"])
            + " | "
            + " | ".join(f"{methods[m]['recall50']:.6f}" for m in MAIN)
            + " |"
        )
    lines += [
        "",
        "## 九项主要配对检验",
        "",
        "差值均为对应方法减content_safe，模拟总体等权0/1/3/5；自然按用户等权。先平均同用户种子和预算，再bootstrap/双侧符号翻转；Holm校正九项比较。",
        "",
        "| 比较 | 用户 | Recall50差值 | 95%区间 | 原始p | Holm p |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for key, test in result["primary_tests"].items():
        lines.append(
            f"| {key} | {test['n']} | {test['mean']:+.6f} | [{test['ci'][0]:+.6f}, {test['ci'][1]:+.6f}] | {test['p']:.6f} | {test['p_holm']:.6f} |"
        )
    lines += [
        "",
        "## 独立准入结果",
        "",
        "只使用guard用户决定启用；最终test没有选择任何阈值。域限定MIND，不能直接发布为HN获益策略。",
        "",
        "| 有效信号组 | 用户 | 验证差值 | Holm p | 启用新适应 |",
        "|---|---:|---:|---:|---|",
    ]
    for key, test in result["guard"]["groups"].items():
        lines.append(
            f"| {key} | {test['n']} | {test['mean']:+.6f} | {test['p_holm']:.6f} | {test['accepted']} |"
        )
    cost = result["standalone_inference_cost"]
    lines += [
        "",
        "## 独立无结果缓存的推理成本",
        "",
        f"目录{cost['catalog_items']:,}篇；模型/数据/索引初始化{cost['model_and_data_load_seconds']:.2f}秒，随后权重和索引驻留。固定前100模拟及前100自然用户，方法轮转顺序，Torch 2线程、BLAS 1线程，共享CPU主机。下表各格为p50/p95毫秒；全部15方法的原始调用与成本在JSON。",
        "",
        "| 场景/历史 | 每方法请求 | 内容回退 | 固定融合 | 旧ColdFuse | 新适应 | 准入回退后 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, methods in cost["groups"].items():
        lines.append(
            "| "
            + group
            + " | "
            + str(methods["content_safe"]["n"])
            + " | "
            + " | ".join(
                f"{methods[m]['p50_ms']:.2f}/{methods[m]['p95_ms']:.2f}" for m in MAIN
            )
            + " |"
        )
    lines += [
        "",
        f"新适应实际生成调用{cost['generation_calls']['adaptation']}次，准入回退后{cost['generation_calls']['guarded']}次。此处不包含HTTP、数据库写入或文本编码，不代表HN线上端到端容量。进程峰值RSS {cost['peak_process_rss_kib'] / 1024:.1f} MiB包含全部基线，不能当作适应模块独占内存。",
        "",
    ]
    lines += [
        "",
        "完整15个方法、所有曝光/召回指标、三种子数值、消融差值、预定义分群、候选互补和研究成本见 [JSON](adaptation_results.json) / [CSV](ADAPTATION_METRICS.csv)。曝光稀疏融合分数与密集DNN/内容分数的AUC支持不同，不作主要收益口径。",
        "",
        "所有目标保留分母。全目录统一100候选；候选联合覆盖只是多路池的上限诊断。新增目录、用户和可用性信息与首轮不同，只比较本轮配对方法，不能将前后两个数据集直接相减。",
    ]
    (output / "ADAPTATION_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report(args.run, args.output)
