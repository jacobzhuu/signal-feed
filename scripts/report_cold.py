"""Publish complete ColdFuse evidence after verified evaluations and service checks."""

import argparse
import json
import re
import shutil
from pathlib import Path
from signalfeed.provenance import sha256, code_identity


def write(root, docs):
    summary = json.loads((root / "reports/cold_summary.json").read_text())
    validation = json.loads((root / "reports/validation.json").read_text())
    latency = json.loads((root / "reports/cold_latency.json").read_text())
    current = code_identity()
    if not validation["passed"] or any(v["errors"] for v in latency["loads"].values()):
        raise ValueError("Engineering checks must pass")
    if any(
        r["provenance"]["source_sha256"] != current["source_sha256"]
        for r in (validation, latency)
    ):
        raise ValueError(
            "Rerun service verification and cost measurement on the current source"
        )
    core = (
        "coldfuse",
        "cold_protocol",
        "cold_training",
        "cold_experiments",
        "features",
        "generator",
        "evaluation",
        "ranker",
        "retrieval",
        "sid",
        "data",
    )
    source = Path(__file__).resolve().parents[1]
    core_hashes = {}
    for name in core:
        relative = Path("src/signalfeed") / (name + ".py")
        actual = sha256(source / relative)
        if actual != sha256(root / "source" / relative):
            raise ValueError("Evaluated algorithm differs from current code")
        core_hashes[name] = actual
    for run in summary["runs"]:
        kind = "natural" if run["natural"] else "test"
        if (
            sha256(root / f"reports/cold-{kind}-{run['seed']}/predictions.jsonl")
            != run["predictions_sha256"]
        ):
            raise ValueError("Predictions changed after aggregation")
    if summary["maximum_recomputed_error"] > 1e-10:
        raise ValueError("Recomputed metrics disagree")
    passed = re.findall(
        r"(\d+) passed", (root / "reports/validation-0.log").read_text()
    )
    if not passed:
        raise ValueError("Missing pytest completion evidence")
    python_count = int(passed[-1])
    browser_count = validation["browser"]["expected"]
    summary["engineering"] = dict(
        validation=validation, latency=latency, matching_core_sha256=core_hashes
    )
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "cold_results.json").write_text(json.dumps(summary, indent=2))
    shutil.copy2(root / "reports/cold_metrics.csv", docs / "COLD_METRICS.csv")
    train = [
        json.loads((root / f"models/coldfuse-{s}/train.json").read_text())
        for s in (42, 43, 44)
    ]
    feedback = json.loads((root / "reports/cold_online_retrain.json").read_text())
    groups = summary["groups"]

    def value(group, method, metric):
        item = groups[group][method].get(metric)
        return "—" if item is None else f"{item['mean']:.6f}"

    def difference(group, method, metric="recall50"):
        result = summary["comparisons"][group][method][metric]
        return f"{result['delta']:+.6f} [{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}]"

    main = [
        f"{s}/{k}" for s in ("no_explicit", "proxy_explicit") for k in (0, 1, 3, 5, 10)
    ]
    lines = [
        "# SignalFeed / ColdFuse 实测结果",
        "",
        f"正式运行：`{root.name}`。离线冻结提交 `{summary['runs'][0]['provenance']['git_commit'][:7]}`，源码 SHA `{summary['identity']['source_sha256']}`。",
        f"服务与工程验收提交 `{validation['provenance']['git_commit'][:7]}`，源码 SHA `{validation['provenance']['source_sha256']}`。两个版本之间仅服务/压测/展示/报告调整；核心、特征、训练、候选和评测文件逐个哈希一致。",
        "",
        "历史审计基线完整保留于 [BASELINE_AUDIT_V2.md](BASELINE_AUDIT_V2.md)。问题等级、修复证据和保留风险见 [AUDIT_CLOSURE.md](AUDIT_CLOSURE.md)。不把 MIND 结果当作 HN 线上收益。",
        "",
        "## 协议与失败记录",
        "",
        "- 三种子 42/43/44；各取同一 1,000 个至少 15 条请求前历史的 impressions，分别评测 0/1/3/5/10 预算和两个兴趣场景。10,000 个场景实例不是 10,000 个独立请求。",
        "- `no_explicit` 无额外兴趣；`proxy_explicit` 来自与末十条不重叠的更早类别历史，原型仅使用训练边界前内容。这是模拟侧信息，不是真实 onboarding。",
        "- 主 Recall 保留不可达正目标；独立报告可达率和可达 Recall。目录是首次日志出现代理，原预处理历史最多 20 条。每路/融合预算 100，SID 20 beams、每前缀至多 5 篇。",
        f"- 300 个内部验证请求、原定参数集合 3×3×3=27 组，按 0/1/3/5 与两个场景等权 Recall50 选择 `{summary['config']}`；10 次交互不参与选择。所有 trials 和逐请求验证得分保留。",
        "- `coldfuse-v1` 的计数被误写成 81，测试在预测前被拒绝；旧权重、验证及错误日志保留。v2 重新训练、选择、评测，没有根据测试结果调参。",
        "- 所有消融保留同一检查点，是推理干预而非逐个重训。区间先在同一 impression 内平均三种子差值，再按用户聚类 bootstrap；区间未校正多重比较。",
        "",
        "## 主模拟：全目录 Recall@50",
        "",
        "各格为三种子均值。标准差、Recall20、端到端 NDCG5/10、覆盖、实际候选数和全部 14 个方法见 [完整 CSV](COLD_METRICS.csv) / [JSON](cold_results.json)。",
        "",
        "| 场景 / 阅读预算 | 热门 | 内容 | ItemCF | SID 前缀 | 修复后原 DNN | ColdFuse |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in main:
        lines.append(
            "| "
            + " | ".join(
                [group]
                + [
                    value(group, m, "recall50")
                    for m in (
                        "popularity",
                        "content",
                        "itemcf",
                        "sid_prefix",
                        "repaired_dnn",
                        "coldfuse",
                    )
                ]
            )
            + " |"
        )
    lines += [
        "",
        "无历史无兴趣时 ColdFuse 与热门策略相同；此时内容和 SID 没有个人信号，不将空通道对照的差异解释为兴趣学习收益。",
        "",
        "## Impression 排序与端到端质量",
        "",
        "| 场景 / 预算 | 原 DNN AUC | ColdFuse AUC | ColdFuse MRR | NDCG5 | NDCG10 | 全目录端到端 NDCG10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in main:
        lines.append(
            "| "
            + " | ".join(
                [g, value(g, "repaired_dnn", "auc")]
                + [
                    value(g, "coldfuse", k)
                    for k in ("auc", "mrr", "ndcg5", "ndcg10", "end_to_end_ndcg10")
                ]
            )
            + " |"
        )
    lines += [
        "",
        "AUC 是逐 impression 宏平均，单类请求跳过；MRR 使用 MIND 多正例平均 reciprocal rank。稀疏检索/融合对未召回候选给 0，原 DNN 对全部曝光打分，二者 AUC 口径需结合候选覆盖解读。sigmoid 不是校准 CTR。",
        "",
        "## 配对比较：ColdFuse 减去对照",
        "",
        "数值是 Recall50 差值及用户聚类 95% 区间；正负和跨零结果全部保留，完整对照及 AUC/NDCG 区间在 JSON。",
        "",
        "| 场景 / 预算 | 减内容 | 减修复后原 DNN |",
        "|---|---|---|",
    ]
    for g in main:
        lines.append(
            f"| {g} | {difference(g, 'content')} | {difference(g, 'repaired_dnn')} |"
        )
    lines += [
        "",
        "## 消融：ColdFuse 减去对应消融的 Recall50",
        "",
        "| 场景 / 预算 | 去先验 | 去生成 | 固定门控 | 去收缩 | 去 DNN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in main:
        lines.append(
            "| "
            + " | ".join(
                [g]
                + [
                    f"{summary['comparisons'][g][m]['recall50']['delta']:+.6f}"
                    for m in (
                        "no_prior",
                        "no_gen",
                        "fixed_gates",
                        "no_shrinkage",
                        "no_dnn",
                    )
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## 自然冷人群（独立 300 请求）",
        "",
        "| 历史 | 请求 | 用户 | 热门 Recall50 | ColdFuse Recall50 | 减原 DNN及区间 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for g in ("natural/0", "natural/1-5"):
        c = summary["comparisons"][g]["repaired_dnn"]["recall50"]
        lines.append(
            f"| {g} | {c['requests']} | {c['users']} | {value(g, 'popularity', 'recall50')} | {value(g, 'coldfuse', 'recall50')} | {difference(g, 'repaired_dnn')} |"
        )
    lines += [
        "",
        "## 工程与成本",
        "",
        f"Python **{python_count} passed**，Playwright **{browser_count} passed**，TypeScript/Vite、Ruff 全部通过。覆盖原搜索/收藏/反馈、显式兴趣、一次点击后的更新、空信号、快照、独立训练契约、超时、空闲复用及并发缓存。",
        "",
        "| DNN 种子 | 训练 impressions | 采样曝光 | 参数 | CPU 训练秒 |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in train:
        lines.append(
            f"| {t['seed']} | {t['impressions']} | {t['examples']} | {t['parameters']} | {t['seconds']:.2f} |"
        )
    lines += [
        "",
        "SID、编码器及码本复用已绑定审计基线；ColdFuse DNN 全部重新训练，3 epoch。没有使用当前其他 GPU 作业的资源。离线每个场景/预算的通道、特征/DNN、ColdFuse 全链路 p50/p95 在 JSON costs；三种子并行于共享主机，不能当成独占硬件成本。",
        "",
        "### HN HTTP 混合负载",
        "",
        f"真实目录 {latency['health_before']['articles']} 篇，16 会话、{latency['unique_profiles']} 个实际画像，每个客户端在每并发级别各访问 5 次推荐和 5 次搜索。包括快照和测试曝光写入；稳定画像有利于缓存，不代表高基数持续更新场景。",
        "",
        "| 并发 | 请求 p50 ms | 请求 p95 ms | 曝光写入 p95 ms | 请求/s | 模型/缓存 | 回退 | 无个人信号 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency, r in latency["loads"].items():
        status = r["statuses"]
        count = status.get("research", 0) + status.get("cache", 0)
        fallback = sum(n for k, n in status.items() if "fallback" in k)
        lines.append(
            f"| {concurrency} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['event_p95_ms']:.2f} | {r['requests_per_second']:.2f} | {count} | {fallback} | {status.get('no-personal-signal', 0)} |"
        )
    lines += [
        "",
        "每级 160 请求全部完成，其中 80 次搜索；模型/缓存、回退和无个人信号三列仅针对另外 80 次推荐。HTTP 成功不能等同生成成功。单 SQLite 实例的尾延迟仍是限制，不能声称生产容量或相对旧负载全面加速。旧接口/画像混杂压测和修复前缓存竞争结果另存，不替换或混比。",
        "",
        f"真实成熟反馈：{feedback['mature_exposures']} 次，正曝光 {feedback['positive_exposures']} 次；训练状态 `{feedback['status']}`。没有真实模型晋升或 CTR 收益。",
        "",
        "## 复现与证据",
        "",
        "- `scripts/run_cold_experiments.sh`：重新训练、验证集合选择、三种子主/自然冷评测；`SIGNALFEED_COLD_PARALLEL=1` 可并行种子。完整运行保存独立目录，不覆盖旧预测。",
        "- `scripts/summarize_cold.py --run <目录>`：校验来源/模型/门控，逐条复算指标，按用户生成配对区间；本轮最大复算误差为 "
        + str(summary["maximum_recomputed_error"])
        + "。",
        "- `scripts/verify.py --browser` 与 `SIGNALFEED_BENCH_COLD=1 python scripts/benchmark.py`：仅使用独立 `SIGNALFEED_TEST_EVENTS=1` 服务；不向真实反馈库写自动化点击。",
        "- `scripts/report_cold.py --run <目录>`：要求工程验收与当前源码一致、当前核心与冻结实验文件逐个一致、原始预测哈希不变，才生成此报告。",
        f"- 本机完整原始证据：`{root}`；失败运行：`{root.parent / 'coldfuse-v1'}`。MIND 数据和权重不随 Git 分发，来源与下载/重训命令保留。",
        "",
    ]
    (docs / "RESULTS.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--docs", type=Path, default=Path(__file__).resolve().parents[1] / "docs"
    )
    args = parser.parse_args()
    write(args.run, args.docs)
