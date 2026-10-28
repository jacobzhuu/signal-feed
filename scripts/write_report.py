"""Render only an explicitly selected, provenance-checked run."""

import argparse
import json
from pathlib import Path
from signalfeed.provenance import sha256


def render(root, output):
    summary = json.loads((root / "reports/summary.json").read_text())
    if summary["missing_seeds"]:
        raise ValueError("Incomplete seeds: " + str(summary["missing_seeds"]))
    for name, expected in summary["input_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError("Changed predictions: " + name)
    validation = json.loads((root / "reports/validation.json").read_text())
    latency = json.loads((root / "reports/latency.json").read_text())
    for report in [validation, latency]:
        if (
            report["run_id"] != root.name
            or report["provenance"]["source_sha256"]
            != summary["provenance"]["source_sha256"]
        ):
            raise ValueError("Mixed run/source evidence")
    if not validation["passed"]:
        raise ValueError("Validation failed")
    lines = [
        "# SignalFeed 实测结果",
        "",
        f"运行：`{root.name}`；源码提交 `{summary['provenance']['git_commit']}`；源码哈希 `{summary['provenance']['source_sha256']}`。",
        "",
        "MIND 为历史日志实验；HN 没有真实成熟反馈时不宣称线上收益。旧审计修复记录保留于 REVIEW_FIXES.md。",
        "",
        "## 排序与全目录召回",
        "",
        "| 方法 | AUC | NDCG@10 | Recall@50 |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(set(summary["ranking"]) | set(summary["retrieval"])):
        values = []
        for section, key in [
            ("ranking", "auc"),
            ("ranking", "ndcg10"),
            ("retrieval", "recall50"),
        ]:
            m = summary[section].get(name, {}).get(key)
            values.append(
                f"{m['mean']:.6f} ± {m['std']:.6f} (n={m['n']})" if m else "—"
            )
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 受控消融与不确定性",
        "",
        "去生成消融保持其余通道权重、预算及 DNN 不变。以下只为 seed 42，按用户聚类 bootstrap。",
        "",
        "| 比较 | 差值 | 95% 区间 |",
        "|---|---:|---|",
    ]
    for name, m in summary["paired_bootstrap"].items():
        lines.append(f"| {name} | {m['mean']:.6f} | {m['ci95']} |")
    lines += [
        "",
        "## 冷人群证据",
        "",
        "| 分组 | 请求数 | 用户数 | 完整模型－去生成 Recall@50 | 95% 区间 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, m in summary["cold_bootstrap"].items():
        lines.append(
            f"| {name} | {m['impressions']} | {m['users']} | {m['mean']} | {m['ci95']} |"
        )
    lines += [
        "",
        "新文章以首次日志出现时间定义，不是真实发布时间；主 Recall 保留全部正目标。可达目标 Recall、可达率及各分组 AUC/NDCG 见 results.json。样本不足和跨零区间不得解释为确定收益。",
        "",
        "## 工程验证与成本",
        "",
        f"验证结果：`{json.dumps(validation, ensure_ascii=False)}`",
        "",
        f"隔离服务成本：`{json.dumps(latency, ensure_ascii=False)}`",
        "",
        "HTTP 基准会写请求/压缩快照；反馈负载仅允许隔离测试服务。共享硬件数据不作为生产容量承诺。",
        "",
        "## 可复现性",
        "",
        "运行 scripts/run_experiments.sh 后，使用 scripts/verify.py 和 scripts/benchmark.py 写入同一独立 SIGNALFEED_HOME，再运行 `python scripts/write_report.py --run <目录>`。报告拒绝混合源码、运行或被修改的预测。",
    ]
    output.write_text("\n".join(lines) + "\n")
    output.with_name("results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/RESULTS.md"))
    args = parser.parse_args()
    render(args.run.resolve(), args.output)
