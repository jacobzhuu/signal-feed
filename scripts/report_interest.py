"""Recompute metrics, audit independence and preserve every negative result."""

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from signalfeed.interest_experiments import (
    records,
    read_rows,
    admission,
    identity,
    write_json,
    load_catalog,
    inputs,
)
from signalfeed.adaptation_experiments import paired_test
from signalfeed.stream_interest import GRID, InterestState
from signalfeed.provenance import sha256
from signalfeed.evaluation import average


def independent():
    spec = importlib.util.spec_from_file_location(
        "previous_metric_audit", Path(__file__).with_name("report_adaptation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.independent_metrics


def groups(row):
    band = row["band"]
    return [
        band,
        f"k/{row['k']}",
        f"{band}/{row['period']}",
        f"{band}/train_{'known' if row['known_large_train'] else 'unseen'}",
        f"{band}/dev_{'first' if row['first_dev_request'] else 'later'}",
        f"{band}/unique_{row['unique_history']}",
        f"{band}/clusters_{row['clusters']}",
    ]


def delta(rows, left, right="content_safe", metric="recall50"):
    return [r["metrics"][left][metric] - r["metrics"][right][metric] for r in rows]


def audit_previous(files):
    """Ingest status is refreshed by the unchanged live site, not experiment data.

    Preserve its original hash and disclose current hash; all research evidence
    remains fail-closed. This exception does not accept arbitrary JSON files.
    """
    operations = {}
    verified = 0
    for file, checksum in files.items():
        path = Path(file)
        current = sha256(path)
        if path.name == "hn_ingest.json":
            status = json.loads(path.read_text())
            if (
                set(status) != {"source", "count", "errors", "seconds"}
                or status["source"] != "https://hacker-news.firebaseio.com/v0"
            ):
                raise ValueError("Unexpected ingestion status schema")
            operations[file] = dict(
                frozen_sha256=checksum,
                current_sha256=current,
                changed=checksum != current,
                reason="Periodic website ingestion status, excluded from research-result immutability",
            )
        elif current != checksum:
            raise ValueError("Old evidence changed: " + file)
        else:
            verified += 1
    return dict(verified_research_files=verified, operational_status=operations)


def report(run, out):
    out.mkdir(parents=True, exist_ok=True)
    output = out / "interest_v3_results.json"
    if output.exists():
        raise FileExistsError("Report already exists")
    catalog = load_catalog(run)
    frozen = json.loads((run / "frozen_previous.json").read_text())
    baseline_frozen = json.loads((run / "frozen_baseline_results.json").read_text())
    frozen["files"].update(baseline_frozen["files"])
    old_audit = audit_previous(frozen["files"])
    excluded = set(json.loads((run / "dataset/excluded_users.json").read_text()))
    all_users, split_rows, audit = set(), {}, {}
    verify = independent()
    max_error = 0.0
    for split in ("select", "guard", "test"):
        rows = records(run, split)
        metadata = json.loads((run / f"reports/{split}/metrics.json").read_text())
        if split != "select" and metadata["selection_sha256"] != sha256(
            run / "selection.json"
        ):
            raise ValueError("Prediction/selection binding changed")
        if split == "test" and metadata["guard_sha256"] != sha256(run / "guard.json"):
            raise ValueError("Prediction/guard binding changed")
        source = {r["user"]: r for r in read_rows(run, split)}
        users = {r["user"] for r in rows}
        if users != set(source) or users & excluded or users & all_users:
            raise ValueError("Independent sample audit failed")
        all_users |= users
        changed, duplicates, unreachable, no_targets = 0, 0, 0, 0
        for r in rows:
            src = source[r["user"]]
            h, items, targets, eligible = inputs(catalog, src)
            if (
                r["history"] != h
                or r["items"] != items
                or r["targets"] != targets
                or r["labels"] != src["labels"]
            ):
                raise ValueError("Prediction/source mismatch")
            if len(h) != r["k"] or len(set(h)) != r["unique_history"]:
                raise ValueError("Not natural history")
            duplicates += len(h) != len(set(h))
            no_targets += not bool(targets)
            unreachable += len(set(targets) - set(r["reachable"]))
            if r["reachable"] != [i for i in targets if eligible[i]]:
                raise ValueError("Availability mismatch")
            unique = list(dict.fromkeys(reversed(h)))[::-1][-20:]
            cf_scores = np.asarray(catalog.cf[unique][:, items].sum(0)).ravel()
            np.testing.assert_array_equal(cf_scores, r["scores"]["itemcf"])
            for method, pred in r["variants"].items():
                if (
                    len(pred) > 100
                    or len(set(pred)) != len(pred)
                    or not all(eligible[i] for i in pred)
                ):
                    raise ValueError("Invalid candidate budget/filter")
                recomputed = verify(pred, targets, r["labels"], r["scores"][method])
                for key, value in recomputed.items():
                    observed = r["metrics"][method][key]
                    if value is None or observed is None:
                        if value != observed:
                            raise ValueError("Null metric mismatch")
                    else:
                        max_error = max(max_error, abs(value - observed))
            if r["variants"]["no_multi"] != r["variants"]["content_safe"]:
                raise ValueError("No-multi must equal content")
            changed += r["variants"]["interest"] != r["variants"]["content_safe"]
            if r["trials"]:
                for trial in r["trials"]:
                    recomputed = verify(
                        trial["ids"], targets, r["labels"], trial["scores"]
                    )
                    for key, value in recomputed.items():
                        if value is not None:
                            max_error = max(
                                max_error, abs(value - trial["metrics"][key])
                            )
        split_rows[split] = rows
        audit[split] = dict(
            users=len(rows),
            duplicate_history_users=duplicates,
            changed_lists=changed,
            unreachable_targets=unreachable,
            targets=sum(len(set(r["targets"])) for r in rows),
            no_targets=no_targets,
            new_targets=sum(len(r["new_targets"]) for r in rows),
            train_unseen=sum(not r["known_large_train"] for r in rows),
            independent_history=dict(Counter(r["unique_history"] for r in rows)),
        )
    if max_error > 1e-12:
        raise ValueError("Metric verification failed")
    selection = json.loads((run / "selection.json").read_text())
    primary = [r for r in split_rows["select"] if r["band"] == "1-5"]
    means = [
        float(np.mean([r["trials"][i]["metrics"]["recall50"] for r in primary]))
        for i in range(len(GRID))
    ]
    best = min(range(len(GRID)), key=lambda i: (-means[i], GRID[i].alpha, -GRID[i].tau))
    if selection["config"] != dict(alpha=GRID[best].alpha, tau=GRID[best].tau):
        raise ValueError("Selection differs from registered objective")
    np.testing.assert_allclose(
        means, [v["recall50"] for v in selection["trials"]], atol=1e-15
    )
    guard = json.loads((run / "guard.json").read_text())
    if guard["groups"] != admission(split_rows["guard"]):
        raise ValueError("Guard decision mismatch")
    rows = split_rows["test"]
    primary_tests = admission(rows)
    eligible_fusion = [
        b
        for b in ("1-5", "10")
        if guard["groups"][b]["accepted"] and primary_tests[b]["accepted"]
    ]
    for r in rows:
        expected = (
            "interest" if guard["groups"][r["band"]]["accepted"] else "content_safe"
        )
        if r["variants"]["guarded"] != r["variants"][expected]:
            raise ValueError("Fallback changed")
    grouped = defaultdict(list)
    for r in rows:
        for g in groups(r):
            grouped[g].append(r)
    summaries = {
        g: dict(
            n=len(rs),
            methods={
                m: average([r["metrics"][m] for r in rs]) for m in rows[0]["variants"]
            },
        )
        for g, rs in grouped.items()
    }
    contrasts = {}
    for band in ("1-5", "10"):
        rs = grouped[band]
        for left, right in (
            ("interest", "content_safe"),
            ("old_interest", "content_safe"),
            ("interest", "old_interest"),
            ("interest", "no_reliability"),
            ("interest", "no_recency"),
            ("no_reliability", "content_safe"),
            ("no_recency", "content_safe"),
        ):
            contrasts[f"{band}/{left}-{right}"] = paired_test(delta(rs, left, right))
    for group in [f"k/{k}" for k in (1, 2, 3, 4, 5, 10)] + [
        f"{b}/{p}" for b in ("1-5", "10") for p in ("early", "late")
    ]:
        contrasts[group + "/interest-content_safe"] = paired_test(
            delta(grouped[group], "interest")
        )
    cost = json.loads((run / "cost/summary.json").read_text())
    if (
        cost["algorithm"] != identity()
        or cost["predictions_sha256"] != sha256(run / "reports/test/predictions.jsonl")
        or cost["raw_sha256"] != sha256(run / "cost/calls.jsonl")
    ):
        raise ValueError("Cost binding changed")
    # Full real-history incremental audit on all final users, no additional labels.
    incremental_max = 0.0
    for r in rows:
        online = InterestState(catalog.x)
        for j, item in enumerate(r["history"]):
            online.update(item)
            batch = InterestState.from_history(catalog.x, r["history"][: j + 1])
            incremental_max = max(
                incremental_max, float(np.max(np.abs(online.sums - batch.sums)))
            )
    if incremental_max > 1e-12:
        raise ValueError("Incremental state differs")
    result = dict(
        version="interest-v3",
        algorithm=identity(),
        selection=selection,
        dataset=catalog.manifest,
        guard=guard,
        primary_tests=primary_tests,
        fusion_research_eligible_groups=eligible_fusion,
        deployment="unchanged content-v1; no automatic MIND-to-HN promotion",
        stop_reason=None
        if eligible_fusion
        else "No group passed both independent admission and final replication; stop parameter search and generative fusion.",
        audits=audit,
        old_files_verified=old_audit["verified_research_files"],
        old_evidence_audit=old_audit,
        metric_max_error=max_error,
        incremental_max_error=incremental_max,
        summaries=summaries,
        exploratory_contrasts=contrasts,
        cost=cost,
        evidence={
            str(p.relative_to(run)): sha256(p)
            for p in [
                run / "frozen_previous.json",
                run / "frozen_baseline_results.json",
                run / "dataset/manifest.json",
                run / "selection.json",
                run / "guard.json",
                run / "cost/summary.json",
            ]
            + [
                run / f"reports/{s}/{f}"
                for s in ("select", "guard", "test")
                for f in ("predictions.jsonl", "metrics.json")
            ]
        },
        report_source_sha256=sha256(Path(__file__)),
    )
    write_json(output, result)
    with (out / "INTEREST_V3_METRICS.csv").open("x", newline="") as stream:
        metric_names = list(rows[0]["metrics"]["interest"])
        writer = csv.DictWriter(
            stream, fieldnames=["group", "n", "method"] + metric_names
        )
        writer.writeheader()
        for g, group in sorted(summaries.items()):
            for method, metric in group["methods"].items():
                writer.writerow(dict(group=g, n=group["n"], method=method, **metric))
    lines = [
        "# 自然冷启动低成本兴趣表示：独立结果",
        "",
        f"基线 `3426a00`；核心冻结 `5e6968e`。选择参数 `{selection['config']}`。",
        "",
        "**部署决定：保持内容基线与现有网站策略，不发布新表示。**"
        if not eligible_fusion
        else "**仅支持继续独立研究，尚不授权HN发布。**",
        "",
        "三阶段用户互斥，并排除旧两轮全部样本与全部small对应用户。按行为长度与早晚时段分层取样；因此总体为本次分层样本均值，不是自然流量分布的CTR或总体收益。全部数据来自2019-11-15，不证明跨日期稳定。",
        "",
        "| 人群 | 阶段 | n | Recall50差值 | 95%区间 | Holm p | 通过 |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for label, tests in (("独立准入", guard["groups"]), ("最终评估", primary_tests)):
        for b, t in tests.items():
            lines.append(
                f"| {b} | {label} | {t['n']} | {t['mean']:+.6f} | [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}] | {t['p_holm']:.6f} | {t['accepted']} |"
            )
    lines += [
        "",
        "两项预注册双侧检验分别针对自然1–5、自然10；用户bootstrap/符号翻转各10,000次。准入还要求早晚各≥100且均值非负、总体NDCG10不降。95%区间为未校正区间，Holm p为两项校正。",
        "",
        "## 关键基线与消融",
        "",
        "| 人群 | n | 内容 | 旧纯兴趣 | 新纯兴趣 | 去可信度 | 去原型衰减 | ItemCF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in (
        "1-5",
        "10",
        "k/1",
        "k/2",
        "k/3",
        "k/4",
        "k/5",
        "1-5/early",
        "1-5/late",
        "10/early",
        "10/late",
    ):
        v = summaries[g]
        vals = " | ".join(
            f"{v['methods'][m]['recall50']:.6f}"
            for m in (
                "content_safe",
                "old_interest",
                "interest",
                "no_reliability",
                "no_recency",
                "itemcf",
            )
        )
        lines.append(f"| {g} | {v['n']} | {vals} |")
    lines += [
        "",
        "去多兴趣与内容逐用户一致。消融、单个行为长度及时段均为探索性，完整用户配对区间在JSON；不据这些结果重新选参。曝光AUC与全目录召回的候选支持不同。",
        "",
        "| 人群 | 内容端到端NDCG10 | 新方法端到端NDCG10 |",
        "|---|---:|---:|",
    ]
    for g in ("1-5", "10"):
        m = summaries[g]["methods"]
        lines.append(
            f"| {g} | {m['content_safe']['end_to_end_ndcg10']:.6f} | {m['interest']['end_to_end_ndcg10']:.6f} |"
        )
    lines += [
        "",
        "## 实际成本",
        "",
        f"目录{cost['catalog_shape'][0]:,}×{cost['catalog_shape'][1]}，{cost['cpu']}，BLAS单线程。{cost['users']}个独立请求，每方法无结果缓存、轮转顺序，全部{cost['calls']}份候选与质量预测完全一致。以下p50/p95为毫秒。",
        "",
        "| 原始行为数 | 内容 | 旧纯兴趣 | 新纯兴趣 | 追加更新（不含检索） | 批量重建（不含检索） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in ("1", "2", "3", "4", "5", "10"):
        vals = []
        for m in ("content_safe", "old_interest", "interest"):
            c = cost["methods"][m][k]
            vals.append(f"{c['p50_ms']:.3f}/{c['p95_ms']:.3f}")
        for kind in ("update", "batch"):
            c = cost["incremental"][k][kind]
            vals.append(f"{c['p50_ms']:.3f}/{c['p95_ms']:.3f}")
        lines.append("| " + k + " | " + " | ".join(vals) + " |")
    lines += [
        "",
        f"初始化{cost['initialization_seconds']:.2f}秒；进程峰值RSS {cost['peak_rss_mib']:.1f} MiB（含所有基线、CF和质量预测，非模块独占）；每用户数值状态{cost['state_numeric_bytes']:,}字节（不含Python对象和共享目录）。成本不含HTTP、数据库与编码，不能当作线上SLA。",
        "",
        "## 复核与边界",
        "",
        f"独立指标重算最大误差 {max_error:.3g}；全部最终用户逐前缀增量状态最大误差 {incremental_max:.3g}；旧研究证据 {old_audit['verified_research_files']} 个文件哈希未变。另有网站定时更新的hn_ingest运行状态，原/现哈希及首次报告中止记录均保留，不属于实验结果。自然原始长度和去重长度均保留；ItemCF候选/曝光统一去重，不将口径修正称为算法收益。",
        "",
        "独立实现贡献是有上限的兴趣原型、扣除最强单条行为的证据收缩、精确线性中心充分统计与重复/窗口过期重建，以及一次目录投影。它们是基于项目诊断的工程研究组合，不声称提出全新聚类或收缩理论，也没有新大模型。低成本成立不等于推荐质量改善。",
        "",
        "未同时通过准入与最终复制时，停止本轮参数搜索与生成配额研究；保留所有负结果。下一步应获取未使用的多日期自然用户数据及真实成熟反馈，先解决时间泛化和标签稀疏，冻结新协议后再研究。不能继续用本轮test调参。",
        "",
        "完整来源、全部四组选参、全部方法/分层/消融/原始成本在 [JSON](interest_v3_results.json)、[CSV](INTEREST_V3_METRICS.csv)；协议见 [INTEREST_V3_PROTOCOL.md](INTEREST_V3_PROTOCOL.md)。工程验收与复现命令另见 INTEREST_V3_ANALYSIS.md。",
    ]
    with (out / "INTEREST_V3_RESULTS.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                primary=primary_tests,
                fusion_eligible=eligible_fusion,
                metric_error=max_error,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report(a.run, a.output)
