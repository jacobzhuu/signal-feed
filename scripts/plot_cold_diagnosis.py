"""Visualize exposed-baseline bottlenecks without selecting new parameters."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source, output):
    groups = json.loads(source.read_text())["groups"]
    budgets = [1, 3, 5, 10]
    rows = [groups[f"no_explicit/{k}"] for k in budgets]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, color in (
        ("popular", "#8b8b8b"),
        ("content", "#1b7a57"),
        ("generated", "#3157b7"),
    ):
        axes[0].plot(
            budgets, [r[f"gate/{name}"] for r in rows], "o-", color=color, label=name
        )
    axes[0].set_ylabel("Mean unnormalized RRF gate")
    axes[0].set_title("Popularity still outweighs content")
    for name, color in (
        ("content", "#1b7a57"),
        ("itemcf", "#b94f49"),
        ("sid_prefix", "#3157b7"),
    ):
        axes[1].plot(
            budgets,
            [r[f"survival100/{name}"] for r in rows],
            "o-",
            color=color,
            label=name,
        )
    axes[1].set_ylabel("Fraction retained in fused top 100")
    axes[1].set_title("Most SID candidates are discarded")
    for key, label, color in (
        ("union_with_pop100_recall", "4-channel union (upper bound)", "#8b8b8b"),
        ("fusion100_recall", "ColdFuse top 100", "#3157b7"),
        ("coldfuse/recall50", "ColdFuse top 50", "#b94f49"),
    ):
        axes[2].plot(budgets, [r[key] for r in rows], "o-", color=color, label=label)
    axes[2].set_ylabel("Fraction of all positive targets")
    axes[2].set_title("Loss occurs before DNN reranking")
    for ax in axes:
        ax.set_xlabel("Observed history budget")
        ax.set_xticks(budgets)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("Exposed MIND-small baseline: no explicit interests, three-seed means")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
