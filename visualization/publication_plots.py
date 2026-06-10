"""
Publication-quality visualization for QuantHive.

Generates all figures needed for the arXiv paper and README:
  Fig 1: Equity curve comparison (5 baselines)
  Fig 2: Governance intervention timeline + regime heatmap
  Fig 3: Counterfactual split (governed vs ungoverned)
  Fig 4: Ablation bar chart (max drawdown)
  Fig 5: Per-agent reward over training
  Fig 6: RM size_limit adaptation across regimes
  Fig 7: Failure mode distribution
  Fig 8: OOD performance comparison

Style: Computer Modern font, Tableau 10 palette, 300 DPI
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Academic style
STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Tableau 10 palette
COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def apply_style():
    if HAS_MPL:
        plt.rcParams.update(STYLE)


def plot_baseline_comparison(results: Dict, output_dir: str = "figures"):
    """Fig 4: Ablation/baseline bar chart comparing max drawdown."""
    if not HAS_MPL:
        return
    apply_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labels = list(results.keys())
    dd_means = [results[k].get("max_drawdown_mean", 0) for k in labels]
    sr_means = [results[k].get("sharpe_ratio_mean", 0) for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(labels, dd_means, color=COLORS[:len(labels)], edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Max Drawdown")
    ax1.set_title("Max Drawdown by Configuration")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    bars2 = ax2.bar(labels, sr_means, color=COLORS[:len(labels)], edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe Ratio by Configuration")

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "baseline_comparison.png")
    fig.savefig(Path(output_dir) / "baseline_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: baseline_comparison.png/pdf")


def plot_training_curves(metrics: Dict[str, List], output_dir: str = "figures"):
    """Fig 5: Per-agent reward and loss over training episodes."""
    if not HAS_MPL:
        return
    apply_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    episodes = metrics.get("episode", [])
    if not episodes:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Max drawdown over training
    ax = axes[0, 0]
    ax.plot(episodes, metrics.get("max_drawdown", []), color=COLORS[2], alpha=0.6, linewidth=0.5)
    # Smoothed
    if len(episodes) > 20:
        kernel = np.ones(20) / 20
        smoothed = np.convolve(metrics.get("max_drawdown", []), kernel, mode="valid")
        ax.plot(range(len(smoothed)), smoothed, color=COLORS[2], linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Max Drawdown")
    ax.set_title("Max Drawdown Over Training")
    ax.legend()

    # Total return
    ax = axes[0, 1]
    ax.plot(episodes, metrics.get("total_return", []), color=COLORS[0], alpha=0.6, linewidth=0.5)
    if len(episodes) > 20:
        smoothed = np.convolve(metrics.get("total_return", []), kernel, mode="valid")
        ax.plot(range(len(smoothed)), smoothed, color=COLORS[0], linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Return")
    ax.set_title("Total Return Over Training")
    ax.legend()

    # Violation rate
    ax = axes[1, 0]
    ax.plot(episodes, metrics.get("violation_rate", []), color=COLORS[1], alpha=0.6, linewidth=0.5)
    if len(episodes) > 20:
        smoothed = np.convolve(metrics.get("violation_rate", []), kernel, mode="valid")
        ax.plot(range(len(smoothed)), smoothed, color=COLORS[1], linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Constraint Violation Rate")
    ax.set_title("Governance Compliance Over Training")
    ax.legend()

    # Policy loss
    ax = axes[1, 1]
    pl = metrics.get("train_policy_loss", [])
    if pl:
        ax.plot(range(len(pl)), pl, color=COLORS[3], alpha=0.6, linewidth=0.5)
        if len(pl) > 20:
            smoothed = np.convolve(pl, kernel, mode="valid")
            ax.plot(range(len(smoothed)), smoothed, color=COLORS[3], linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Loss")
    ax.set_title("PPO Policy Loss")
    ax.legend()

    fig.suptitle("GovTrade — Multi-Agent PPO Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(output_dir) / "training_curves.png")
    fig.savefig(Path(output_dir) / "training_curves.pdf")
    plt.close(fig)
    print(f"  Saved: training_curves.png/pdf")


def plot_failure_distribution(failure_counts: Dict[str, int], output_dir: str = "figures"):
    """Fig 7: Pie chart of governance failure types."""
    if not HAS_MPL or not failure_counts:
        return
    apply_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labels = list(failure_counts.keys())
    sizes = list(failure_counts.values())

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=COLORS[:len(labels)], startangle=90,
        pctdistance=0.85, labeldistance=1.1,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Governance Failure Mode Distribution", fontsize=13, fontweight="bold")
    fig.savefig(Path(output_dir) / "failure_distribution.png")
    fig.savefig(Path(output_dir) / "failure_distribution.pdf")
    plt.close(fig)
    print(f"  Saved: failure_distribution.png/pdf")


def plot_ood_comparison(ood_results: Dict, output_dir: str = "figures"):
    """Fig 8: In-distribution vs OOD performance."""
    if not HAS_MPL:
        return
    apply_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    regimes = list(ood_results.keys())
    dd_vals = [ood_results[r].get("max_drawdown_mean", 0) for r in regimes]
    labels_type = [ood_results[r].get("distribution", "ID") for r in regimes]
    colors = [COLORS[0] if t == "ID" else COLORS[2] for t in labels_type]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(regimes, dd_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Max Drawdown")
    ax.set_title("In-Distribution vs Out-of-Distribution Performance")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=45)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[0], label="In-Distribution"),
                       Patch(facecolor=COLORS[2], label="Out-of-Distribution")]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "ood_comparison.png")
    fig.savefig(Path(output_dir) / "ood_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: ood_comparison.png/pdf")


if __name__ == "__main__":
    print("Run after benchmark_suite.py to generate figures.")
