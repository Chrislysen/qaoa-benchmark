"""All plot generation for the QAOA Pipeline Optimizer benchmark."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Consistent colors and styles for all plots
OPTIMIZER_STYLES = {
    "budget_aware":  {"color": "#4CAF50", "marker": "o", "label": "Budget-Aware (ours)"},
    "random":        {"color": "#9E9E9E", "marker": "s", "label": "Random Search"},
    "cobyla":        {"color": "#2196F3", "marker": "^", "label": "COBYLA (fixed pipeline)"},
    "spsa":          {"color": "#FF9800", "marker": "v", "label": "SPSA (fixed pipeline)"},
    "random_replay": {"color": "#E91E63", "marker": "D", "label": "Random + Replay"},
}

OPTIMIZER_ORDER = ["budget_aware", "random", "cobyla", "spsa", "random_replay"]


def plot_hero(results: dict, save_path: str):
    """
    Plot 1 — Hero Plot: Approximation ratio vs evaluation budget.
    One line per optimizer, error bands +/- 1 std.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(results["budgets"])

    for opt_name in OPTIMIZER_ORDER:
        if opt_name not in results["by_optimizer"]:
            continue
        style = OPTIMIZER_STYLES[opt_name]
        opt_data = results["by_optimizer"][opt_name]

        means = [np.mean(opt_data[b]["ratios"]) for b in budgets]
        stds = [np.std(opt_data[b]["ratios"]) for b in budgets]

        ax.errorbar(budgets, means, yerr=stds, marker=style["marker"],
                    label=style["label"], color=style["color"],
                    linewidth=2, markersize=7, capsize=4)

    ax.set_xlabel("Evaluation Budget", fontsize=13)
    ax.set_ylabel("Approximation Ratio", fontsize=13)
    ax.set_title("Approximation Ratio vs Evaluation Budget", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    if "random_baseline" in results:
        ax.axhline(y=results["random_baseline"], color="red", linestyle=":",
                   alpha=0.5, label=f"Random assignment ({results['random_baseline']:.3f})")
        ax.legend(fontsize=10, loc="lower right")

    ax.set_ylim(0.65, 0.82)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_garbage_rate(results: dict, save_path: str):
    """
    Plot 2 — Garbage rate vs evaluation budget.
    Shows what fraction of evaluations produce garbage results.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    budgets = sorted(results["budgets"])

    for opt_name in OPTIMIZER_ORDER:
        if opt_name not in results["by_optimizer"]:
            continue
        style = OPTIMIZER_STYLES[opt_name]
        opt_data = results["by_optimizer"][opt_name]

        means = [np.mean(opt_data[b]["garbage_rates"]) for b in budgets]
        stds = [np.std(opt_data[b]["garbage_rates"]) for b in budgets]

        ax.errorbar(budgets, means, yerr=stds, marker=style["marker"],
                    label=style["label"], color=style["color"],
                    linewidth=2, markersize=7, capsize=4)

    ax.set_xlabel("Evaluation Budget", fontsize=13)
    ax.set_ylabel("Garbage Rate", fontsize=13)
    ax.set_title("Fraction of Wasted Evaluations vs Budget", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_win_rate(results: dict, save_path: str):
    """
    Plot 3 — Per-budget win rate bar chart.
    For each budget, what fraction of (graph, seed) pairs each optimizer wins.
    """
    budgets = sorted(results["budgets"])
    opt_names = [n for n in OPTIMIZER_ORDER if n in results["by_optimizer"]]
    n_opts = len(opt_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(budgets))
    width = 0.8 / n_opts

    for j, opt_name in enumerate(opt_names):
        style = OPTIMIZER_STYLES[opt_name]
        win_rates = []
        for b in budgets:
            if "win_rates" in results["by_optimizer"][opt_name][b]:
                win_rates.append(results["by_optimizer"][opt_name][b]["win_rates"])
            else:
                win_rates.append(0.0)

        offset = (j - n_opts / 2 + 0.5) * width
        ax.bar(x + offset, win_rates, width * 0.9, label=style["label"],
               color=style["color"], alpha=0.8)

    ax.set_xlabel("Evaluation Budget", fontsize=13)
    ax.set_ylabel("Win Rate", fontsize=13)
    ax.set_title("Per-Budget Win Rate", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_convergence(results: dict, save_path: str):
    """
    Plot 4 — Convergence curves: best-so-far vs evaluation number.
    One subplot per selected budget level.
    """
    # Pick 3 representative budgets
    budgets = sorted(results["budgets"])
    show_budgets = [b for b in [15, 25, 50] if b in budgets]
    if not show_budgets:
        show_budgets = budgets[:3]

    fig, axes = plt.subplots(1, len(show_budgets), figsize=(6 * len(show_budgets), 5))
    if len(show_budgets) == 1:
        axes = [axes]

    for ax, budget in zip(axes, show_budgets):
        for opt_name in OPTIMIZER_ORDER:
            if opt_name not in results["by_optimizer"]:
                continue
            style = OPTIMIZER_STYLES[opt_name]
            opt_data = results["by_optimizer"][opt_name]

            if budget not in opt_data or "trajectories" not in opt_data[budget]:
                continue

            trajectories = opt_data[budget]["trajectories"]
            if not trajectories:
                continue

            # Pad shorter trajectories to budget length
            max_len = budget
            padded = []
            for traj in trajectories:
                t = list(traj)
                while len(t) < max_len:
                    t.append(t[-1] if t else 0)
                padded.append(t[:max_len])

            arr = np.array(padded)
            mean_traj = np.mean(arr, axis=0)
            std_traj = np.std(arr, axis=0)
            evals = np.arange(1, max_len + 1)

            ax.plot(evals, mean_traj, label=style["label"],
                    color=style["color"], linewidth=2)
            ax.fill_between(evals, mean_traj - std_traj, mean_traj + std_traj,
                           alpha=0.15, color=style["color"])

        ax.set_xlabel("Evaluation Number", fontsize=11)
        ax.set_ylabel("Best-So-Far Ratio", fontsize=11)
        ax.set_title(f"Budget = {budget}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    plt.suptitle("Convergence Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
