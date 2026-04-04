"""
Phase 2 Benchmark: Full QAOA Pipeline Optimizer Evaluation
==========================================================
Runs all 5 optimizers across multiple graphs, budgets, and seeds.
Generates the 4 required plots and a summary table.

Usage:
  python experiments/run_benchmark.py                   # Full benchmark
  python experiments/run_benchmark.py --checkpoint      # One-graph checkpoint
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.problems.maxcut import (generate_maxcut_graph, brute_force_maxcut,
                                  random_assignment_expected_ratio)
from src.evaluation.noise import make_noise_model
from src.evaluation.metrics import approximation_ratio, garbage_rate
from src.evaluation.plotting import plot_hero, plot_garbage_rate, plot_win_rate, plot_convergence
from src.optimizers import random_search, cobyla_fixed, spsa_fixed, random_replay, budget_aware

warnings.filterwarnings("ignore")

# Optimizer registry
OPTIMIZERS = {
    "budget_aware": budget_aware,
    "random": random_search,
    "cobyla": cobyla_fixed,
    "spsa": spsa_fixed,
    "random_replay": random_replay,
}


def run_single_experiment(G, opt_cut, noise_model, optimizer_name, budget, seed,
                          random_baseline_ratio):
    """Run a single optimizer on a single (graph, budget, seed) combination."""
    opt_module = OPTIMIZERS[optimizer_name]

    if optimizer_name == "budget_aware":
        best_cut, history = opt_module.run(G, budget, seed, noise_model,
                                           optimal_cut=opt_cut)
    else:
        best_cut, history = opt_module.run(G, budget, seed, noise_model)

    ratio = best_cut / opt_cut
    garb = garbage_rate(history, opt_cut, random_baseline_ratio)
    trajectory = [h["best_so_far"] / opt_cut for h in history]

    # Get optimizer instance for budget_aware
    optimizer_instance = None
    if optimizer_name == "budget_aware" and history and "_optimizer" in history[0]:
        optimizer_instance = history[0]["_optimizer"]

    return {
        "ratio": ratio,
        "garbage_rate": garb,
        "trajectory": trajectory,
        "best_cut": best_cut,
        "optimizer_instance": optimizer_instance,
    }


def run_benchmark(graph_seeds, budgets, n_experiment_seeds, verbose=True):
    """Run the full benchmark."""
    noise_model = make_noise_model()
    results = {
        "budgets": budgets,
        "by_optimizer": defaultdict(lambda: defaultdict(lambda: {
            "ratios": [], "garbage_rates": [], "trajectories": [],
            "win_rates": 0.0
        })),
        "raw": [],  # All individual results for win rate calculation
    }

    total_runs = len(graph_seeds) * len(OPTIMIZERS) * len(budgets) * n_experiment_seeds
    run_count = 0
    start_time = time.time()

    # Pre-compute graphs and optimal cuts
    graphs = {}
    for gs in graph_seeds:
        G = generate_maxcut_graph(10, 0.5, gs)
        opt_cut, _ = brute_force_maxcut(G)
        rb = random_assignment_expected_ratio(G, opt_cut)
        graphs[gs] = (G, opt_cut, rb)
        if verbose:
            print(f"Graph seed={gs}: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges, optimal={opt_cut}, "
                  f"random_baseline={rb:.3f}")

    results["random_baseline"] = np.mean([rb for _, _, rb in graphs.values()])

    if verbose:
        print(f"\nTotal runs: {total_runs}")
        print(f"Optimizers: {list(OPTIMIZERS.keys())}")
        print(f"Budgets: {budgets}")
        print(f"Seeds per experiment: {n_experiment_seeds}")
        print()

    for budget in budgets:
        if verbose:
            print(f"--- Budget = {budget} ---")

        # Collect results for win rate calculation
        budget_results = defaultdict(list)  # opt_name -> list of ratios

        for gs in graph_seeds:
            G, opt_cut, rb = graphs[gs]

            for exp_seed in range(n_experiment_seeds):
                seed = gs * 1000 + exp_seed  # Unique seed per (graph, experiment)

                run_results = {}
                for opt_name in OPTIMIZERS:
                    res = run_single_experiment(
                        G, opt_cut, noise_model, opt_name, budget, seed, rb
                    )
                    run_results[opt_name] = res

                    results["by_optimizer"][opt_name][budget]["ratios"].append(res["ratio"])
                    results["by_optimizer"][opt_name][budget]["garbage_rates"].append(res["garbage_rate"])
                    results["by_optimizer"][opt_name][budget]["trajectories"].append(res["trajectory"])
                    budget_results[opt_name].append(res["ratio"])

                    run_count += 1

                # Determine winner for this (graph, seed) pair
                best_opt = max(run_results, key=lambda k: run_results[k]["ratio"])
                results["raw"].append({
                    "graph_seed": gs, "exp_seed": exp_seed,
                    "budget": budget, "winner": best_opt,
                    "ratios": {k: v["ratio"] for k, v in run_results.items()}
                })

        # Compute win rates for this budget
        budget_raw = [r for r in results["raw"] if r["budget"] == budget]
        n_total = len(budget_raw)
        for opt_name in OPTIMIZERS:
            n_wins = sum(1 for r in budget_raw if r["winner"] == opt_name)
            results["by_optimizer"][opt_name][budget]["win_rates"] = n_wins / n_total if n_total else 0

        if verbose:
            elapsed = time.time() - start_time
            pct = run_count / total_runs * 100
            for opt_name in OPTIMIZERS:
                data = results["by_optimizer"][opt_name][budget]
                m = np.mean(data["ratios"])
                s = np.std(data["ratios"])
                wr = data["win_rates"]
                print(f"  {opt_name:15s}: {m:.3f} +/- {s:.3f}  win={wr:.0%}")
            print(f"  [{pct:.0f}% done, {elapsed:.0f}s elapsed]")

    return results


def print_checkpoint_results(results, graph_seeds):
    """Print detailed results for the one-graph checkpoint."""
    print("\n" + "=" * 70)
    print("ONE-GRAPH CHECKPOINT RESULTS")
    print("=" * 70)

    budget = 25
    if budget not in results["budgets"]:
        budget = results["budgets"][0]

    print(f"\nBudget = {budget}")
    print(f"{'Optimizer':20s} {'Ratio':>10s} {'Garbage':>10s} {'Win Rate':>10s}")
    print("-" * 55)

    for opt_name in OPTIMIZERS:
        data = results["by_optimizer"][opt_name][budget]
        m = np.mean(data["ratios"])
        g = np.mean(data["garbage_rates"])
        w = data["win_rates"]
        print(f"{opt_name:20s} {m:10.3f} {g:10.1%} {w:10.0%}")


def print_summary_table(results):
    """Print a markdown summary table of all results."""
    print("\n\n## Summary Table\n")
    budgets = sorted(results["budgets"])

    header = "| Budget |"
    sep = "|--------|"
    for opt_name in OPTIMIZERS:
        label = opt_name.replace("_", " ").title()
        header += f" {label:>20s} |"
        sep += f"{'':->22s}|"

    print(header)
    print(sep)

    for budget in budgets:
        row = f"| {budget:6d} |"
        for opt_name in OPTIMIZERS:
            data = results["by_optimizer"][opt_name][budget]
            m = np.mean(data["ratios"])
            s = np.std(data["ratios"])
            row += f" {m:.3f} +/- {s:.3f}      |"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="QAOA Pipeline Benchmark")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Run one-graph checkpoint instead of full benchmark")
    parser.add_argument("--full", action="store_true",
                        help="Run full benchmark (5 graphs, 7 budgets, 10 seeds)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "figures"
    results_dir = project_root / "results"

    if args.checkpoint:
        print("QAOA Pipeline Optimizer -- One-Graph Checkpoint")
        print("=" * 70)
        graph_seeds = [42]
        budgets = [25]
        n_seeds = 10
    elif args.full:
        print("QAOA Pipeline Optimizer -- Full Benchmark")
        print("=" * 70)
        graph_seeds = [42, 123, 456, 789, 1000]
        budgets = [10, 15, 20, 25, 30, 40, 50]
        n_seeds = 10
    else:
        # Default: moderate benchmark (3 graphs, 5 budgets, 5 seeds)
        # ~375 runs, ~30-60 min
        print("QAOA Pipeline Optimizer -- Benchmark")
        print("=" * 70)
        graph_seeds = [42, 123, 456]
        budgets = [10, 15, 25, 40, 50]
        n_seeds = 5

    results = run_benchmark(graph_seeds, budgets, n_seeds)

    if args.checkpoint:
        print_checkpoint_results(results, graph_seeds)

        # Print sampling breakdown for budget-aware optimizer
        print("\n--- Budget-Aware Optimizer Sampling Breakdown ---")
        # Re-run one instance to get the breakdown
        G = generate_maxcut_graph(10, 0.5, 42)
        opt_cut, _ = brute_force_maxcut(G)
        nm = make_noise_model()
        _, hist = budget_aware.run(G, 25, 42042, nm, optimal_cut=opt_cut)
        opt_instance = hist[0].get("_optimizer")
        if opt_instance:
            print(opt_instance.get_sampling_breakdown())
    else:
        # Generate all 4 plots
        plot_hero(results, str(figures_dir / "hero_plot.png"))
        print(f"\nSaved: {figures_dir / 'hero_plot.png'}")

        plot_garbage_rate(results, str(figures_dir / "garbage_rate.png"))
        print(f"Saved: {figures_dir / 'garbage_rate.png'}")

        plot_win_rate(results, str(figures_dir / "win_rate.png"))
        print(f"Saved: {figures_dir / 'win_rate.png'}")

        plot_convergence(results, str(figures_dir / "convergence.png"))
        print(f"Saved: {figures_dir / 'convergence.png'}")

        print_summary_table(results)

        # Save raw results
        os.makedirs(results_dir, exist_ok=True)
        save_data = {
            "budgets": results["budgets"],
            "random_baseline": results["random_baseline"],
            "summary": {}
        }
        for opt_name in OPTIMIZERS:
            save_data["summary"][opt_name] = {}
            for b in budgets:
                data = results["by_optimizer"][opt_name][b]
                save_data["summary"][opt_name][b] = {
                    "mean_ratio": float(np.mean(data["ratios"])),
                    "std_ratio": float(np.std(data["ratios"])),
                    "mean_garbage": float(np.mean(data["garbage_rates"])),
                    "win_rate": float(data["win_rates"]),
                }
        with open(results_dir / "benchmark_results.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved: {results_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
