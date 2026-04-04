"""Evaluation metrics for QAOA benchmarking."""

import numpy as np


def approximation_ratio(cut_value: float, optimal_cut: int) -> float:
    """Approximation ratio = best cut found / optimal cut."""
    return cut_value / optimal_cut


def garbage_rate(history: list[dict], optimal_cut: int,
                 random_baseline_ratio: float) -> float:
    """Fraction of evaluations at or below random assignment baseline (+0.02 margin)."""
    threshold = (random_baseline_ratio + 0.02) * optimal_cut
    n_garbage = sum(1 for h in history if h["cut"] <= threshold)
    return n_garbage / len(history) if history else 0.0


def best_so_far_trajectory(history: list[dict]) -> list[float]:
    """Extract the best-so-far trajectory from evaluation history."""
    return [h["best_so_far"] for h in history]
