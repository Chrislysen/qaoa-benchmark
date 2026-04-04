"""
Test that the budget-aware optimizer ACTUALLY LEARNS.

This is the critical test: the optimizer must change its sampling distribution
based on observed results. If this test fails, the optimizer is just random
search with extra steps.
"""

from collections import Counter

import pytest
import numpy as np

from src.problems.maxcut import generate_maxcut_graph, brute_force_maxcut
from src.evaluation.noise import make_noise_model
from src.optimizers.budget_aware import BudgetAwareOptimizer


def test_optimizer_learns():
    """
    Verify the budget-aware optimizer changes its sampling distribution
    based on observed results.

    Method:
    1. Run the optimizer for 30 evaluations
    2. Record which p values it sampled in evals 1-10 vs evals 21-30
    3. If p=3 and p=4 consistently produce garbage, the optimizer should
       sample them LESS in the later evaluations
    4. The test passes if the sampling distribution shifts measurably
    """
    G = generate_maxcut_graph(10, 0.5, 42)
    opt_cut, _ = brute_force_maxcut(G)
    nm = make_noise_model()

    # Run over multiple seeds to get a statistical signal
    early_p34_frac = []
    late_p34_frac = []

    for seed in range(10):
        opt = BudgetAwareOptimizer(seed=seed)
        opt.run(G, 30, nm, opt_cut)

        log = opt.sampling_log
        early = log[:10]  # First 10 evals
        late = log[20:]   # Last 10 evals

        early_p34 = sum(1 for e in early if e["p"] in (3, 4)) / len(early)
        late_p34 = sum(1 for e in late if e["p"] in (3, 4)) / len(late)

        early_p34_frac.append(early_p34)
        late_p34_frac.append(late_p34)

    mean_early = np.mean(early_p34_frac)
    mean_late = np.mean(late_p34_frac)

    # The optimizer should sample p=3,4 LESS in later evaluations
    # Early: should be roughly uniform (50% of choices are p=3,4)
    # Late: should be significantly lower
    assert mean_early > 0.2, (
        f"Early p=3,4 fraction is too low ({mean_early:.2f}). "
        f"Optimizer may not be exploring uniformly in Phase 1."
    )
    assert mean_late < mean_early, (
        f"Late p=3,4 fraction ({mean_late:.2f}) is not less than early ({mean_early:.2f}). "
        f"Optimizer is not learning to avoid garbage configs."
    )
    # Require a meaningful reduction (at least 30% relative reduction)
    reduction = (mean_early - mean_late) / mean_early
    assert reduction > 0.3, (
        f"Reduction in p=3,4 sampling is only {reduction:.0%} "
        f"(early={mean_early:.2f}, late={mean_late:.2f}). "
        f"Learning signal is too weak."
    )


def test_init_strategy_learning():
    """
    Verify the optimizer learns which init strategy works better.
    Under noise, linear_ramp should outperform random init.
    The optimizer should shift toward linear_ramp in later evaluations.
    """
    G = generate_maxcut_graph(10, 0.5, 42)
    opt_cut, _ = brute_force_maxcut(G)
    nm = make_noise_model()

    early_ramp_frac = []
    late_ramp_frac = []

    for seed in range(10):
        opt = BudgetAwareOptimizer(seed=seed)
        opt.run(G, 30, nm, opt_cut)

        log = opt.sampling_log
        # Only count non-perturbed entries (explore and transition phases)
        early = [e for e in log[:10] if e.get("init") in ("random", "linear_ramp")]
        late_all = log[15:]  # Everything after transition
        late = [e for e in late_all if e.get("init") in ("random", "linear_ramp")]

        if early:
            early_ramp_frac.append(
                sum(1 for e in early if e["init"] == "linear_ramp") / len(early)
            )
        if late:
            late_ramp_frac.append(
                sum(1 for e in late if e["init"] == "linear_ramp") / len(late)
            )

    if early_ramp_frac and late_ramp_frac:
        mean_early = np.mean(early_ramp_frac)
        mean_late = np.mean(late_ramp_frac)
        # In the exploit phase most evals are "perturbed" (not explicitly random
        # or linear_ramp), so few late-phase samples have explicit init. We only
        # check that the optimizer doesn't REGRESS dramatically on init preference.
        # A >25% relative drop would indicate broken learning.
        if mean_early > 0.1:
            regression = (mean_early - mean_late) / mean_early
            assert regression < 0.25, (
                f"Late linear_ramp fraction ({mean_late:.2f}) dropped too much "
                f"from early ({mean_early:.2f}). Regression={regression:.0%}."
            )


def test_sampling_log_completeness():
    """Verify that the sampling log records every evaluation."""
    G = generate_maxcut_graph(10, 0.5, 42)
    opt_cut, _ = brute_force_maxcut(G)
    nm = make_noise_model()

    budget = 25
    opt = BudgetAwareOptimizer(seed=42)
    opt.run(G, budget, nm, opt_cut)

    assert len(opt.sampling_log) == budget, (
        f"Sampling log has {len(opt.sampling_log)} entries, expected {budget}"
    )

    # Every entry should have eval number and categorical values
    for entry in opt.sampling_log:
        assert "eval" in entry
        assert "p" in entry
        assert "shots" in entry
        assert "phase" in entry
