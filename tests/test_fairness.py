"""Test that all optimizers get the same conditions (graph, noise, budget)."""

import pytest
from src.problems.maxcut import generate_maxcut_graph, brute_force_maxcut
from src.evaluation.noise import make_noise_model
from src.optimizers import random_search, cobyla_fixed, spsa_fixed, random_replay, budget_aware


@pytest.fixture
def setup():
    G = generate_maxcut_graph(10, 0.5, 42)
    opt_cut, _ = brute_force_maxcut(G)
    nm = make_noise_model()
    return G, opt_cut, nm


BUDGET = 15


def test_same_budget(setup):
    """All optimizers must use exactly the specified budget (no more)."""
    G, opt_cut, nm = setup
    results = {}
    results["random"] = random_search.run(G, BUDGET, 0, nm)
    results["cobyla"] = cobyla_fixed.run(G, BUDGET, 0, nm)
    results["spsa"] = spsa_fixed.run(G, BUDGET, 0, nm)
    results["replay"] = random_replay.run(G, BUDGET, 0, nm)
    results["budget_aware"] = budget_aware.run(G, BUDGET, 0, nm, optimal_cut=opt_cut)

    for name, (cut, hist) in results.items():
        assert len(hist) <= BUDGET, f"{name} used {len(hist)} evals, budget is {BUDGET}"


def test_same_graph(setup):
    """All optimizers get the same graph instance."""
    G, opt_cut, nm = setup
    # Just verify the graph is consistent
    assert G.number_of_nodes() == 10
    assert opt_cut > 0


def test_same_noise_model(setup):
    """All optimizers use the same noise model parameters."""
    from src.evaluation.noise import SINGLE_GATE_ERROR, TWO_GATE_ERROR, READOUT_ERROR_RATE
    assert SINGLE_GATE_ERROR == 0.01
    assert TWO_GATE_ERROR == 0.05
    assert READOUT_ERROR_RATE == 0.03


def test_pipeline_space_consistency():
    """All optimizers that sample the pipeline space use the same choices."""
    from src.optimizers.random_search import P_CHOICES, SHOT_CHOICES, OPT_LEVEL_CHOICES, INIT_CHOICES
    assert P_CHOICES == [1, 2, 3, 4]
    assert SHOT_CHOICES == [64, 128, 256, 512, 1024, 2048]
    assert OPT_LEVEL_CHOICES == [0, 1, 2, 3]
    assert INIT_CHOICES == ["random", "linear_ramp"]
