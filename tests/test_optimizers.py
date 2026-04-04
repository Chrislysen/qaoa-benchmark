"""Test that each optimizer runs without error and returns valid results."""

import pytest
import networkx as nx
import numpy as np

from src.problems.maxcut import generate_maxcut_graph, brute_force_maxcut
from src.evaluation.noise import make_noise_model
from src.optimizers import random_search, cobyla_fixed, spsa_fixed, random_replay, budget_aware


@pytest.fixture
def setup():
    G = generate_maxcut_graph(10, 0.5, 42)
    opt_cut, _ = brute_force_maxcut(G)
    nm = make_noise_model()
    return G, opt_cut, nm


BUDGET = 10


class TestRandomSearch:
    def test_runs(self, setup):
        G, opt_cut, nm = setup
        cut, hist = random_search.run(G, BUDGET, 42, nm)
        assert len(hist) == BUDGET
        assert cut > 0
        assert cut == hist[-1]["best_so_far"]

    def test_history_structure(self, setup):
        G, opt_cut, nm = setup
        _, hist = random_search.run(G, BUDGET, 42, nm)
        for h in hist:
            assert "eval" in h
            assert "cut" in h
            assert "best_so_far" in h
            assert h["cut"] > 0


class TestCOBYLA:
    def test_runs(self, setup):
        G, opt_cut, nm = setup
        cut, hist = cobyla_fixed.run(G, BUDGET, 42, nm)
        assert len(hist) <= BUDGET
        assert cut > 0

    def test_fixed_pipeline(self, setup):
        G, opt_cut, nm = setup
        _, hist = cobyla_fixed.run(G, BUDGET, 42, nm)
        for h in hist:
            assert h["config"]["p"] == 1
            assert h["config"]["shots"] == 1024


class TestSPSA:
    def test_runs(self, setup):
        G, opt_cut, nm = setup
        cut, hist = spsa_fixed.run(G, BUDGET, 42, nm)
        assert len(hist) <= BUDGET
        assert cut > 0

    def test_fixed_pipeline(self, setup):
        G, opt_cut, nm = setup
        _, hist = spsa_fixed.run(G, BUDGET, 42, nm)
        for h in hist:
            assert h["config"]["p"] == 1
            assert h["config"]["shots"] == 1024


class TestRandomReplay:
    def test_runs(self, setup):
        G, opt_cut, nm = setup
        cut, hist = random_replay.run(G, BUDGET, 42, nm)
        assert len(hist) == BUDGET
        assert cut > 0

    def test_two_phases(self, setup):
        G, opt_cut, nm = setup
        _, hist = random_replay.run(G, BUDGET, 42, nm)
        phases = [h["phase"] for h in hist]
        assert 1 in phases
        assert 2 in phases


class TestBudgetAware:
    def test_runs(self, setup):
        G, opt_cut, nm = setup
        cut, hist = budget_aware.run(G, BUDGET, 42, nm, optimal_cut=opt_cut)
        assert len(hist) == BUDGET
        assert cut > 0

    def test_history_has_phases(self, setup):
        G, opt_cut, nm = setup
        _, hist = budget_aware.run(G, 20, 42, nm, optimal_cut=opt_cut)
        phases = set(h["phase"] for h in hist)
        assert "explore" in phases

    def test_best_so_far_monotonic(self, setup):
        G, opt_cut, nm = setup
        _, hist = budget_aware.run(G, 20, 42, nm, optimal_cut=opt_cut)
        bests = [h["best_so_far"] for h in hist]
        for i in range(1, len(bests)):
            assert bests[i] >= bests[i - 1]
