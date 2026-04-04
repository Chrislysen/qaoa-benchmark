"""Baseline: Random explore + best config replay.

Phase 1: random search for 50% of budget.
Phase 2: lock the best discrete config, do random angle search within it.

This is a "dumb" explore-then-exploit — no learning model, just replays the
best config. Tests whether simply locking a config and refining angles helps,
even without a proper learning mechanism.
"""

import networkx as nx
import numpy as np
from qiskit_aer.noise import NoiseModel

from src.quantum.qaoa_circuit import initialize_angles
from src.quantum.qaoa_objective import evaluate_pipeline
from src.optimizers.random_search import P_CHOICES, SHOT_CHOICES, OPT_LEVEL_CHOICES, INIT_CHOICES


def run(G: nx.Graph, budget: int, seed: int,
        noise_model: NoiseModel) -> tuple[float, list[dict]]:
    """
    Phase 1: random pipeline search (50% of budget).
    Phase 2: fix best discrete config, random angle search (50% of budget).
    """
    rng = np.random.RandomState(seed)
    phase1_budget = budget // 2
    phase2_budget = budget - phase1_budget

    best_cut = 0.0
    best_discrete = None
    history = []

    # Phase 1: Random exploration
    for i in range(phase1_budget):
        p = int(rng.choice(P_CHOICES))
        shots = int(rng.choice(SHOT_CHOICES))
        opt_level = int(rng.choice(OPT_LEVEL_CHOICES))
        init = str(rng.choice(INIT_CHOICES))
        gamma, beta = initialize_angles(p, init, rng)

        config = {"p": p, "shots": shots, "optimization_level": opt_level,
                  "init": init, "gamma": gamma, "beta": beta}
        cut = evaluate_pipeline(G, config, noise_model)

        if cut > best_cut:
            best_cut = cut
            best_discrete = {"p": p, "shots": shots,
                             "optimization_level": opt_level}

        history.append({"eval": i + 1, "cut": cut, "best_so_far": best_cut,
                         "config": {"p": p, "shots": shots,
                                    "optimization_level": opt_level,
                                    "init": init},
                         "phase": 1})

    # Phase 2: Replay best config with random angles
    if best_discrete is None:
        best_discrete = {"p": 1, "shots": 1024, "optimization_level": 1}

    for i in range(phase2_budget):
        p = best_discrete["p"]
        # Use both init strategies randomly
        init = str(rng.choice(INIT_CHOICES))
        gamma, beta = initialize_angles(p, init, rng)

        config = {**best_discrete, "init": init, "gamma": gamma, "beta": beta}
        cut = evaluate_pipeline(G, config, noise_model)
        best_cut = max(best_cut, cut)

        history.append({"eval": phase1_budget + i + 1, "cut": cut,
                         "best_so_far": best_cut,
                         "config": {**best_discrete, "init": init},
                         "phase": 2})

    return best_cut, history
