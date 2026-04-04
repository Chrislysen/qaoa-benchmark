"""Baseline: Pure random search over the full pipeline space."""

import networkx as nx
import numpy as np
from qiskit_aer.noise import NoiseModel

from src.quantum.qaoa_circuit import initialize_angles
from src.quantum.qaoa_objective import evaluate_pipeline

# Pipeline search space — shared by all optimizers
P_CHOICES = [1, 2, 3, 4]
SHOT_CHOICES = [64, 128, 256, 512, 1024, 2048]
OPT_LEVEL_CHOICES = [0, 1, 2, 3]
INIT_CHOICES = ["random", "linear_ramp"]


def run(G: nx.Graph, budget: int, seed: int,
        noise_model: NoiseModel) -> tuple[float, list[dict]]:
    """
    Pure random search. Draws each pipeline variable uniformly.
    No learning, no adaptation. This is the floor.
    """
    rng = np.random.RandomState(seed)
    best_cut = 0.0
    history = []

    for i in range(budget):
        p = int(rng.choice(P_CHOICES))
        shots = int(rng.choice(SHOT_CHOICES))
        opt_level = int(rng.choice(OPT_LEVEL_CHOICES))
        init = str(rng.choice(INIT_CHOICES))
        gamma, beta = initialize_angles(p, init, rng)

        config = {"p": p, "shots": shots, "optimization_level": opt_level,
                  "init": init, "gamma": gamma, "beta": beta}
        cut = evaluate_pipeline(G, config, noise_model)
        best_cut = max(best_cut, cut)
        history.append({"eval": i + 1, "cut": cut, "best_so_far": best_cut,
                         "config": {k: v for k, v in config.items()
                                    if k not in ("gamma", "beta")}})

    return best_cut, history
