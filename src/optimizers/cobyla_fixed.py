"""Baseline: COBYLA optimizer on angles only, with fixed pipeline settings.

Uses generic reasonable defaults (p=1, shots=1024, opt_level=1, init=random).
These are NOT cherry-picked to be optimal — they represent the "standard QAOA
approach" of picking sensible defaults and optimizing just the angles.
"""

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qiskit_aer.noise import NoiseModel

from src.quantum.qaoa_circuit import initialize_angles
from src.quantum.qaoa_objective import evaluate_pipeline

# Fixed pipeline — generic reasonable defaults
FIXED_P = 1
FIXED_SHOTS = 1024
FIXED_OPT_LEVEL = 1
FIXED_INIT = "random"


def run(G: nx.Graph, budget: int, seed: int,
        noise_model: NoiseModel) -> tuple[float, list[dict]]:
    """
    COBYLA on gamma, beta only. Pipeline is fixed to defaults.
    Represents the standard approach: pick reasonable settings, optimize angles.
    """
    rng = np.random.RandomState(seed)
    p = FIXED_P

    eval_count = [0]
    best_cut = [0.0]
    history = []

    def objective(params):
        if eval_count[0] >= budget:
            return -best_cut[0]
        eval_count[0] += 1

        gamma = np.array(params[:p])
        beta = np.array(params[p:])
        config = {"p": p, "shots": FIXED_SHOTS,
                  "optimization_level": FIXED_OPT_LEVEL,
                  "init": "cobyla", "gamma": gamma, "beta": beta}
        cut = evaluate_pipeline(G, config, noise_model)
        best_cut[0] = max(best_cut[0], cut)

        history.append({"eval": eval_count[0], "cut": cut,
                         "best_so_far": best_cut[0],
                         "config": {"p": p, "shots": FIXED_SHOTS,
                                    "optimization_level": FIXED_OPT_LEVEL,
                                    "init": "cobyla"}})
        return -cut

    # Random starting point
    gamma0, beta0 = initialize_angles(p, FIXED_INIT, rng)
    x0 = np.concatenate([gamma0, beta0])

    minimize(objective, x0, method="COBYLA",
             options={"maxiter": budget, "rhobeg": 0.5})

    return best_cut[0], history
