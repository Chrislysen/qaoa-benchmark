"""Baseline: SPSA optimizer on angles only, with fixed pipeline settings.

SPSA (Simultaneous Perturbation Stochastic Approximation) is the standard
optimizer for noisy quantum objectives. Uses only 2 function evaluations
per iteration regardless of parameter dimension.
"""

import networkx as nx
import numpy as np
from qiskit_aer.noise import NoiseModel

from src.quantum.qaoa_circuit import initialize_angles
from src.quantum.qaoa_objective import evaluate_pipeline

# Fixed pipeline — same defaults as COBYLA baseline
FIXED_P = 1
FIXED_SHOTS = 1024
FIXED_OPT_LEVEL = 1
FIXED_INIT = "random"

# SPSA hyperparameters (standard choices from Spall 1998)
A_FRAC = 0.1       # a = a0 / (A + k + 1)^alpha, A = A_FRAC * budget
ALPHA = 0.602
GAMMA = 0.101
C0 = 0.2            # Initial perturbation size
A0 = 0.5            # Initial step size


def run(G: nx.Graph, budget: int, seed: int,
        noise_model: NoiseModel) -> tuple[float, list[dict]]:
    """
    SPSA on gamma, beta only. Pipeline is fixed to defaults.
    Each SPSA iteration uses 2 evaluations (f(x+delta) and f(x-delta)).
    """
    rng = np.random.RandomState(seed)
    p = FIXED_P

    # Initialize
    gamma0, beta0 = initialize_angles(p, FIXED_INIT, rng)
    theta = np.concatenate([gamma0, beta0])
    dim = len(theta)

    best_cut = 0.0
    best_theta = theta.copy()
    history = []
    eval_count = 0

    A = A_FRAC * budget
    n_iterations = budget // 2  # 2 evals per iteration

    def eval_theta(t):
        nonlocal eval_count, best_cut
        eval_count += 1
        gamma_val = np.clip(t[:p], 0, 2 * np.pi)
        beta_val = np.clip(t[p:], 0, np.pi)
        config = {"p": p, "shots": FIXED_SHOTS,
                  "optimization_level": FIXED_OPT_LEVEL,
                  "init": "spsa", "gamma": gamma_val, "beta": beta_val}
        cut = evaluate_pipeline(G, config, noise_model)
        best_cut = max(best_cut, cut)
        history.append({"eval": eval_count, "cut": cut,
                         "best_so_far": best_cut,
                         "config": {"p": p, "shots": FIXED_SHOTS,
                                    "optimization_level": FIXED_OPT_LEVEL,
                                    "init": "spsa"}})
        return cut

    for k in range(n_iterations):
        if eval_count >= budget:
            break

        ak = A0 / (A + k + 1) ** ALPHA
        ck = C0 / (k + 1) ** GAMMA

        # Random perturbation direction (Bernoulli +/-1)
        delta = rng.choice([-1, 1], size=dim).astype(float)

        # Evaluate at theta + ck*delta and theta - ck*delta
        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta

        if eval_count >= budget - 1:
            # Only budget for one more eval
            eval_theta(theta)
            break

        cut_plus = eval_theta(theta_plus)
        cut_minus = eval_theta(theta_minus)

        # SPSA gradient estimate (we maximize, so ascend)
        # g_k = (f(x+) - f(x-)) / (2 * ck * delta)
        g_hat = (cut_plus - cut_minus) / (2 * ck * delta)

        # Update (ascent since we maximize cut)
        theta = theta + ak * g_hat

    # If we have remaining budget, evaluate the final theta
    while eval_count < budget:
        eval_theta(theta)
        # Perturb slightly to explore
        theta = theta + rng.normal(0, 0.1, size=dim)

    return best_cut, history
