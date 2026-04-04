"""
Budget-Aware Pipeline Optimizer (our method).

Uses a multi-armed bandit approach over discrete pipeline variables:
  - Starts BLIND: uniform sampling of all categorical variables
  - LEARNS from evaluations: tracks mean approximation ratio per value
  - ADAPTS: shifts sampling probability proportional to observed performance
  - EXPLOITS: refines angles within the best-found discrete config

The optimizer has ZERO hard-coded knowledge about which configs are good.
It discovers that low-p is better under noise, that linear_ramp beats random
init, etc. — all from its own evaluation data.

Mandatory logging: records which categorical values were sampled at each
evaluation, enabling verification that learning actually occurred.
"""

from collections import defaultdict

import networkx as nx
import numpy as np
from qiskit_aer.noise import NoiseModel

from src.quantum.qaoa_circuit import initialize_angles
from src.quantum.qaoa_objective import evaluate_pipeline
from src.optimizers.random_search import P_CHOICES, SHOT_CHOICES, OPT_LEVEL_CHOICES, INIT_CHOICES


class BudgetAwareOptimizer:
    """
    Multi-armed bandit over discrete pipeline variables + local angle refinement.

    Phase 1 (Explore): Sample uniformly, build a performance model.
    Phase 2 (Exploit): Sample categoricals proportional to observed performance,
                       refine angles around the best-found configuration.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

        # Performance tracking per categorical value
        # key -> list of observed approximation ratios
        self.observations = {
            "p": defaultdict(list),
            "shots": defaultdict(list),
            "optimization_level": defaultdict(list),
            "init": defaultdict(list),
        }

        # Sampling log for verification
        self.sampling_log = []  # List of {"eval": i, "p": v, "shots": v, ...}

        # Best-found state
        self.best_cut = 0.0
        self.best_config = None
        self.best_angles = None

    def _sample_uniform(self) -> dict:
        """Sample a pipeline config uniformly (no prior knowledge)."""
        p = int(self.rng.choice(P_CHOICES))
        shots = int(self.rng.choice(SHOT_CHOICES))
        opt_level = int(self.rng.choice(OPT_LEVEL_CHOICES))
        init = str(self.rng.choice(INIT_CHOICES))
        gamma, beta = initialize_angles(p, init, self.rng)
        return {"p": p, "shots": shots, "optimization_level": opt_level,
                "init": init, "gamma": gamma, "beta": beta}

    def _compute_sampling_weights(self, variable: str) -> dict:
        """
        Compute sampling probability for each value of a categorical variable,
        proportional to its observed mean performance.

        Uses softmax with temperature to avoid zero-probability categories
        (always leaves some exploration probability).
        """
        obs = self.observations[variable]
        if variable == "p":
            choices = P_CHOICES
        elif variable == "shots":
            choices = SHOT_CHOICES
        elif variable == "optimization_level":
            choices = OPT_LEVEL_CHOICES
        elif variable == "init":
            choices = INIT_CHOICES
        else:
            raise ValueError(f"Unknown variable: {variable}")

        # Compute mean performance for each value
        means = {}
        for val in choices:
            if obs[val]:
                means[val] = np.mean(obs[val])
            else:
                # Unseen values get the global mean (optimistic)
                all_obs = [r for vals in obs.values() for r in vals]
                means[val] = np.mean(all_obs) if all_obs else 0.5

        # Softmax with temperature for exploration
        # Temperature 0.1 = fairly aggressive exploitation
        # Temperature 1.0 = nearly uniform
        temperature = 0.3
        vals = list(means.keys())
        scores = np.array([means[v] for v in vals])

        # Shift for numerical stability
        scores = scores - scores.max()
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / exp_scores.sum()

        return dict(zip(vals, probs))

    def _sample_learned(self) -> dict:
        """Sample a pipeline config using learned performance model."""
        # Sample each categorical variable according to learned weights
        p_weights = self._compute_sampling_weights("p")
        p_vals, p_probs = zip(*p_weights.items())
        p = int(self.rng.choice(p_vals, p=p_probs))

        shot_weights = self._compute_sampling_weights("shots")
        s_vals, s_probs = zip(*shot_weights.items())
        shots = int(self.rng.choice(s_vals, p=s_probs))

        ol_weights = self._compute_sampling_weights("optimization_level")
        ol_vals, ol_probs = zip(*ol_weights.items())
        opt_level = int(self.rng.choice(ol_vals, p=ol_probs))

        init_weights = self._compute_sampling_weights("init")
        i_vals, i_probs = zip(*init_weights.items())
        init = str(self.rng.choice(i_vals, p=i_probs))

        gamma, beta = initialize_angles(p, init, self.rng)
        return {"p": p, "shots": shots, "optimization_level": opt_level,
                "init": init, "gamma": gamma, "beta": beta}

    def _sample_exploit(self) -> dict:
        """
        Exploit: use the best-found discrete config and perturb angles
        around the best-found angles.
        """
        if self.best_config is None:
            return self._sample_learned()

        p = self.best_config["p"]
        center_gamma, center_beta = self.best_angles

        # Perturbation radius — moderate, not too tight
        gamma = center_gamma + self.rng.uniform(-0.8, 0.8, size=p)
        beta = center_beta + self.rng.uniform(-0.4, 0.4, size=p)
        gamma = np.clip(gamma, 0, 2 * np.pi)
        beta = np.clip(beta, 0, np.pi)

        return {"p": self.best_config["p"],
                "shots": self.best_config["shots"],
                "optimization_level": self.best_config["optimization_level"],
                "init": "perturbed",
                "gamma": gamma, "beta": beta}

    def _record_observation(self, config: dict, ratio: float):
        """Record the result of an evaluation."""
        self.observations["p"][config["p"]].append(ratio)
        self.observations["shots"][config["shots"]].append(ratio)
        self.observations["optimization_level"][config["optimization_level"]].append(ratio)
        # Don't record "perturbed" as an init strategy
        if config["init"] in ("random", "linear_ramp"):
            self.observations["init"][config["init"]].append(ratio)

    def run(self, G: nx.Graph, budget: int,
            noise_model: NoiseModel, optimal_cut: int) -> tuple[float, list[dict]]:
        """
        Run the budget-aware optimizer.

        Phase 1 (first 35% of budget): Uniform exploration.
        Transition (next 25%): Learned sampling.
        Phase 2 (final 40%): Exploit best config with angle refinement,
                             with occasional learned sampling for diversity.
        """
        phase1_end = max(3, int(budget * 0.35))
        transition_end = max(phase1_end + 1, int(budget * 0.60))
        history = []

        for i in range(budget):
            # Decide sampling strategy
            if i < phase1_end:
                config = self._sample_uniform()
                phase = "explore"
            elif i < transition_end:
                config = self._sample_learned()
                phase = "transition"
            else:
                # 70% exploit best config, 30% learned sampling for diversity
                if self.rng.random() < 0.7 and self.best_angles is not None:
                    config = self._sample_exploit()
                    phase = "exploit"
                else:
                    config = self._sample_learned()
                    phase = "learned"

            # Evaluate
            cut = evaluate_pipeline(G, config, noise_model)
            ratio = cut / optimal_cut

            # Update model
            self._record_observation(config, ratio)

            # Update best
            if cut > self.best_cut:
                self.best_cut = cut
                self.best_config = {k: v for k, v in config.items()
                                    if k not in ("gamma", "beta")}
                self.best_angles = (config["gamma"].copy(), config["beta"].copy())

            # Log
            discrete_config = {k: v for k, v in config.items()
                               if k not in ("gamma", "beta")}
            self.sampling_log.append({
                "eval": i + 1, "phase": phase, **discrete_config
            })
            history.append({
                "eval": i + 1, "cut": cut, "best_so_far": self.best_cut,
                "config": discrete_config, "phase": phase
            })

        return self.best_cut, history

    def get_sampling_breakdown(self, n_bins: int = 3) -> str:
        """
        Return a human-readable breakdown of sampling frequencies over time.
        Splits the log into n_bins equal-sized windows.
        """
        if not self.sampling_log:
            return "No evaluations recorded."

        total = len(self.sampling_log)
        bin_size = max(1, total // n_bins)
        lines = []

        for b in range(n_bins):
            start = b * bin_size
            end = min(start + bin_size, total) if b < n_bins - 1 else total
            window = self.sampling_log[start:end]
            if not window:
                continue

            # Count p values
            p_counts = defaultdict(int)
            init_counts = defaultdict(int)
            for entry in window:
                p_counts[entry["p"]] += 1
                init_counts[entry.get("init", "?")] += 1

            p_str = ", ".join(f"p={k}: {v}x" for k, v in sorted(p_counts.items()))
            init_str = ", ".join(f"{k}: {v}x" for k, v in sorted(init_counts.items()))
            lines.append(f"Evals {start+1:3d}-{end:3d}:  {p_str}  |  {init_str}")

        return "\n".join(lines)


def run(G: nx.Graph, budget: int, seed: int,
        noise_model: NoiseModel, optimal_cut: int = None) -> tuple[float, list[dict]]:
    """
    Module-level entry point matching the baseline interface.
    If optimal_cut is not provided, uses a rough estimate.
    """
    if optimal_cut is None:
        # Rough estimate: E[edges]/2 * 2 (generous upper bound)
        optimal_cut = G.number_of_edges()

    opt = BudgetAwareOptimizer(seed=seed)
    best_cut, history = opt.run(G, budget, noise_model, optimal_cut)

    # Attach the optimizer instance for logging access
    history_with_meta = history
    if history_with_meta:
        history_with_meta[0]["_optimizer"] = opt

    return best_cut, history_with_meta
