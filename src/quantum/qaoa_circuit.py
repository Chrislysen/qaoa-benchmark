"""QAOA circuit builder and angle initialization strategies."""

import networkx as nx
import numpy as np
from qiskit.circuit import QuantumCircuit


def build_qaoa_circuit(G: nx.Graph, p: int, gamma: list[float],
                       beta: list[float]) -> QuantumCircuit:
    """
    Build a p-layer QAOA circuit for Max-Cut.

    Cost unitary: for each edge (u,v), apply e^{-i * gamma * Z_u Z_v}
                  via CNOT-Rz-CNOT decomposition.
    Mixer unitary: R_x(2*beta) on every qubit.
    """
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)
    qc.h(range(n))

    for layer in range(p):
        for u, v in G.edges():
            qc.cx(u, v)
            qc.rz(2 * gamma[layer], v)
            qc.cx(u, v)
        for q in range(n):
            qc.rx(2 * beta[layer], q)

    return qc


def initialize_angles(p: int, strategy: str,
                      rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize QAOA angles.
      random: uniform random in [0, 2pi] x [0, pi]
      linear_ramp: literature heuristic with small perturbation
    """
    if strategy == "random":
        return rng.uniform(0, 2 * np.pi, size=p), rng.uniform(0, np.pi, size=p)
    elif strategy == "linear_ramp":
        gamma = np.linspace(0.1, np.pi / 3, p) + rng.uniform(-0.1, 0.1, size=p)
        beta = np.linspace(np.pi / 3, 0.1, p) + rng.uniform(-0.1, 0.1, size=p)
        return np.clip(gamma, 0, 2 * np.pi), np.clip(beta, 0, np.pi)
    raise ValueError(f"Unknown strategy: {strategy}")
