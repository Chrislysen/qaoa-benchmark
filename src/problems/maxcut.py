"""Max-Cut problem: graph generation, brute-force solver, cost function."""

import networkx as nx
import numpy as np


def generate_maxcut_graph(n_nodes: int = 10, edge_prob: float = 0.5,
                          seed: int = 42) -> nx.Graph:
    """Generate a random Erdos-Renyi graph for Max-Cut."""
    return nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)


def brute_force_maxcut(G: nx.Graph) -> tuple[int, list[int]]:
    """Exact Max-Cut by brute force enumeration. Feasible for n <= 20."""
    n = G.number_of_nodes()
    best_cut, best_assignment = 0, None
    for bits in range(2**n):
        assignment = [(bits >> i) & 1 for i in range(n)]
        cut = sum(1 for u, v in G.edges() if assignment[u] != assignment[v])
        if cut > best_cut:
            best_cut = cut
            best_assignment = assignment
    return best_cut, best_assignment


def maxcut_cost(bitstring: str, G: nx.Graph) -> int:
    """Evaluate the cut value for a given bitstring."""
    return sum(1 for u, v in G.edges() if bitstring[u] != bitstring[v])


def random_assignment_expected_ratio(G: nx.Graph, optimal_cut: int) -> float:
    """Expected approximation ratio of a random binary assignment."""
    return G.number_of_edges() / 2 / optimal_cut
