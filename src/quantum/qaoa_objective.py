"""QAOA objective functions: statevector and shot-based evaluation."""

import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.problems.maxcut import maxcut_cost
from src.quantum.qaoa_circuit import build_qaoa_circuit


def qaoa_statevector_objective(params: np.ndarray, G: nx.Graph, p: int) -> float:
    """Exact statevector evaluation. Returns NEGATIVE expected cut."""
    gamma, beta = params[:p], params[p:]
    qc = build_qaoa_circuit(G, p, gamma.tolist(), beta.tolist())
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    n = G.number_of_nodes()

    expected_cut = sum(
        prob * maxcut_cost(format(idx, f"0{n}b"), G)
        for idx, prob in enumerate(probs) if prob > 1e-15
    )
    return -expected_cut


def qaoa_shot_objective(params: np.ndarray, G: nx.Graph, p: int,
                        shots: int = 1024, optimization_level: int = 1,
                        noise_model: NoiseModel = None) -> float:
    """Shot-based evaluation (optionally noisy). Returns NEGATIVE expected cut."""
    gamma, beta = params[:p], params[p:]
    qc = build_qaoa_circuit(G, p, gamma.tolist(), beta.tolist())
    qc.measure_all()

    sim = AerSimulator(**({"noise_model": noise_model} if noise_model else {}))
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=sim)
    transpiled = pm.run(qc)
    result = sim.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    n = G.number_of_nodes()
    total = sum(counts.values())
    expected_cut = sum(
        (count / total) * maxcut_cost(bs[::-1], G)
        for bs, count in counts.items()
    )
    return -expected_cut


def evaluate_pipeline(G: nx.Graph, config: dict,
                      noise_model: NoiseModel) -> float:
    """Evaluate a full pipeline config. Returns POSITIVE expected cut."""
    params = np.concatenate([config["gamma"], config["beta"]])
    return -qaoa_shot_objective(
        params, G, config["p"], shots=config["shots"],
        optimization_level=config["optimization_level"],
        noise_model=noise_model
    )
