# QAOA Parameter Optimization Benchmark: Do Budget-Aware Classical Optimizers Improve Variational Quantum Circuits?

**Short answer: No.** Shallow QAOA parameter landscapes are too smooth and low-dimensional for constrained optimization techniques to outperform standard methods.

---

## Problem Statement

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical algorithm where a classical optimizer tunes circuit parameters (γ, β angles) by repeatedly evaluating a quantum circuit. Each evaluation costs real QPU time. On IBM Quantum's free tier (10 minutes/month on 100+ qubit processors), a typical QAOA optimization run with 100 iterations × 1000 shots can exhaust the entire monthly budget on a single problem instance.

This creates an expensive-evaluation, noisy-objective optimization problem under a hard budget constraint — a setting where budget-aware optimizers should, in theory, outperform generic methods.

## Hypothesis

We hypothesized that a budget-aware classical optimizer — adapted from Thermal Budget Annealing (TBA), a simulated annealing + TPE hybrid designed for crash-heavy ML deployment configuration spaces — would find better QAOA circuit parameters in fewer evaluations than standard optimizers (COBYLA, SPSA), making QAOA more practical under tight QPU budgets.

The reasoning: TBA was designed for search spaces where some configurations "crash" (produce garbage results), evaluations are expensive and noisy, and the budget is severely limited. The QAOA parameter search under real hardware constraints shares all three properties.

## Methodology

### Quantum Setup
- **Algorithm:** QAOA for Max-Cut on random graphs
- **Problem sizes:** 6, 8, 10 nodes (random 3-regular graphs)
- **QAOA depths:** p=1 (2 parameters), p=2 (4 parameters)
- **Framework:** Qiskit + qiskit-ibm-runtime
- **Execution environments:** Ideal statevector simulator, shot-based simulator (1024 shots)

### Classical Optimizers Compared
| Optimizer | Type | Description |
|-----------|------|-------------|
| **COBYLA** | Gradient-free | Standard constrained optimizer; widely used in QAOA literature |
| **SPSA** | Stochastic perturbation | Designed for noisy objective functions; common in variational quantum algorithms |
| **TBA-SA** | Budget-aware SA | Simulated annealing variant with thermal budget scheduling, adapted from deployment optimization |
| **Random Search** | Baseline | Uniform random sampling of parameter space |

### Evaluation Protocol
- **Metric:** Approximation ratio (QAOA solution quality / known optimal cut value)
- **Budget sweep:** Performance measured at 10, 25, 50, 100, and 200 circuit evaluations
- **Seeds:** 10 random graph instances per problem size, 5 optimizer seeds per instance
- **Brute-force reference:** Exact Max-Cut computed for all graph sizes to establish ground truth

## Results

### Approximation Ratio vs. Evaluation Budget (p=1, 8-node graphs, shot-based simulator)

| Evaluations | COBYLA | SPSA | TBA-SA | Random |
|-------------|--------|------|--------|--------|
| 10          | 0.82   | 0.71 | 0.78   | 0.65   |
| 25          | 0.91   | 0.84 | 0.88   | 0.73   |
| 50          | 0.95   | 0.91 | 0.93   | 0.79   |
| 100         | 0.97   | 0.95 | 0.95   | 0.83   |
| 200         | 0.97   | 0.96 | 0.96   | 0.86   |

### Key Finding

**COBYLA consistently reaches near-optimal solutions fastest.** At p=1 (2 parameters), COBYLA converges to >0.95 approximation ratio within ~50 evaluations. TBA-SA provides no meaningful advantage at any budget level. The gap narrows at higher budgets, but COBYLA is never worse.

At p=2 (4 parameters), the pattern holds. The parameter space remains smooth and low-dimensional enough that COBYLA's simplex-based search is well-matched to the problem. TBA-SA's exploration-exploitation scheduling — designed for high-dimensional spaces with crash regions — finds no crash regions to navigate around and no complex structure to exploit.

### Why TBA-SA Doesn't Help Here

The QAOA parameter landscape for Max-Cut at shallow depths has three properties that neutralize budget-aware optimization:

1. **Low dimensionality.** p=1 has 2 parameters; p=2 has 4. TBA's advantage emerges in spaces with tens or hundreds of dimensions and mixed variable types. A 2D continuous landscape is trivial for COBYLA.

2. **Smoothness.** The QAOA cost landscape for Max-Cut is relatively smooth with broad basins. There are no "crash regions" analogous to OOM failures in deployment configuration — every parameter setting produces a valid (if suboptimal) result.

3. **Noise is uniform.** Shot noise affects all parameter settings equally. TBA's crash-awareness mechanism has nothing to trigger on because there is no region-dependent failure mode.

## Why This Matters

Negative results are underreported in quantum computing research. The community benefits from knowing:

- **Standard optimizers are sufficient for shallow QAOA.** Practitioners do not need specialized optimization machinery for p=1 or p=2 QAOA on small-to-moderate problem sizes. COBYLA works.

- **The classical optimization bottleneck in QAOA is not the optimizer.** The real bottlenecks are hardware noise, limited circuit depth, and qubit connectivity — not the parameter search strategy. Efforts to improve QAOA should focus on error mitigation, circuit design, and hardware improvements rather than classical optimizer engineering.

- **Budget-aware optimization has a domain boundary.** Techniques designed for crash-heavy, high-dimensional, mixed-variable spaces (like ML deployment configuration) do not automatically transfer to smooth, low-dimensional, continuous spaces. The problem structure matters more than the budget constraint.

- **Reproducible negative results prevent wasted effort.** Other researchers considering the same "use a better classical optimizer for QAOA" approach can reference this work and redirect their effort.

## Reproducing This Work

### Requirements
```
pip install qiskit qiskit-ibm-runtime qiskit-algorithms networkx numpy scipy matplotlib
```

### Run the Benchmark
```bash
# Full benchmark (simulator only, no QPU needed)
python run_benchmark.py

# Single quick test
python run_benchmark.py --nodes 6 --depth 1 --evaluations 50 --seeds 3
```

### Generate Figures
```bash
python plot_results.py --input results/ --output figures/
```

## Limitations

- **Problem sizes are small** (6–10 nodes). At larger scales or deeper circuits, the parameter landscape may become more complex and budget-aware optimization could become relevant. We could not test this due to simulator runtime constraints.
- **Max-Cut only.** Other combinatorial problems (e.g., portfolio optimization, graph coloring) may have different QAOA landscape properties.
- **No real QPU results.** All experiments used simulators. Real hardware noise could change the landscape enough to create the crash-like regions where TBA-SA excels — but the 10-minute monthly QPU budget made systematic hardware experiments infeasible.
- **TBA-SA was adapted, not redesigned.** A purpose-built quantum parameter optimizer might perform differently than an adapted deployment optimizer.

## Context

This project followed directly from [Thermal Budget Annealing](https://github.com/Chrislysen/deploy-agent) (TBA), which optimizes ML model deployment configurations under crash-heavy hardware constraints. The hypothesis was that QAOA's expensive, noisy parameter search would benefit from the same budget-aware techniques. The negative result clarifies that these are fundamentally different optimization landscapes.

## Citation

If you reference this work:
```
@misc{lysen2026qaoa,
  author = {Lysen, Christian},
  title = {QAOA Parameter Optimization Benchmark: Budget-Aware Classical Optimizers vs Standard Methods},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Chrislysen/qaoa-benchmark}
}
```

## License

MIT
