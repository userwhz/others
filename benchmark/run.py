"""Benchmark ShadowGrouping on a 10-qubit random Hamiltonian.

Sweeps shot budgets, records runtime and energy estimation error.
Results are written to benchmark/results/.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from shadowgrouping.hamiltonian import random_hamiltonian, char_to_int
from shadowgrouping.measurement_schemes import Shadow_Grouping
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.weight_functions import Bernstein_bound


@dataclass
class RunResult:
    seed: int
    nshots: int
    E_exact: float
    E_estimated: float
    abs_error: float
    time_setup: float
    time_propose: float
    time_measure: float
    time_total: float


@dataclass
class BenchmarkResult:
    nqubit: int
    kterm: int
    runs: list[RunResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Benchmark: {self.nqubit} qubits, {self.kterm} Pauli terms"]
        lines.append(f"Runs: {len(self.runs)}")
        lines.append("")
        lines.append(f"{'nshots':>8}  {'n':>4}  {'err_mean':>10}  {'err_std':>10}  {'err_max':>10}  {'time_mean':>10}  {'time_std':>10}")
        lines.append("-" * 80)

        for nshots in sorted(set(r.nshots for r in self.runs)):
            group = [r for r in self.runs if r.nshots == nshots]
            errors = [r.abs_error for r in group]
            times = [r.time_total for r in group]
            lines.append(
                f"{nshots:>8}  {len(group):>4}  "
                f"{np.mean(errors):>10.4f}  {np.std(errors):>10.4f}  {np.max(errors):>10.4f}  "
                f"{np.mean(times):>10.2f}  {np.std(times):>10.2f}"
            )
        return "\n".join(lines)


def run_single(nqubit: int, kterm: int, nshots: int, epsilon: float, seed: int) -> RunResult:
    t0 = time.perf_counter()

    np.random.seed(seed)
    ham = random_hamiltonian(nqubit, kterm)
    pstrings = list(ham.keys())
    weights = np.array(list(ham.values()))
    obs = np.array([[char_to_int[c] for c in p] for p in pstrings], dtype=int)

    # Exact ground truth via SparsePauliOp
    op = SparsePauliOp.from_list([(p[::-1], c) for p, c in ham.items()])
    mat = op.to_matrix()
    evalues, evectors = np.linalg.eigh(mat)
    idx = int(np.argmin(evalues))
    E_exact = float(evalues[idx])
    state = evectors[:, idx]
    t_setup = time.perf_counter() - t0

    # ShadowGrouping
    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(obs, weights, epsilon=epsilon, weight_function=wf())
    sampler = StateSampler(state)
    estimator = Energy_estimator(scheme, sampler, offset=0)

    t1 = time.perf_counter()
    estimator.propose_next_settings(nshots)
    t_propose = time.perf_counter() - t1

    t2 = time.perf_counter()
    estimator.measure()
    t_measure = time.perf_counter() - t2

    E_estimated = estimator.get_energy()
    t_total = time.perf_counter() - t0

    return RunResult(
        seed=seed,
        nshots=nshots,
        E_exact=E_exact,
        E_estimated=E_estimated,
        abs_error=abs(E_estimated - E_exact),
        time_setup=t_setup,
        time_propose=t_propose,
        time_measure=t_measure,
        time_total=t_total,
    )


def main() -> None:
    nqubit = 10
    kterm = 50
    epsilon = 0.1
    n_seeds = 10
    shot_budgets: list[int] = [1000, 2000, 5000, 10000, 20000]

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"benchmark_{nqubit}q_{timestamp}.csv"
    txt_path = results_dir / f"benchmark_{nqubit}q_{timestamp}.txt"

    benchmark = BenchmarkResult(nqubit=nqubit, kterm=kterm)
    total_runs = len(shot_budgets) * n_seeds

    print(f"Running {total_runs} trials ({nqubit}q, {kterm} terms, {n_seeds} seeds × {len(shot_budgets)} shot budgets)...")
    print()

    n = 0
    for nshots in shot_budgets:
        for seed in range(n_seeds):
            n += 1
            result = run_single(nqubit, kterm, nshots, epsilon, seed)
            benchmark.runs.append(result)
            print(
                f"  [{n:>3}/{total_runs}]  "
                f"shots={nshots:>5}  seed={seed:>2}  "
                f"error={result.abs_error:.4f}  "
                f"time={result.time_total:.1f}s"
            )

    summary = benchmark.summary()
    print()
    print(summary)

    # Write text summary
    txt_path.write_text(summary + "\n")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "nshots", "E_exact", "E_estimated", "abs_error",
            "time_setup", "time_propose", "time_measure", "time_total",
        ])
        writer.writeheader()
        for r in benchmark.runs:
            writer.writerow({
                "seed": r.seed, "nshots": r.nshots,
                "E_exact": r.E_exact, "E_estimated": r.E_estimated,
                "abs_error": r.abs_error,
                "time_setup": r.time_setup, "time_propose": r.time_propose,
                "time_measure": r.time_measure, "time_total": r.time_total,
            })

    print(f"\nResults written to:")
    print(f"  {txt_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
