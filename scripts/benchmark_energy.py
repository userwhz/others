"""Benchmark ShadowGrouping energy estimation with loaded state and Hamiltonian.

Usage:
    uv run python scripts/benchmark_energy.py state_14q.npy hamiltonian_H2O_sto3g_14.json --repeat 20 --shots 572 2038 3845
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from shadowgrouping import Shadow_Grouping, Energy_estimator, StateSampler, Bernstein_bound, char_to_int

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_hamiltonian(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        sys.exit(f"Hamiltonian file not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_state(name: str) -> np.ndarray:
    path = DATA_DIR / name
    if not path.exists():
        sys.exit(f"State file not found: {path}")
    return np.load(path)


def compute_exact_energy(ham: dict, psi: np.ndarray) -> float:
    op = SparsePauliOp.from_list([(p[::-1], c) for p, c in ham.items()])
    return float(Statevector(psi).expectation_value(op).real)


def run_benchmark(
    state_file: str,
    ham_file: str,
    repeat: int,
    shots: int,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> int:
    """Run the benchmark for a single shots value. Returns exit code."""
    if seed is not None:
        np.random.seed(seed)

    ham = load_hamiltonian(ham_file)
    psi = load_state(state_file)

    pauli_strings = list(ham.keys())
    weights = np.array(list(ham.values()))
    obs = np.array([[char_to_int[c] for c in p] for p in pauli_strings], dtype=int)

    nqubit = len(pauli_strings[0])
    E_exact = compute_exact_energy(ham, psi)

    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(obs, weights, epsilon=epsilon, weight_function=wf())
    sampler = StateSampler(psi)
    estimator = Energy_estimator(scheme, sampler, offset=0)

    epoch_energies = []
    for epoch in range(repeat):
        estimator.propose_next_settings(shots)
        estimator.measure()
        energy = estimator.get_energy()
        epoch_energies.append(energy)

    epoch_energies_arr = np.array(epoch_energies)
    rmse_val = float(np.sqrt(np.mean((epoch_energies_arr - E_exact) ** 2)))

    OUTPUT_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / f"benchmark_{nqubit}q_{shots}shots.log"

    lines = []
    lines.append(
        f"state={state_file} hamiltonian={ham_file} "
        f"nqubit={nqubit} repeat={repeat} shots={shots} "
        f"E_exact={E_exact:.10f}"
    )
    for i, e in enumerate(epoch_energies):
        lines.append(f"epoch {i + 1}: {e:.10f}")
    lines.append(f"RMSE: {rmse_val:.10f}")

    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[shots={shots}] done, RMSE={rmse_val:.6f}, log={log_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ShadowGrouping energy estimation."
    )
    parser.add_argument("state", help="State .npy filename in data/ (e.g. state_14q.npy)")
    parser.add_argument("hamiltonian", help="Hamiltonian .json filename in data/")
    parser.add_argument("--repeat", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--shots", type=int, nargs="+", required=True,
                        help="Shots per epoch (can specify multiple values)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="ShadowGrouping epsilon (default: 0.1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    processes = []
    for shots_val in args.shots:
        p = multiprocessing.Process(
            target=run_benchmark,
            args=(args.state, args.hamiltonian, args.repeat, shots_val, args.epsilon, args.seed),
        )
        p.start()
        processes.append(p)
        print(f"Launched process for shots={shots_val} (pid={p.pid})")

    for p in processes:
        p.join()

    print("All done.")


if __name__ == "__main__":
    main()
