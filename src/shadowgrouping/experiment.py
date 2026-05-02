"""Batch experiment runner: m independent energy estimates from a single sampling pass."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .energy_estimator import Energy_estimator, StateSampler
from .hamiltonian import char_to_int
from .measurement_schemes import Shadow_Grouping
from .weight_functions import Bernstein_bound


def _exact_ground_energy(ham: dict[str, float]) -> tuple[float, np.ndarray]:
    op = SparsePauliOp.from_list([(p[::-1], c) for p, c in ham.items()])
    mat = op.to_matrix()
    evalues, evectors = np.linalg.eigh(mat)
    idx = int(np.argmin(evalues))
    return float(evalues[idx]), evectors[:, idx]


def rmse(
    ham: dict[str, float],
    shots: int,
    m: int = 30,
    epsilon: float = 0.1,
    commutation_mode: str = "qwc",
    seed: int | None = None,
) -> dict:
    """Estimate energy m times from one sampling pass and return stats including RMSE.

    Args:
        ham: Pauli Hamiltonian dict (e.g. {"XX": -0.5, "ZZ": 1.2}).
        shots: Total number of measurement settings to propose.
        m: Number of independent estimates (samples m*reps per setting).
        epsilon: ShadowGrouping convergence parameter.
        commutation_mode: "qwc" or "fc".
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: E_exact, E_mean, E_std, rmse, rmse_rel, estimates, shots, m.
    """
    if seed is not None:
        np.random.seed(seed)

    pauli_strings = list(ham.keys())
    weights = np.array(list(ham.values()))
    observables = np.array(
        [[char_to_int[c] for c in p] for p in pauli_strings], dtype=int
    )

    E_exact, state = _exact_ground_energy(ham)

    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(
        observables, weights,
        epsilon=epsilon,
        weight_function=wf(),
        commutation_mode=commutation_mode,
        max_support_qubits=None,
    )
    sampler = StateSampler(state)
    estimator = Energy_estimator(scheme, sampler, offset=0)

    estimator.propose_next_settings(shots)
    estimates = estimator.measure_batch(m)

    error = estimates - E_exact
    rmse_val = float(np.sqrt(np.mean(error ** 2)))

    return {
        "E_exact": E_exact,
        "E_mean": float(np.mean(estimates)),
        "E_std": float(np.std(estimates)),
        "rmse": rmse_val,
        "rmse_rel": float(rmse_val / abs(E_exact)) if E_exact != 0 else float("inf"),
        "estimates": estimates,
        "shots": shots,
        "m": m,
    }
