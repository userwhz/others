from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from shadowgrouping.hamiltonian import random_hamiltonian, char_to_int
from shadowgrouping.measurement_schemes import Shadow_Grouping
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.weight_functions import Bernstein_bound


def _exact_ground_energy(ham: dict[str, float]) -> tuple[float, np.ndarray]:
    labels = list(ham.keys())
    coeffs = np.array(list(ham.values()))
    op = SparsePauliOp.from_list([(p[::-1], c) for p, c in ham.items()])
    mat = op.to_matrix()
    evalues, evectors = np.linalg.eigh(mat)
    idx = int(np.argmin(evalues))
    return float(evalues[idx]), evectors[:, idx]


def test_shadowgrouping_fc_random() -> None:
    """Integration test: ShadowGrouping FC mode on a random 4-qubit Hamiltonian."""
    nqubit = 4
    kterm = 20
    nshots = 3000
    epsilon = 0.1
    tolerance = 35.0  # 100-seed stats: mean=7.2, max=11.5, 3x max ≈ 35

    np.random.seed(42)

    ham = random_hamiltonian(nqubit, kterm)
    pauli_strings = list(ham.keys())
    weights = np.array(list(ham.values()))
    observables = np.array(
        [[char_to_int[c] for c in p] for p in pauli_strings], dtype=int
    )

    E_exact, state = _exact_ground_energy(ham)

    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(
        observables, weights, epsilon=epsilon,
        weight_function=wf(), commutation_mode="fc",
    )
    sampler = StateSampler(state)
    estimator = Energy_estimator(scheme, sampler, offset=0)

    estimator.propose_next_settings(nshots)
    estimator.measure()
    E_estimated = estimator.get_energy()

    error = abs(E_estimated - E_exact)
    assert error < tolerance, (
        f"Absolute error {error:.4f} exceeds tolerance {tolerance}.\n"
        f"  Exact (SparsePauliOp): {E_exact:.6f}\n"
        f"  ShadowGrouping FC est:  {E_estimated:.6f}"
    )


def test_shadowgrouping_fc_consistency() -> None:
    """FC mode produces finite energy estimates at various shot budgets."""
    nqubit = 4
    kterm = 20
    epsilon = 0.1

    np.random.seed(123)

    ham = random_hamiltonian(nqubit, kterm)
    weights = np.array(list(ham.values()))
    observables = np.array(
        [[char_to_int[c] for c in p] for p in ham], dtype=int
    )

    E_exact, state = _exact_ground_energy(ham)
    wf = Bernstein_bound(alpha=1)

    for nshots in [1000, 3000, 5000]:
        scheme = Shadow_Grouping(
            observables, weights, epsilon=epsilon,
            weight_function=wf(), commutation_mode="fc",
        )
        sampler = StateSampler(state)
        estimator = Energy_estimator(scheme, sampler, offset=0)
        estimator.propose_next_settings(nshots)
        estimator.measure()
        E_estimated = estimator.get_energy()
        error = abs(E_estimated - E_exact)
        assert np.isfinite(error), f"Non-finite error at nshots={nshots}: {error}"
