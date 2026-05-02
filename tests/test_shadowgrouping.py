import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from shadowgrouping.hamiltonian import random_hamiltonian, char_to_int
from shadowgrouping.measurement_schemes import Shadow_Grouping
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.weight_functions import Bernstein_bound


def _exact_ground_energy(ham: dict[str, float]) -> tuple[float, np.ndarray]:
    """Compute exact ground-state energy via SparsePauliOp diagonalization."""
    labels = list(ham.keys())
    coeffs = np.array(list(ham.values()))
    # Pauli strings in ham are little-endian (pos 0 = qubit 0).
    # SparsePauliOp uses big-endian (rightmost = qubit 0), so reverse.
    op = SparsePauliOp.from_list([(p[::-1], c) for p, c in ham.items()])
    mat = op.to_matrix()
    evalues, evectors = np.linalg.eigh(mat)
    idx = int(np.argmin(evalues))
    return float(evalues[idx]), evectors[:, idx]


def test_shadowgrouping_random() -> None:
    """Integration test: ShadowGrouping on a random 4-qubit Hamiltonian."""
    nqubit = 4
    kterm = 20
    nshots = 3000
    epsilon = 0.1
    tolerance = 1.2  # 1000-seed stats: max=1.00, p99.9=0.91

    np.random.seed(42)

    ham = random_hamiltonian(nqubit, kterm)
    pauli_strings = list(ham.keys())
    weights = np.array(list(ham.values()))
    observables = np.array(
        [[char_to_int[c] for c in p] for p in pauli_strings], dtype=int
    )

    # Theoretical ground truth via SparsePauliOp (qiskit builtin)
    E_exact, state = _exact_ground_energy(ham)

    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(observables, weights, epsilon=epsilon,
                             weight_function=wf())
    sampler = StateSampler(state)
    estimator = Energy_estimator(scheme, sampler, offset=0)

    estimator.propose_next_settings(nshots)
    estimator.measure()
    E_estimated = estimator.get_energy()

    error = abs(E_estimated - E_exact)
    assert error < tolerance, (
        f"Absolute error {error:.4f} exceeds tolerance {tolerance}.\n"
        f"  Exact (SparsePauliOp): {E_exact:.6f}\n"
        f"  ShadowGrouping est:    {E_estimated:.6f}"
    )
