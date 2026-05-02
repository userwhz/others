import numpy as np
import pytest
from shadowgrouping.hamiltonian import random_hamiltonian, Hamiltonian, char_to_int
from shadowgrouping.measurement_schemes import Shadow_Grouping
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.weight_functions import Bernstein_bound


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

    H = Hamiltonian(weights, pauli_strings)
    E_GS, state = H.ground()

    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(observables, weights, epsilon=epsilon,
                             weight_function=wf())
    sampler = StateSampler(state)
    estimator = Energy_estimator(scheme, sampler, offset=0)

    estimator.propose_next_settings(nshots)
    estimator.measure()
    estimated_energy = estimator.get_energy()

    error = abs(estimated_energy - E_GS)
    assert error < tolerance, (
        f"Energy estimate error {error:.4f} exceeds tolerance {tolerance}.\n"
        f"Exact: {E_GS:.6f}, Estimated: {estimated_energy:.6f}"
    )
