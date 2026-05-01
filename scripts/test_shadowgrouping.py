"""Test ShadowGrouping with a random Hamiltonian.

Generates a random Hamiltonian, computes the exact ground-state energy via
diagonalization, then estimates the energy via ShadowGrouping with 3000
measurement rounds. The estimated value should agree with the exact value
within a tolerance.
"""
import numpy as np
from shadowgrouping.hamiltonian import random_hamiltonian, Hamiltonian, char_to_int
from shadowgrouping.measurement_schemes import Shadow_Grouping
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.weight_functions import Bernstein_bound


def test_shadowgrouping_random():
    nqubit = 4
    kterm = 20
    nshots = 3000
    epsilon = 0.1
    seed = 42

    np.random.seed(seed)

    # 1. Generate random Hamiltonian
    ham = random_hamiltonian(nqubit, kterm)

    # 2. Convert to ShadowGrouping format
    pauli_strings = list(ham.keys())
    weights = np.array(list(ham.values()))
    observables = np.array(
        [[char_to_int[c] for c in p] for p in pauli_strings], dtype=int
    )

    # 3. Get exact ground-state energy via diagonalization
    H = Hamiltonian(weights, pauli_strings)
    E_GS, state = H.ground()
    print(f"Exact ground-state energy: {E_GS:.6f}")

    # 4. Set up ShadowGrouping measurement scheme
    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(observables, weights, epsilon=epsilon,
                             weight_function=wf())

    # 5. Set up StateSampler with ground state and Energy_estimator
    sampler = StateSampler(state)
    estimator = Energy_estimator(scheme, sampler, offset=0)

    # 6. Propose settings and measure
    estimator.propose_next_settings(nshots)
    estimator.measure()
    estimated_energy = estimator.get_energy()

    print(f"Estimated energy:       {estimated_energy:.6f}")
    print(f"Absolute error:         {abs(estimated_energy - E_GS):.6f}")

    # 7. Assert the estimate is reasonably close
    error = abs(estimated_energy - E_GS)
    # With 3000 shots on a 4-qubit system, error should be well under 2.0
    assert error < 2.0, (
        f"Energy estimate error {error:.4f} exceeds tolerance 2.0.\n"
        f"Exact: {E_GS:.6f}, Estimated: {estimated_energy:.6f}"
    )
    print("PASSED: ShadowGrouping estimate matches exact diagonalization.")


if __name__ == "__main__":
    test_shadowgrouping_random()
