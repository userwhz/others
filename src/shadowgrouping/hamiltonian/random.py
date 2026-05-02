import numpy as np

int_to_char: dict[int, str] = {0: "I", 1: "X", 2: "Y", 3: "Z"}


def random_hamiltonian(nqubit: int, kterm: int) -> dict[str, float]:
    """Generate a random Pauli Hamiltonian.

    Args:
        nqubit: Number of qubits.
        kterm: Number of Pauli terms (excluding identity).

    Returns:
        dict mapping Pauli strings (e.g. "XXYZ") to float coefficients.
    """
    max_terms = 4**nqubit - 1  # exclude all-identity
    if kterm > max_terms:
        raise ValueError(
            f"kterm ({kterm}) exceeds maximum unique Pauli strings "
            f"({max_terms}) for {nqubit} qubits."
        )

    seen: set[str] = set()
    coeffs = np.random.randn(kterm)
    paulis: list[str] = []

    while len(paulis) < kterm:
        indices = np.random.randint(0, 4, size=nqubit)
        if np.all(indices == 0):
            continue  # skip all-identity
        key = "".join(int_to_char[i] for i in indices)
        if key not in seen:
            seen.add(key)
            paulis.append(key)

    return dict(zip(paulis, coeffs))
