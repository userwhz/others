from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliOp, SummedOp

char_to_int: dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}
int_to_char: dict[int, str] = {item: key for key, item in char_to_int.items()}


class Hamiltonian:
    def __init__(self, weights: np.ndarray, observables: list[str]) -> None:
        self.weights: np.ndarray = weights
        self.observables: list[str] = observables

    def SummedOp(self) -> SummedOp:
        paulis = []
        for P, coeff_P in zip(self.observables, self.weights):
            paulis.append(coeff_P * PauliOp(Pauli(P[::-1])))
        return SummedOp(paulis)

    def ground(self) -> tuple[float, np.ndarray]:
        mat = self.SummedOp().to_matrix()
        evalues, evectors = np.linalg.eigh(mat)
        index = np.argmin(evalues)
        return float(evalues[index]), evectors[:, index]
