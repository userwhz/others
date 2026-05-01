import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliOp, SummedOp

char_to_int = {"I": 0, "X": 1, "Y": 2, "Z": 3}
int_to_char = {item: key for key, item in char_to_int.items()}


class Hamiltonian:
    def __init__(self, weights, observables):
        self.weights = weights
        self.observables = observables

    def SummedOp(self):
        paulis = []
        for P, coeff_P in zip(self.observables, self.weights):
            # Reverse P: codebase uses little-endian (pos i = qubit i),
            # but qiskit Pauli uses big-endian (pos 0 = qubit N-1).
            paulis.append(coeff_P * PauliOp(Pauli(P[::-1])))
        return SummedOp(paulis)

    def ground(self):
        mat = self.SummedOp().to_matrix()
        evalues, evectors = np.linalg.eigh(mat)
        index = np.argmin(evalues)
        return evalues[index], evectors[:, index]
