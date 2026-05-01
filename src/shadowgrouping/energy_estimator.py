import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from .hamiltonian import int_to_char, char_to_int


class StateSampler:
    def __init__(self, state, density_matrix=False):
        self.state = np.array(state)
        self.num_qubits = int(np.log2(self.state.shape[0]))
        self.density_matrix = bool(density_matrix or self.state.ndim == 2)
        assert len(self.state) == 2 ** self.num_qubits, (
            "State size has to be of size 2**N for some integer N."
        )
        if self.density_matrix:
            self._state = DensityMatrix(self.state)
        else:
            self._state = Statevector(self.state)

    def _evolve_and_sample(self, circuit, nshots):
        if self.density_matrix:
            evolved = self._state.evolve(circuit)
            probs = evolved.probabilities()
            indices = np.random.choice(len(probs), size=nshots, p=probs)
            bits = (
                (indices[:, None] & (1 << np.arange(self.num_qubits)[::-1])) > 0
            ).astype(int)
        else:
            evolved = self._state.evolve(circuit)
            memory = evolved.sample_memory(nshots)
            bits = np.array(
                [[int(b) for b in reversed(bitstring)] for bitstring in memory],
                dtype=int,
            )
        return bits

    def sample(self, meas_basis=None, nshots=1):
        if meas_basis is None:
            meas_basis = "Z" * self.num_qubits

        assert len(meas_basis) == self.num_qubits, (
            "Measurement basis has to be specified for each qubit."
        )
        circuit = QuantumCircuit(self.num_qubits)
        for i, s in enumerate(meas_basis):
            if s == "X":
                circuit.h(i)
            elif s == "Y":
                circuit.sdg(i)
                circuit.h(i)
        bits = self._evolve_and_sample(circuit, nshots)
        mask = np.array([s != "I" for s in meas_basis], dtype=int)[np.newaxis, :]
        return -2 * bits * mask + 1

    def index_to_string(self, index_list):
        pauli_string = ""
        for ind in np.array(index_list, dtype=int):
            assert ind in range(4), "Elements of index_list have to be in {0,1,2,3}."
            pauli_string += int_to_char[ind]
        return pauli_string


class Energy_estimator:
    def __init__(self, measurement_scheme, state, offset=0):
        assert measurement_scheme.num_qubits == state.num_qubits, (
            "Measurement and state scheme do not match in qubit number."
        )
        self.measurement_scheme = measurement_scheme
        self.state = state
        self.offset = offset
        self.settings_dict = {}
        self.settings_buffer = {}
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N = np.zeros(len(self.running_avgs), dtype=int)
        self.num_settings = 0
        self.num_outcomes = 0
        self.measurement_scheme.reset()

    def reset(self):
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N = np.zeros(len(self.running_avgs), dtype=int)
        self.settings_dict = {}
        self.settings_buffer = {}
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()

    def clear_outcomes(self):
        self.settings_buffer = self.settings_dict.copy()
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N = np.zeros(len(self.running_avgs), dtype=int)
        self.num_outcomes = 0

    def __setting_to_str(self, p):
        out = ""
        for c in p:
            out += int_to_char[c]
        return out

    def __settings_to_dict(self, settings):
        unique_settings, counts = np.unique(settings, axis=0, return_counts=True)
        for setting, nshots in zip(unique_settings, counts):
            paulistring = self.__setting_to_str(setting)
            for diction in (self.settings_dict, self.settings_buffer):
                val = diction.get(paulistring, 0)
                diction[paulistring] = nshots + val

    def propose_next_settings(self, num_steps=1):
        settings = []
        for _ in range(num_steps):
            p, _ = self.measurement_scheme.find_setting()
            settings.append(p)
        settings = np.array(settings)
        self.num_settings += num_steps
        self.__settings_to_dict(settings)

    def measure(self):
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("No new settings to measure. Call propose_next_settings() first.")
            return

        for setting, reps in self.settings_buffer.items():
            samples = self.state.sample(meas_basis=setting, nshots=reps)
            for i, o in enumerate(self.measurement_scheme.obs):
                if not self.measurement_scheme.is_hit(o, [char_to_int[c] for c in setting]):
                    continue
                mask = np.zeros(samples.shape, dtype=int)
                mask += (o == 0)[np.newaxis, :]
                temp = samples.copy()
                temp[mask.astype(bool)] = 1
                self.running_avgs[i] = (
                    self.running_avgs[i] * self.running_N[i] + np.prod(temp, axis=1).sum()
                ) / (self.running_N[i] + reps)
                self.running_N[i] += reps

        self.num_outcomes = self.num_settings
        self.settings_buffer = {}

    def get_energy(self):
        energy = np.sum(self.measurement_scheme.w * self.running_avgs)
        return energy + self.offset
