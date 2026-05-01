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
            # qiskit bitstrings have qubit 0 as the rightmost bit;
            # reverse so column i corresponds to qubit i (matching qibo convention).
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

    def sample_with_transform(self, qubits, unitary, nshots=1):
        circuit = QuantumCircuit(self.num_qubits)
        if qubits is not None and len(qubits) > 0:
            circuit.unitary(unitary, list(map(int, qubits)))
        return self._evolve_and_sample(circuit, nshots)

    def index_to_string(self, index_list):
        pauli_string = ""
        for ind in np.array(index_list, dtype=int):
            assert ind in range(4), "Elements of index_list have to be in {0,1,2,3}."
            pauli_string += int_to_char[ind]
        return pauli_string

class Energy_estimator():
    """ Convenience class that holds both a measurement scheme and a StateSampler instance.
        The main workflow consists of proposing the next (few) measurement settings and measuring them in the respective bases.
        Furthermore, it tracks all measurement settings and their respective outcomes (of value +/-1 per qubit).
        Based on these values, the current energy estimate can be calculated.

        Inputs:
        - measurement_scheme, see class Measurement_Scheme and subclasses for information.
        - state, see class StateSampler.
        - Energy offset (defaults to 0) for the energy estimation.
          This consists of the identity term in the corresponding Hamiltonian decomposition.
    """
    def __init__(self,measurement_scheme,state,offset=0,spin_corr=None):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.offset       = offset
        # convenience counters to keep track of measurements settings and respective outcomes
        self.settings_dict = {}
        self.settings_buffer = {}
        self.group_dict = {}
        self.group_buffer = {}
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.num_settings = 0
        self.num_outcomes = 0
        self.measurement_scheme.reset()
        return

    def _uses_group_circuit_mode(self):
        return (
            hasattr(self.measurement_scheme, "get_group_plan")
            and getattr(self.measurement_scheme, "commutation_mode", "qwc") == "fc"
            and getattr(self.measurement_scheme, "group_plans", None) is not None
        )

    def reset(self):
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.settings_dict = {}
        self.settings_buffer = {}
        self.group_dict = {}
        self.group_buffer = {}
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()
        return

    def clear_outcomes(self):
        self.settings_buffer = self.settings_dict.copy()
        self.group_buffer = self.group_dict.copy()
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.num_outcomes = 0
        return

    def __setting_to_str(self,p):
        out = ""
        for c in p:
            out += int_to_char[c]
        return out

    def __settings_to_dict(self,settings):
        unique_settings, counts = np.unique(settings,axis=0,return_counts=True)
        for setting,nshots in zip(unique_settings,counts):
            paulistring = self.__setting_to_str(setting)
            for diction in (self.settings_dict,self.settings_buffer):
                val = diction.get(paulistring,0)
                diction[paulistring] = nshots + val
        return

    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        settings = []
        group_ids = []
        for i in range(num_steps):
            p, info = self.measurement_scheme.find_setting()
            settings.append(p)
            if self._uses_group_circuit_mode():
                group_ids.append(int(info["group_id"]))
        settings = np.array(settings)
        self.num_settings += num_steps
        self.__settings_to_dict(settings)
        if self._uses_group_circuit_mode() and len(group_ids) > 0:
            unique_gid, counts = np.unique(np.array(group_ids, dtype=int), return_counts=True)
            for gid, reps in zip(unique_gid, counts):
                gid = int(gid)
                reps = int(reps)
                for diction in (self.group_dict, self.group_buffer):
                    val = diction.get(gid, 0)
                    diction[gid] = val + reps
        return

    def measure(self):
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("Trying to measure more measurement settings than allocated. Please allocate measurements first by calling propose_next_settings() first.")
            return

        if self._uses_group_circuit_mode():
            for gid, reps in self.group_buffer.items():
                plan = self.measurement_scheme.get_group_plan(gid)
                samples = self.state.sample_with_transform(plan["qubits"], plan["unitary"], nshots=reps)
                q = plan["qubits"]
                if len(q) == 0:
                    basis_index = np.zeros(reps, dtype=int)
                else:
                    bits = samples[:, q].astype(int)
                    powers = (1 << np.arange(len(q)-1, -1, -1)).astype(int)
                    basis_index = np.sum(bits * powers[np.newaxis, :], axis=1)

                eigvals = plan["eigenvalues"]
                for local_i, obs_idx in enumerate(plan["obs_indices"]):
                    vals = eigvals[local_i, basis_index]
                    obs_idx = int(obs_idx)
                    self.running_avgs[obs_idx] = (
                        self.running_avgs[obs_idx] * self.running_N[obs_idx] + np.sum(vals)
                    ) / (self.running_N[obs_idx] + reps)
                    self.running_N[obs_idx] += reps

            self.num_outcomes = self.num_settings
            self.group_buffer = {}
            self.settings_buffer = {}
            return

        for setting,reps in self.settings_buffer.items():
            samples = self.state.sample(meas_basis=setting,nshots=reps)
            for i,o in enumerate(self.measurement_scheme.obs):
                if not self.measurement_scheme.is_hit(o,[char_to_int[c] for c in setting]):
                    continue
                mask = np.zeros(samples.shape,dtype=int)
                mask += (o == 0)[np.newaxis,:]
                temp = samples.copy()
                temp[mask.astype(bool)] = 1
                self.running_avgs[i] = ( self.running_avgs[i]*self.running_N[i] + np.prod(temp,axis=1).sum() ) / (self.running_N[i] + reps)
                self.running_N[i] += reps

        self.num_outcomes = self.num_settings
        self.settings_buffer = {}
        return

    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        energy = np.sum(self.measurement_scheme.w*self.running_avgs)
        return energy + self.offset
