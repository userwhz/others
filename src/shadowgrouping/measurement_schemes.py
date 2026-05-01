import numpy as np
from itertools import product
from time import time
from typing import Optional

##########################################################################################
### Helper functions #####################################################################
##########################################################################################
def hit_by(O,P):
    """ Returns whether o is hit by p """
    for o,p in zip(O,P):
        if not (o==0 or p==0 or o==p):
            return False
    return True

def pauli_commute(O0, O1):
    """Returns whether two Pauli strings commute globally."""
    anti_count = 0
    for a, b in zip(O0, O1):
        if a == 0 or b == 0 or a == b:
            continue
        anti_count += 1
    return anti_count % 2 == 0

def qwc_compatible(O0, O1):
    """Returns whether two Pauli strings are qubit-wise commuting (QWC)."""
    for a, b in zip(O0, O1):
        if not (a == 0 or b == 0 or a == b):
            return False
    return True

def hit_by_mode(O, P, commutation_mode="qwc"):
    """Returns whether observable O is considered hit by setting P under the selected mode."""
    if commutation_mode == "qwc":
        return hit_by(O, P)
    if commutation_mode == "fc":
        return pauli_commute(O, P)
    raise ValueError("Unknown commutation_mode '{}'. Use 'qwc' or 'fc'.".format(commutation_mode))

def setting_to_str(arr):
    out = ""
    for a in np.array(arr).flatten():
        out += str(a)
    return out

# equation 6 from manuscript
N_delta = lambda delta: 4*(2*np.sqrt(-np.log(delta))+1)**2

def _pauli_single_matrix(ind):
    if ind == 0:
        return np.array([[1, 0], [0, 1]], dtype=complex)
    if ind == 1:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if ind == 2:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if ind == 3:
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError("Unknown Pauli index {}".format(ind))

def pauli_row_to_matrix(row):
    mat = np.array([[1]], dtype=complex)
    for ind in row:
        mat = np.kron(mat, _pauli_single_matrix(int(ind)))
    return mat

def build_commuting_groups(observables, commutation_mode="qwc", max_support_qubits=None):
    """Build commuting groups via greedy coloring on the non-commutation graph.

    If max_support_qubits is set (int), additionally enforce that the union support
    (non-identity qubits) of observables within a group does not exceed this limit.
    This is critical for FC "joint measurement" implementations that rely on explicitly
    constructing 2^k x 2^k unitaries.
    """
    if commutation_mode not in ("qwc", "fc"):
        raise ValueError("commutation_mode has to be either 'qwc' or 'fc'.")
    n = len(observables)
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        oi = observables[i]
        for j in range(i + 1, n):
            compatible = qwc_compatible(oi, observables[j]) if commutation_mode == "qwc" else pauli_commute(oi, observables[j])
            if not compatible:
                neighbors[i].add(j)
                neighbors[j].add(i)

    order = sorted(range(n), key=lambda idx: len(neighbors[idx]), reverse=True)
    colors = [-1] * n
    for idx in order:
        used = {colors[nn] for nn in neighbors[idx] if colors[nn] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[idx] = color

    num_colors = max(colors) + 1 if colors else 0
    groups = [[] for _ in range(num_colors)]
    for obs_idx, color in enumerate(colors):
        groups[color].append(obs_idx)

    groups = [np.array(g, dtype=int) for g in groups if len(g) > 0]

    if max_support_qubits is None:
        return groups

    max_support_qubits = int(max_support_qubits)
    if max_support_qubits <= 0:
        raise ValueError("max_support_qubits must be a positive integer.")

    # Post-process: split any group whose union support exceeds max_support_qubits.
    split_groups = []
    for g in groups:
        bins = []
        bin_supports = []
        for idx in g:
            o = observables[int(idx)]
            o_support = set(np.where(o != 0)[0].tolist())
            placed = False
            for b_i in range(len(bins)):
                if len(bin_supports[b_i] | o_support) > max_support_qubits:
                    continue
                ok = True
                for jdx in bins[b_i]:
                    compatible = (
                        qwc_compatible(o, observables[int(jdx)])
                        if commutation_mode == "qwc"
                        else pauli_commute(o, observables[int(jdx)])
                    )
                    if not compatible:
                        ok = False
                        break
                if ok:
                    bins[b_i].append(int(idx))
                    bin_supports[b_i] |= o_support
                    placed = True
                    break
            if not placed:
                bins.append([int(idx)])
                bin_supports.append(set(o_support))
        for b in bins:
            split_groups.append(np.array(b, dtype=int))

    return split_groups

def build_fc_group_plans(observables, groups, max_support_qubits=None):
    """Build per-group basis-change plans for FC grouped measurement."""
    plans = {}
    for gid, group in enumerate(groups):
        obs_indices = np.array(group, dtype=int)
        support_mask = np.any(observables[obs_indices] != 0, axis=0)
        qubits = np.where(support_mask)[0]

        if max_support_qubits is not None and len(qubits) > int(max_support_qubits):
            raise MemoryError(
                "FC group {} spans {} qubits, exceeding max_support_qubits={}."
                " Reduce group support size or change joint-measurement implementation.".format(
                    gid, len(qubits), int(max_support_qubits)
                )
            )

        if len(qubits) == 0:
            plans[gid] = {
                "group_id": gid,
                "obs_indices": obs_indices,
                "qubits": np.array([], dtype=int),
                "unitary": np.array([[1]], dtype=complex),
                "eigenvalues": np.ones((len(obs_indices), 1), dtype=int),
            }
            continue

        pauli_mats = [pauli_row_to_matrix(observables[idx][qubits]) for idx in obs_indices]
        basis = None
        for trial in range(8):
            coeffs = np.array([np.cos((k + 1) * (trial + 1)) for k in range(len(pauli_mats))], dtype=float)
            H = np.zeros_like(pauli_mats[0], dtype=complex)
            for c, mat in zip(coeffs, pauli_mats):
                H += c * mat
            _, vecs = np.linalg.eigh(H)

            ok = True
            for mat in pauli_mats:
                d = vecs.conj().T @ mat @ vecs
                off = d - np.diag(np.diag(d))
                if np.max(np.abs(off)) > 1e-7:
                    ok = False
                    break
            if ok:
                basis = vecs
                break

        if basis is None:
            basis = np.eye(pauli_mats[0].shape[0], dtype=complex)

        eigenvalues = []
        for mat in pauli_mats:
            d = basis.conj().T @ mat @ basis
            vals = np.real(np.diag(d))
            vals = np.where(vals >= 0, 1, -1).astype(int)
            eigenvalues.append(vals)

        plans[gid] = {
            "group_id": gid,
            "obs_indices": obs_indices,
            "qubits": qubits.astype(int),
            "unitary": basis.conj().T,
            "eigenvalues": np.array(eigenvalues, dtype=int),
        }
    return plans

##########################################################################################
### Measurement scheme base class ########################################################
##########################################################################################

class Measurement_scheme:
    """ Parent class for measurement schemes. Requires
        observables: Array of shape (num_obs x num_qubits) with entries in {0,1,2,3} (the Pauli operators)
        weights:     Array of shape (num_obs) with the corresponding weight in the Hamiltonian decomposition.
                     Array is flattened upon input.
        epsilon:     Absolute error threshold, see child methods for an individual interpretation.
    """

    def __init__(self,observables,weights,epsilon):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"

        self.obs           = observables
        self.num_obs       = M
        self.num_qubits    = n
        self.w             = weights
        self.eps           = epsilon
        self.scheme_params = {"eps": epsilon, "num_obs": M}
        self.N_hits        = np.zeros(M,dtype=int)
        self.is_adaptive   = False
        self.commutation_mode = "qwc"

        return

    def is_hit(self, observable, setting):
        return hit_by_mode(observable, setting, self.commutation_mode)

    def find_setting(self):
        pass

    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return

    def get_epsilon_sys_stat(self,delta):
        """ Applies the truncation strategy (see truncate() for details) and returns the corresponding epsilon values for the
            systematic and the statistical error, respectively. Does not alter the scheme in-place, compared to truncate() would do.
        """
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit)
        if np.sum(keep) == 0:
            eps_syst = np.sum(np.abs(self.w))
            eps_stat = 0
        elif np.sum(keep) == len(keep):
            eps_syst = 0
            eps_stat = self.get_epsilon_Bernstein(delta)
        else:
            w, N = self.w, self.N_hits
            self.w = self.w[keep]
            self.N_hits = self.N_hits[keep]
            eps_syst = np.sum(np.abs(w[np.bitwise_not(keep)]))
            eps_stat = self.get_epsilon_Bernstein(delta)
            self.w = w
            self.N_hits = N
        return eps_syst, eps_stat

    def truncate(self,delta):
        """ Truncation function to apply the truncation criterion given a certain inconfidence level delta.
            Assumes that scheme has called the function find_setting() sufficiently often.
            Truncates all observables that fulfill the truncation criterion and save the sum of their absolute coefficient values.
            Returns the resulting introduced systematic error epsilon.
        """
        N_unmeasured = np.sum(self.N_hits == 0)
        if N_unmeasured > 0:
            print("Warning! {} observable(s) have not been measured at least once.".format(N_unmeasured))
            print("If you have set alpha large, this can result in a non-optimal truncation.")
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit)
        if np.sum(keep) == 0:
            print("No observable reached the threshold. Ensure that you have sampled often enough or provide a smaller delta!")
            print("Scheme unaltered.")
            return 0
        if np.sum(keep) == len(keep):
            print("Nothing had to be truncated.")
            return 0
        eps_sys = np.sum(np.abs(self.w[np.bitwise_not(keep)]))
        self.w = self.w[keep]
        self.obs = self.obs[keep]
        self.N_hits = self.N_hits[keep]
        self.num_obs = len(self.w)
        return eps_sys

    def get_epsilon_Bernstein(self,delta):
        """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
            If at least one of the N_hits is 0, epsilon is set equal to infinity.
        """
        if np.min(self.N_hits) == 0:
            return np.infty
        w_abs  = np.abs(self.w)
        w_abs /= np.sqrt(self.N_hits)
        norm   = np.sum(w_abs)
        w_abs /= np.sqrt(self.N_hits)
        norm2  = np.sum(w_abs)
        epsilon = norm * np.sqrt(N_delta(delta))
        if epsilon > 2*norm*(1+2*norm/norm2):
            print("Warning! Epsilon out of validity range.")
        return epsilon

##########################################################################################
### ShadowGrouping #######################################################################
##########################################################################################

class Shadow_Grouping(Measurement_scheme):
    """ Grouping method based on weights obtained from classical shadows.
        The next measurement setting p is found as follows: it is initialized as the identity operator.
        Next, we obtain an ordering of the observables in terms of their respective weight_function.
        For each observable o in the ordered list of observables in descending order, it checks qubit-wise commutativity (QWC).
        If so, the qubits in p that fall in the support of o are overwritten by those in o.
        Eventually, the list is either exhausted or p does not contain identity operators anymore.
        The function weight_function takes in the weights,epsilon and the current number of N_hits and is supposed to return an numpy-array of length len(w).
        Instead, weight_function can also be set to None (this is useful for instances where the function is actually never called).

        Returns p and a dictionary info holding further details on the matching procedure.
    """

    def __init__(
        self,
        observables,
        weights,
        epsilon,
        weight_function,
        commutation_mode="qwc",
        max_support_qubits: Optional[int] = 8,
    ):
        super().__init__(observables,weights,epsilon)
        if commutation_mode not in ("qwc", "fc"):
            raise ValueError("commutation_mode has to be either 'qwc' or 'fc'.")
        self.commutation_mode = commutation_mode
        self.max_support_qubits = max_support_qubits
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.groups_fc = None
        self.group_plans = None
        if self.commutation_mode == "fc":
            self.groups_fc = self._build_fc_groups()
            self.group_plans = self._build_fc_group_plans()
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return

    def _build_fc_groups(self):
        return build_commuting_groups(
            self.obs, commutation_mode="fc", max_support_qubits=self.max_support_qubits
        )

    def _build_fc_group_plans(self):
        return build_fc_group_plans(
            self.obs, self.groups_fc, max_support_qubits=self.max_support_qubits
        )

    def get_group_plan(self, group_id):
        if self.group_plans is None:
            return None
        return self.group_plans[int(group_id)]

    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return

    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        return np.sum(inconf)

    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound

    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits,dtype=int)

        if verbose:
            print("Checking list of observables.")
        tstart = time()
        chosen_gid = None
        chosen_group = None
        if self.commutation_mode == "qwc":
            for idx in reversed(order):
                o = self.obs[idx]
                if verbose:
                    print("Checking",o)
                if hit_by(o,setting):
                    non_id = o!=0
                    setting[non_id] = o[non_id]
                    if verbose:
                        print("p =",setting)
                    if np.min(setting) > 0:
                        break
        else:
            # FC mode: select a globally-commuting group and perform joint measurement
            group_scores = np.array([np.sum(weights[g]) for g in self.groups_fc], dtype=float)
            chosen_gid = int(np.argmax(group_scores))
            chosen_group = self.groups_fc[chosen_gid]
            rep_local_idx = int(np.argmax(weights[chosen_group]))
            best_idx = int(chosen_group[rep_local_idx])
            setting = self.obs[best_idx].copy()

        tend = time()

        if self.commutation_mode == "fc":
            is_hit = np.zeros(self.num_obs, dtype=bool)
            is_hit[chosen_group] = True
        else:
            is_hit = np.array([self.is_hit(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit

        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        if self.commutation_mode == "fc":
            info["group_id"] = chosen_gid
            info["group_size"] = len(chosen_group)
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info
