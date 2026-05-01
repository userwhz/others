import numpy as np
from time import time

##########################################################################################
### Helper functions #####################################################################
##########################################################################################


def hit_by(O, P):
    for o, p in zip(O, P):
        if not (o == 0 or p == 0 or o == p):
            return False
    return True


def qwc_compatible(O0, O1):
    for a, b in zip(O0, O1):
        if not (a == 0 or b == 0 or a == b):
            return False
    return True


def hit_by_mode(O, P, commutation_mode="qwc"):
    if commutation_mode == "qwc":
        return hit_by(O, P)
    raise ValueError("Unknown commutation_mode '{}'. Use 'qwc'.".format(commutation_mode))


def setting_to_str(arr):
    out = ""
    for a in np.array(arr).flatten():
        out += str(a)
    return out


# equation 6 from manuscript
N_delta = lambda delta: 4 * (2 * np.sqrt(-np.log(delta)) + 1) ** 2


##########################################################################################
### Measurement scheme base class ########################################################
##########################################################################################


class Measurement_scheme:
    def __init__(self, observables, weights, epsilon):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M, n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"

        self.obs = observables
        self.num_obs = M
        self.num_qubits = n
        self.w = weights
        self.eps = epsilon
        self.scheme_params = {"eps": epsilon, "num_obs": M}
        self.N_hits = np.zeros(M, dtype=int)
        self.is_adaptive = False

    def is_hit(self, observable, setting):
        return hit_by(observable, setting)

    def find_setting(self):
        pass

    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)

    def get_epsilon_sys_stat(self, delta):
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

    def truncate(self, delta):
        N_unmeasured = np.sum(self.N_hits == 0)
        if N_unmeasured > 0:
            print("Warning! {} observable(s) have not been measured at least once.".format(N_unmeasured))
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit)
        if np.sum(keep) == 0:
            print("No observable reached the threshold. Ensure that you have sampled often enough!")
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

    def get_epsilon_Bernstein(self, delta):
        if np.min(self.N_hits) == 0:
            return np.infty
        w_abs = np.abs(self.w)
        w_abs /= np.sqrt(self.N_hits)
        norm = np.sum(w_abs)
        w_abs /= np.sqrt(self.N_hits)
        norm2 = np.sum(w_abs)
        epsilon = norm * np.sqrt(N_delta(delta))
        if epsilon > 2 * norm * (1 + 2 * norm / norm2):
            print("Warning! Epsilon out of validity range.")
        return epsilon


##########################################################################################
### ShadowGrouping #######################################################################
##########################################################################################


class Shadow_Grouping(Measurement_scheme):
    def __init__(self, observables, weights, epsilon, weight_function):
        super().__init__(observables, weights, epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function returned array of shape {} instead of {}.".format(test.shape, self.w.shape)
            )
        self.is_sampling = False

    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)

    def get_inconfidence_bound(self):
        inconf = np.exp(-0.5 * self.eps * self.eps * self.N_hits / (self.w ** 2))
        return np.sum(inconf)

    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            return -1
        bound = np.exp(-0.25 * (self.eps / 2 / np.sum(np.abs(self.w) / np.sqrt(self.N_hits)) - 1) ** 2)
        return bound

    def find_setting(self, verbose=False):
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits, dtype=int)

        tstart = time()
        for idx in reversed(order):
            o = self.obs[idx]
            if hit_by(o, setting):
                non_id = o != 0
                setting[non_id] = o[non_id]
                if np.min(setting) > 0:
                    break
        tend = time()

        is_hit = np.array([self.is_hit(o, setting) for o in self.obs], dtype=bool)
        self.N_hits += is_hit

        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        return setting, info
