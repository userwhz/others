from __future__ import annotations

import numpy as np
from time import time
from typing import Callable


def hit_by(O: list[int] | np.ndarray, P: list[int] | np.ndarray) -> bool:
    for o, p in zip(O, P):
        if not (o == 0 or p == 0 or o == p):
            return False
    return True


def pauli_commute(O0: list[int] | np.ndarray, O1: list[int] | np.ndarray) -> bool:
    """True if two Pauli strings commute globally (anticommuting qubits count is even)."""
    anti_count = 0
    for a, b in zip(O0, O1):
        if a == 0 or b == 0 or a == b:
            continue
        anti_count += 1
    return anti_count % 2 == 0


def hit_by_mode(
    O: list[int] | np.ndarray, P: list[int] | np.ndarray, commutation_mode: str = "qwc"
) -> bool:
    if commutation_mode == "qwc":
        return hit_by(O, P)
    if commutation_mode == "fc":
        return pauli_commute(O, P)
    raise ValueError("Unknown commutation_mode '{}'. Use 'qwc' or 'fc'.".format(commutation_mode))


def N_delta(delta: float) -> float:
    """Equation 6 from manuscript."""
    return 4 * (2 * np.sqrt(-np.log(delta)) + 1) ** 2


class Measurement_scheme:
    def __init__(self, observables: np.ndarray, weights: np.ndarray, epsilon: float) -> None:
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M, n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"

        self.obs: np.ndarray = observables
        self.num_obs: int = M
        self.num_qubits: int = n
        self.w: np.ndarray = weights
        self.eps: float = epsilon
        self.N_hits: np.ndarray = np.zeros(M, dtype=int)
        self.commutation_mode: str = "qwc"

    def is_hit(self, observable: np.ndarray | list[int], setting: np.ndarray | list[int]) -> bool:
        return hit_by_mode(observable, setting, self.commutation_mode)

    def find_setting(self) -> None:
        pass

    def reset(self) -> None:
        self.N_hits = np.zeros_like(self.N_hits)

    def get_epsilon_sys_stat(self, delta: float) -> tuple[float, float]:
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit)
        if np.sum(keep) == 0:
            eps_syst = float(np.sum(np.abs(self.w)))
            eps_stat = 0.0
        elif np.sum(keep) == len(keep):
            eps_syst = 0.0
            eps_stat = self.get_epsilon_Bernstein(delta)
        else:
            w, N = self.w, self.N_hits
            self.w = self.w[keep]
            self.N_hits = self.N_hits[keep]
            eps_syst = float(np.sum(np.abs(w[np.bitwise_not(keep)])))
            eps_stat = self.get_epsilon_Bernstein(delta)
            self.w = w
            self.N_hits = N
        return eps_syst, eps_stat

    def truncate(self, delta: float) -> float:
        N_unmeasured = np.sum(self.N_hits == 0)
        if N_unmeasured > 0:
            print("Warning! {} observable(s) have not been measured at least once.".format(N_unmeasured))
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit)
        if np.sum(keep) == 0:
            print("No observable reached the threshold. Ensure that you have sampled often enough!")
            print("Scheme unaltered.")
            return 0.0
        if np.sum(keep) == len(keep):
            print("Nothing had to be truncated.")
            return 0.0
        eps_sys = float(np.sum(np.abs(self.w[np.bitwise_not(keep)])))
        self.w = self.w[keep]
        self.obs = self.obs[keep]
        self.N_hits = self.N_hits[keep]
        self.num_obs = len(self.w)
        return eps_sys

    def get_epsilon_Bernstein(self, delta: float) -> float | np.floating:
        if np.min(self.N_hits) == 0:
            return np.inf
        w_abs = np.abs(self.w)
        w_abs /= np.sqrt(self.N_hits)
        norm = np.sum(w_abs)
        w_abs /= np.sqrt(self.N_hits)
        norm2 = np.sum(w_abs)
        epsilon = norm * np.sqrt(N_delta(delta))
        if epsilon > 2 * norm * (1 + 2 * norm / norm2):
            print("Warning! Epsilon out of validity range.")
        return epsilon


class Shadow_Grouping(Measurement_scheme):
    def __init__(
        self,
        observables: np.ndarray,
        weights: np.ndarray,
        epsilon: float,
        weight_function: Callable[[np.ndarray, float, np.ndarray], np.ndarray] | None,
        commutation_mode: str = "qwc",
    ) -> None:
        super().__init__(observables, weights, epsilon)
        if commutation_mode not in ("qwc", "fc"):
            raise ValueError("commutation_mode has to be either 'qwc' or 'fc'.")
        self.commutation_mode: str = commutation_mode
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function returned array of shape {} instead of {}.".format(test.shape, self.w.shape)
            )

    def reset(self) -> None:
        self.N_hits = np.zeros_like(self.N_hits)

    def get_inconfidence_bound(self) -> float | np.floating:
        inconf = np.exp(-0.5 * self.eps * self.eps * self.N_hits / (self.w ** 2))
        return np.sum(inconf)

    def get_Bernstein_bound(self) -> float | np.floating:
        if np.min(self.N_hits) == 0:
            return -1.0
        bound = np.exp(-0.25 * (self.eps / 2 / np.sum(np.abs(self.w) / np.sqrt(self.N_hits)) - 1) ** 2)
        return bound

    def find_setting(self, verbose: bool = False) -> tuple[np.ndarray, dict]:
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        tstart = time()
        info: dict = {}

        if self.commutation_mode == "fc":
            setting = self._find_fc_setting(weights)
        else:
            setting = self._find_qwc_setting(weights)

        is_hit = np.array([self.is_hit(o, setting) for o in self.obs], dtype=bool)
        tend = time()
        self.N_hits += is_hit

        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        return setting, info

    def _find_qwc_setting(self, weights: np.ndarray) -> np.ndarray:
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits, dtype=int)

        for idx in reversed(order):
            o = self.obs[idx]
            if hit_by(o, setting):
                non_id = o != 0
                setting[non_id] = o[non_id]
                if np.min(setting) > 0:
                    break
        return setting

    def _find_fc_setting(self, weights: np.ndarray) -> np.ndarray:
        """Algorithm 1 from the paper: dynamically build an FC measurement setting.

        Sorts observables by weight, then greedily adds them to the setting.
        When an observable has odd anticommutations with the current setting,
        patches one idle qubit to flip the parity.
        """
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits, dtype=int)
        OTHER = {1: [2, 3], 2: [1, 3], 3: [1, 2]}  # two other Paulis for each non-I Pauli

        for idx in reversed(order):
            o = self.obs[idx]
            support = setting != 0

            # Count anticommutations on setting's current support
            anti_count = 0
            for i in range(self.num_qubits):
                if support[i] and o[i] != 0 and o[i] != setting[i]:
                    anti_count += 1

            if anti_count % 2 == 0:
                # Compatible: fill O's Paulis into idle positions
                idle = setting == 0
                setting[idle] = o[idle]
            else:
                # Odd: try to patch — find an idle qubit where O is non-I
                idle_nonzero = np.where((setting == 0) & (o != 0))[0]
                if len(idle_nonzero) > 0:
                    q = int(idle_nonzero[0])
                    setting[q] = OTHER[int(o[q])][0]
                    idle = setting == 0
                    setting[idle] = o[idle]
                # else: odd and no idle non-I qubit — skip this observable

            if np.min(setting) > 0:
                break

        return setting
