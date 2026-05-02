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


def _pauli_single_matrix(ind: int) -> np.ndarray:
    if ind == 0:
        return np.array([[1, 0], [0, 1]], dtype=complex)
    if ind == 1:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if ind == 2:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if ind == 3:
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError("Unknown Pauli index {}".format(ind))


def pauli_row_to_matrix(row: np.ndarray | list[int]) -> np.ndarray:
    mat = np.array([[1]], dtype=complex)
    for ind in row:
        mat = np.kron(mat, _pauli_single_matrix(int(ind)))
    return mat


def build_fc_group_plans(
    observables: np.ndarray, groups: list[np.ndarray], max_support_qubits: int | None = None,
) -> dict[int, dict]:
    """Build per-group diagonalizing unitaries and eigenvalue tables for FC measurement."""
    plans: dict[int, dict] = {}
    for gid, group in enumerate(groups):
        obs_indices = np.array(group, dtype=int)
        support_mask = np.any(observables[obs_indices] != 0, axis=0)
        qubits = np.where(support_mask)[0]

        if max_support_qubits is not None and len(qubits) > int(max_support_qubits):
            raise MemoryError(
                "FC group {} spans {} qubits, exceeding max_support_qubits={}.".format(
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
        basis = _find_diagonalizing_basis(pauli_mats)

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


def _find_diagonalizing_basis(pauli_mats: list[np.ndarray]) -> np.ndarray:
    """Find a unitary that simultaneously diagonalizes all given Pauli matrices."""
    dim = pauli_mats[0].shape[0]
    for trial in range(8):
        coeffs = np.array(
            [np.cos((k + 1) * (trial + 1)) for k in range(len(pauli_mats))], dtype=float
        )
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
            return vecs

    # fallback: identity
    return np.eye(dim, dtype=complex)


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
        max_support_qubits: int | None = 8,
    ) -> None:
        super().__init__(observables, weights, epsilon)
        if commutation_mode not in ("qwc", "fc"):
            raise ValueError("commutation_mode has to be either 'qwc' or 'fc'.")
        self.commutation_mode: str = commutation_mode
        self.max_support_qubits: int | None = max_support_qubits
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.groups_fc: list[np.ndarray] | None = None
        self.group_plans: dict[int, dict] | None = None
        if self.commutation_mode == "fc":
            self.groups_fc = self._build_fc_groups()
            self.group_plans = self._build_fc_group_plans()
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function returned array of shape {} instead of {}.".format(test.shape, self.w.shape)
            )

    def _build_fc_groups(self) -> list[np.ndarray]:
        n = self.num_obs
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            oi = self.obs[i]
            for j in range(i + 1, n):
                if not pauli_commute(oi, self.obs[j]):
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

        if self.max_support_qubits is None:
            return groups

        # Split groups exceeding max_support_qubits
        split_groups = []
        max_sup = int(self.max_support_qubits)
        for g in groups:
            bins: list[list[int]] = []
            bin_supports: list[set[int]] = []
            for idx in g:
                o = self.obs[int(idx)]
                o_support = set(np.where(o != 0)[0].tolist())
                placed = False
                for b_i in range(len(bins)):
                    if len(bin_supports[b_i] | o_support) > max_sup:
                        continue
                    ok = True
                    for jdx in bins[b_i]:
                        if not pauli_commute(o, self.obs[int(jdx)]):
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

    def _build_fc_group_plans(self) -> dict[int, dict]:
        assert self.groups_fc is not None
        return build_fc_group_plans(self.obs, self.groups_fc, self.max_support_qubits)

    def get_group_plan(self, group_id: int) -> dict | None:
        if self.group_plans is None:
            return None
        return self.group_plans[int(group_id)]

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
            assert self.groups_fc is not None
            group_scores = np.array([np.sum(weights[g]) for g in self.groups_fc], dtype=float)
            chosen_gid = int(np.argmax(group_scores))
            chosen_group = self.groups_fc[chosen_gid]
            rep_local_idx = int(np.argmax(weights[chosen_group]))
            best_idx = int(chosen_group[rep_local_idx])
            setting = self.obs[best_idx].copy()

            is_hit = np.zeros(self.num_obs, dtype=bool)
            is_hit[chosen_group] = True
            info["group_id"] = chosen_gid
            info["group_size"] = len(chosen_group)
        else:
            order = np.argsort(weights)
            setting = np.zeros(self.num_qubits, dtype=int)

            for idx in reversed(order):
                o = self.obs[idx]
                if hit_by(o, setting):
                    non_id = o != 0
                    setting[non_id] = o[non_id]
                    if np.min(setting) > 0:
                        break

            is_hit = np.array([self.is_hit(o, setting) for o in self.obs], dtype=bool)

        tend = time()
        self.N_hits += is_hit

        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        return setting, info
