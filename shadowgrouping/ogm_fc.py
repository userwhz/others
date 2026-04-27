import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .measurement_schemes import pauli_commute


@dataclass(frozen=True)
class OGMGroupDistribution:
    """A distribution over (possibly overlapping) commuting groups.

    - groups: list of 1D int arrays, each contains observable indices.
    - p: 1D float array, probabilities aligned with groups.
    """

    groups: List[np.ndarray]
    p: np.ndarray


def _build_fc_overlap_groups_greedy(
    observables: np.ndarray,
    weights: np.ndarray,
    *,
    max_groups: Optional[int] = None,
    max_support_qubits: Optional[int] = 8,
) -> List[np.ndarray]:
    """Greedy construction of overlapping globally-commuting groups.

    This mirrors the common OGM-style heuristic:
    - Sort observables by |w| descending
    - Start a group from the next-unassigned observable
    - Add any other observable that commutes with all current group members
    - Optionally do a second sweep over earlier indices to maximize overlap

    Returns groups as lists of observable indices (0-based).
    """
    obs = np.asarray(observables, dtype=int)
    w = np.asarray(weights, dtype=float).reshape(-1)
    m = obs.shape[0]

    order = np.argsort(np.abs(w))[::-1]
    assigned = np.zeros(m, dtype=bool)
    groups: List[np.ndarray] = []

    # map from sorted-rank -> original index
    sorted_idx = order

    def commute_with_group(idx: int, group_members: Sequence[int]) -> bool:
        oi = obs[idx]
        for jj in group_members:
            if not pauli_commute(oi, obs[jj]):
                return False
        if max_support_qubits is not None:
            # enforce union support size constraint for joint measurement feasibility
            sup = set(np.where(oi != 0)[0].tolist())
            for jj in group_members:
                sup |= set(np.where(obs[jj] != 0)[0].tolist())
            if len(sup) > int(max_support_qubits):
                return False
        return True

    k = 0
    while k < m:
        # find next unassigned in sorted order
        while k < m and assigned[sorted_idx[k]]:
            k += 1
        if k >= m:
            break

        seed = int(sorted_idx[k])
        group_members: List[int] = [seed]
        assigned[seed] = True

        # forward sweep: add later terms that commute with current group
        for t in range(k + 1, m):
            idx = int(sorted_idx[t])
            if commute_with_group(idx, group_members):
                group_members.append(idx)
                assigned[idx] = True

        # backward sweep: allow overlap by adding earlier terms too
        for t in range(0, k):
            idx = int(sorted_idx[t])
            if commute_with_group(idx, group_members):
                group_members.append(idx)

        groups.append(np.array(group_members, dtype=int))
        if max_groups is not None and len(groups) >= int(max_groups):
            break

        k += 1

    return groups


def _cover_matrix(m: int, groups: Sequence[np.ndarray]) -> np.ndarray:
    cover = np.zeros((m, len(groups)), dtype=bool)
    for g_idx, g in enumerate(groups):
        cover[g.astype(int), g_idx] = True
    return cover


def _diag_variance_objective(p: np.ndarray, w: np.ndarray, cover: np.ndarray, T: float) -> float:
    """OGM diagonal-variance style objective: sum_i w_i^2 / sum_{g covers i} p_g with penalty if uncovered."""
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)

    cov_prob = cover @ p  # (m,)
    # penalty for uncovered observables
    out = 0.0
    for i in range(len(w)):
        if cov_prob[i] > 0:
            out += (w[i] ** 2) / cov_prob[i]
        else:
            out += (w[i] ** 2) * T
    return float(out)


def optimize_ogm_fc_distribution(
    observables: np.ndarray,
    weights: np.ndarray,
    *,
    T: float = 100000.0,
    max_groups: Optional[int] = None,
    max_support_qubits: Optional[int] = 8,
    solver_maxiter: int = 300,
) -> OGMGroupDistribution:
    """Compute an OGM-style distribution over globally commuting (FC) groups.

    Output is compatible with joint-measurement mode in `Energy_estimator` via `SettingSampler(commutation_mode="fc")`
    when saved in the `.groups.txt` format (see `save_group_distribution`).
    """
    obs = np.asarray(observables, dtype=int)
    w = np.asarray(weights, dtype=float).reshape(-1)
    m = obs.shape[0]
    if len(w) != m:
        raise ValueError("weights length does not match observables")

    groups = _build_fc_overlap_groups_greedy(
        obs, w, max_groups=max_groups, max_support_qubits=max_support_qubits
    )
    if len(groups) == 0:
        raise ValueError("no groups built")

    cover = _cover_matrix(m, groups).astype(float)

    # Initialize p proportional to sum |w| in group (common heuristic, consistent with many OGM baselines)
    g_mass = np.array([np.sum(np.abs(w[g])) for g in groups], dtype=float)
    if np.sum(g_mass) <= 0:
        p0 = np.ones(len(groups), dtype=float) / len(groups)
    else:
        p0 = g_mass / np.sum(g_mass)

    # Optimize probabilities on simplex with scipy if available; else keep p0.
    p = p0.copy()
    try:
        from scipy.optimize import minimize

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(len(groups))]

        res = minimize(
            fun=lambda x: _diag_variance_objective(x, w, cover, float(T)),
            x0=p0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": int(solver_maxiter), "ftol": 1e-9, "disp": False},
        )
        if res.success and np.all(np.isfinite(res.x)):
            p = np.maximum(res.x, 0.0)
            s = float(np.sum(p))
            if s > 0:
                p = p / s
    except Exception:
        # Keep p0 if scipy unavailable or optimization fails
        p = p0

    return OGMGroupDistribution(groups=groups, p=p)


def save_group_distribution(path: str, dist: OGMGroupDistribution) -> None:
    """Save as:
        # p_g obs_indices...
        0.12 0 5 9
        ...
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# p_g  obs_indices (0-based)\n")
        for pg, g in zip(dist.p, dist.groups):
            g = np.asarray(g, dtype=int).reshape(-1)
            f.write("{:.16g}".format(float(pg)))
            for idx in g:
                f.write(" {}".format(int(idx)))
            f.write("\n")

