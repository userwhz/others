from __future__ import annotations

import numpy as np
from typing import Callable


class Bernstein_bound:
    def __init__(self, alpha: float = 1) -> None:
        self.alpha: float = alpha
        assert alpha >= 1, "alpha has to be chosen larger or equal 1, but was {}.".format(alpha)

    def get_weights(self, w: np.ndarray, eps: float, N_hits: np.ndarray) -> np.ndarray:
        inconf = self.alpha * np.abs(w)
        condition = N_hits != 0
        N = np.sqrt(N_hits[condition])
        Nplus1 = np.sqrt(N_hits[condition] + 1)
        inconf[condition] /= self.alpha * N * Nplus1 / (Nplus1 - N)
        return inconf

    def __call__(self) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
        return self.get_weights
