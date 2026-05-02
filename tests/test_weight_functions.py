import numpy as np
import pytest
from shadowgrouping.weight_functions import Bernstein_bound


class TestBernsteinBound:
    @pytest.fixture
    def wf(self) -> Bernstein_bound:
        return Bernstein_bound(alpha=1)

    def test_alpha_default(self) -> None:
        wf = Bernstein_bound()
        assert wf.alpha == 1

    def test_alpha_custom(self) -> None:
        wf = Bernstein_bound(alpha=2.5)
        assert wf.alpha == 2.5

    def test_alpha_below_1_raises(self) -> None:
        with pytest.raises(AssertionError):
            Bernstein_bound(alpha=0.5)

    def test_no_hits_returns_alpha_abs_w(self, wf: Bernstein_bound) -> None:
        w = np.array([1.0, 2.0, 3.0])
        N_hits = np.zeros(3, dtype=int)
        weights = wf.get_weights(w, 0.1, N_hits)
        np.testing.assert_array_almost_equal(weights, np.abs(w))

    def test_with_hits_reduces_weight(self, wf: Bernstein_bound) -> None:
        w = np.array([2.0, 2.0])
        N_hits = np.array([0, 100])
        weights = wf.get_weights(w, 0.1, N_hits)
        assert weights[0] == pytest.approx(2.0)
        assert weights[1] < 2.0

    def test_more_hits_gives_lower_weight(self, wf: Bernstein_bound) -> None:
        w = np.array([1.0, 1.0])
        few = wf.get_weights(w, 0.1, np.array([10, 100]))
        many = wf.get_weights(w, 0.1, np.array([10, 1000]))
        assert many[1] < few[1]

    def test_call_returns_callable(self, wf: Bernstein_bound) -> None:
        fn = wf()
        w = np.array([1.0])
        result = fn(w, 0.1, np.array([10]))
        assert isinstance(result, np.ndarray)
