import numpy as np
import pytest
from shadowgrouping.measurement_schemes import (
    hit_by, qwc_compatible, hit_by_mode, setting_to_str, N_delta,
    Measurement_scheme, Shadow_Grouping,
)
from shadowgrouping.weight_functions import Bernstein_bound


class TestHitBy:
    def test_identical(self) -> None:
        assert hit_by([1, 2, 3, 0], [1, 2, 3, 0])

    def test_identity_in_O_does_not_block(self) -> None:
        assert hit_by([1, 0, 3], [1, 2, 3])

    def test_identity_in_P_does_not_block(self) -> None:
        assert hit_by([1, 2, 3], [0, 0, 3])

    def test_mismatch_rejected(self) -> None:
        assert not hit_by([1, 2, 3], [1, 3, 3])

    @pytest.mark.parametrize("O,P,expected", [
        ([0, 0, 0], [1, 2, 3], True),
        ([1, 2, 3], [0, 0, 0], True),
        ([1, 1, 1], [1, 1, 1], True),
        ([1, 0, 1], [0, 1, 0], True),
        ([1, 2, 3], [2, 1, 3], False),
    ])
    def test_parametrized(self, O: list[int], P: list[int], expected: bool) -> None:
        assert hit_by(O, P) == expected


class TestQwcCompatible:
    def test_same_as_hit_by(self) -> None:
        import random
        rng = random.Random(42)
        for _ in range(100):
            o = [rng.randint(0, 3) for _ in range(4)]
            p = [rng.randint(0, 3) for _ in range(4)]
            assert qwc_compatible(o, p) == hit_by(o, p)


class TestHitByMode:
    def test_qwc_mode_delegates_to_hit_by(self) -> None:
        assert hit_by_mode([1, 0, 3], [1, 2, 3], "qwc")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown commutation_mode"):
            hit_by_mode([1, 0], [1, 0], "fc")


class TestSettingToStr:
    def test_single_element(self) -> None:
        assert setting_to_str(np.array([1])) == "1"

    def test_multiple(self) -> None:
        assert setting_to_str(np.array([1, 0, 3, 2])) == "1032"

    def test_2d_array_flattened(self) -> None:
        assert setting_to_str(np.array([[1, 2], [3, 0]])) == "1230"


class TestNDelta:
    def test_value(self) -> None:
        delta = 0.05
        expected = 4 * (2 * np.sqrt(-np.log(delta)) + 1) ** 2
        assert N_delta(delta) == pytest.approx(expected)

    def test_smaller_delta_gives_larger_N(self) -> None:
        assert N_delta(0.01) > N_delta(0.1)


class TestMeasurementScheme:
    @pytest.fixture
    def scheme(self) -> Measurement_scheme:
        obs = np.array([
            [1, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 1, 0],
        ])
        w = np.array([1.0, 2.0, 0.5])
        return Measurement_scheme(obs, w, epsilon=0.1)

    def test_init(self, scheme: Measurement_scheme) -> None:
        assert scheme.num_obs == 3
        assert scheme.num_qubits == 4
        assert np.array_equal(scheme.N_hits, [0, 0, 0])

    def test_reset(self, scheme: Measurement_scheme) -> None:
        scheme.N_hits = np.array([5, 3, 0])
        scheme.reset()
        assert np.array_equal(scheme.N_hits, [0, 0, 0])

    def test_is_hit(self, scheme: Measurement_scheme) -> None:
        assert scheme.is_hit([1, 0, 0, 0], [1, 2, 0, 0])

    def test_get_epsilon_Bernstein_no_hits(self, scheme: Measurement_scheme) -> None:
        assert scheme.get_epsilon_Bernstein(0.05) == np.inf

    def test_get_epsilon_Bernstein_all_hit(self, scheme: Measurement_scheme) -> None:
        scheme.N_hits = np.array([100, 100, 100])
        eps = scheme.get_epsilon_Bernstein(0.05)
        assert eps < np.inf
        assert eps > 0

    def test_truncate_no_threshold_reached(self, scheme: Measurement_scheme) -> None:
        eps_sys = scheme.truncate(1e-10)
        assert eps_sys == 0

    def test_get_epsilon_sys_stat_no_keep(self, scheme: Measurement_scheme) -> None:
        esys, estat = scheme.get_epsilon_sys_stat(1e-10)
        assert esys == pytest.approx(3.5)
        assert estat == 0


class TestShadowGrouping:
    @pytest.fixture
    def wf(self):
        return Bernstein_bound(alpha=1)()

    @pytest.fixture
    def scheme(self, wf) -> Shadow_Grouping:
        obs = np.array([
            [1, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
        ])
        w = np.array([1.0, 1.0, 1.0, 0.5])
        return Shadow_Grouping(obs, w, epsilon=0.1, weight_function=wf)

    def test_init(self, scheme: Shadow_Grouping) -> None:
        assert scheme.num_obs == 4
        assert scheme.num_qubits == 4

    def test_find_setting_returns_valid_setting(self, scheme: Shadow_Grouping) -> None:
        setting, info = scheme.find_setting()
        assert len(setting) == 4
        assert all(x in (0, 1, 2, 3) for x in setting)
        assert "total_weight" in info
        assert "Bernstein bound" in info

    def test_find_setting_increments_hits(self, scheme: Shadow_Grouping) -> None:
        assert np.all(scheme.N_hits == 0)
        scheme.find_setting()
        assert np.sum(scheme.N_hits) > 0

    def test_find_setting_covers_all_obs_eventually(self, scheme: Shadow_Grouping) -> None:
        for _ in range(20):
            scheme.find_setting()
        assert np.all(scheme.N_hits > 0)

    def test_reset(self, scheme: Shadow_Grouping) -> None:
        scheme.find_setting()
        scheme.reset()
        assert np.all(scheme.N_hits == 0)

    def test_get_inconfidence_bound(self, scheme: Shadow_Grouping) -> None:
        scheme.N_hits = np.array([100, 100, 100, 100])
        bound = scheme.get_inconfidence_bound()
        assert 0 < bound < 4

    def test_get_Bernstein_bound_no_hits(self, scheme: Shadow_Grouping) -> None:
        assert scheme.get_Bernstein_bound() == -1

    def test_weight_function_none_accepted(self) -> None:
        obs = np.array([[1, 0], [0, 3]])
        w = np.array([1.0, 1.0])
        scheme = Shadow_Grouping(obs, w, epsilon=0.1, weight_function=None)
        assert scheme.weight_function is None
