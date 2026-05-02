import numpy as np
import pytest
from shadowgrouping.measurement_schemes import (
    hit_by, qwc_compatible, pauli_commute, hit_by_mode, setting_to_str, N_delta,
    Measurement_scheme, Shadow_Grouping, pauli_row_to_matrix, build_fc_group_plans,
    _find_diagonalizing_basis,
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

    def test_fc_mode_uses_pauli_commute(self) -> None:
        assert hit_by_mode([1, 1], [1, 1], "fc")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown commutation_mode"):
            hit_by_mode([1, 0], [1, 0], "xyz")


class TestPauliCommute:
    def test_identical_commutes(self) -> None:
        assert pauli_commute([1, 2, 3], [1, 2, 3])

    def test_identity_everywhere_commutes(self) -> None:
        assert pauli_commute([0, 0, 0], [1, 2, 3])

    def test_single_anticommute_does_not_commute(self) -> None:
        # X and Z on same qubit anticommute
        assert not pauli_commute([1, 0], [3, 0])

    def test_two_anticommuting_qubits_commute(self) -> None:
        # Two anticommuting pairs → even → globally commute
        assert pauli_commute([1, 1, 0], [3, 3, 0])

    def test_three_anticommuting_qubits_do_not_commute(self) -> None:
        # Three anticommuting pairs → odd → globally do not commute
        assert not pauli_commute([1, 1, 1], [3, 3, 3])

    @pytest.mark.parametrize("a,b,expected", [
        ([0, 0], [0, 0], True),
        ([1, 2], [2, 1], True),   # both anticommute → even
        ([1, 0], [3, 0], False),  # single anticommute → odd
        ([1, 2], [3, 2], False),  # X-Z anticommutes, Y-Y commutes → odd
        ([1, 1], [2, 3], True),   # X-Y=ant, X-Z=ant → 2 → even
    ])
    def test_parametrized(self, a: list[int], b: list[int], expected: bool) -> None:
        assert pauli_commute(a, b) == expected


class TestPauliRowToMatrix:
    def test_I_is_identity(self) -> None:
        mat = pauli_row_to_matrix([0, 0])
        assert mat.shape == (4, 4)
        assert np.allclose(mat, np.eye(4))

    def test_single_X_traceless_hermitian(self) -> None:
        mat = pauli_row_to_matrix([1])
        assert mat.shape == (2, 2)
        assert np.allclose(mat, mat.conj().T)
        assert np.isclose(np.trace(mat), 0)

    def test_single_Z_diagonal(self) -> None:
        mat = pauli_row_to_matrix([3])
        expected = np.diag([1, -1])
        assert np.allclose(mat, expected)

    def test_YY_kron(self) -> None:
        # Y ⊗ Y
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        expected = np.kron(y, y)
        mat = pauli_row_to_matrix([2, 2])
        assert mat.shape == (4, 4)
        assert np.allclose(mat, expected)


class TestFindDiagonalizingBasis:
    def test_identity_alone(self) -> None:
        mat = np.array([[1, 0], [0, 1]], dtype=complex)
        basis = _find_diagonalizing_basis([mat])
        assert np.allclose(basis, np.eye(2))

    def test_single_Z_diagonal_in_self(self) -> None:
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        basis = _find_diagonalizing_basis([z])
        d = basis.conj().T @ z @ basis
        off = d - np.diag(np.diag(d))
        assert np.max(np.abs(off)) < 1e-7


class TestBuildFcGroupPlans:
    def test_empty_list(self) -> None:
        obs = np.empty((0, 4), dtype=int)
        plans = build_fc_group_plans(obs, [])
        assert plans == {}

    def test_single_observable(self) -> None:
        obs = np.array([[1, 0, 0, 0]], dtype=int)
        groups = [np.array([0])]
        plans = build_fc_group_plans(obs, groups)
        assert 0 in plans
        plan = plans[0]
        assert len(plan["obs_indices"]) == 1
        assert len(plan["qubits"]) == 1  # only qubit 0
        assert plan["eigenvalues"].shape == (1, 2)

    def test_empty_qubits_group(self) -> None:
        obs = np.array([[0, 0, 0]], dtype=int)
        groups = [np.array([0])]
        plans = build_fc_group_plans(obs, groups)
        assert len(plans[0]["qubits"]) == 0
        assert np.array_equal(plans[0]["eigenvalues"], np.ones((1, 1), dtype=int))


class TestShadowGroupingFC:
    @pytest.fixture
    def wf(self):
        return Bernstein_bound(alpha=1)()

    @pytest.fixture
    def fc_scheme(self, wf) -> Shadow_Grouping:
        obs = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        return Shadow_Grouping(obs, w, epsilon=0.1, weight_function=wf, commutation_mode="fc")

    def test_init_builds_groups(self, fc_scheme: Shadow_Grouping) -> None:
        assert fc_scheme.groups_fc is not None
        assert len(fc_scheme.groups_fc) > 0
        assert fc_scheme.group_plans is not None

    def test_get_group_plan(self, fc_scheme: Shadow_Grouping) -> None:
        plan = fc_scheme.get_group_plan(0)
        assert plan is not None
        assert "unitary" in plan
        assert "eigenvalues" in plan

    def test_find_setting_returns_group_id(self, fc_scheme: Shadow_Grouping) -> None:
        _, info = fc_scheme.find_setting()
        assert "group_id" in info
        assert "group_size" in info
        assert info["group_size"] > 0

    def test_find_setting_hits_all_group_members(self, fc_scheme: Shadow_Grouping) -> None:
        _, info = fc_scheme.find_setting()
        gid = info["group_id"]
        group = fc_scheme.groups_fc[gid]
        assert np.all(fc_scheme.N_hits[group] == 1)

    def test_commutation_mode_validation(self) -> None:
        obs = np.array([[1, 0], [0, 1]])
        w = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="commutation_mode"):
            Shadow_Grouping(obs, w, epsilon=0.1, weight_function=None, commutation_mode="xyz")


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
