import numpy as np
import pytest
from shadowgrouping.hamiltonian import Hamiltonian, char_to_int, int_to_char, random_hamiltonian


class TestCharIntMapping:
    def test_char_to_int(self) -> None:
        assert char_to_int == {"I": 0, "X": 1, "Y": 2, "Z": 3}

    def test_int_to_char(self) -> None:
        assert int_to_char == {0: "I", 1: "X", 2: "Y", 3: "Z"}

    def test_roundtrip(self) -> None:
        for c in "IXYZ":
            assert int_to_char[char_to_int[c]] == c


class TestHamiltonian:
    def test_single_Z(self) -> None:
        H = Hamiltonian(np.array([1.0]), ["Z"])
        energy, state = H.ground()
        assert energy == pytest.approx(-1.0)
        assert len(state) == 2

    def test_single_X(self) -> None:
        H = Hamiltonian(np.array([1.0]), ["X"])
        energy, state = H.ground()
        assert energy == pytest.approx(-1.0)

    def test_two_qubit_ZZ(self) -> None:
        H = Hamiltonian(np.array([1.0]), ["ZZ"])
        energy, state = H.ground()
        assert energy == pytest.approx(-1.0)
        assert len(state) == 4

    def test_SummedOp_returns_summed_op(self) -> None:
        H = Hamiltonian(np.array([1.0, 0.5]), ["Z", "X"])
        op = H.SummedOp()
        assert op is not None


class TestRandomHamiltonian:
    def test_size(self) -> None:
        ham = random_hamiltonian(4, 20)
        assert len(ham) == 20

    def test_pauli_format(self) -> None:
        ham = random_hamiltonian(4, 5)
        for pauli, coeff in ham.items():
            assert len(pauli) == 4
            assert all(c in "IXYZ" for c in pauli)
            assert isinstance(coeff, float)

    def test_no_identity_term(self) -> None:
        ham = random_hamiltonian(3, 10)
        assert "III" not in ham

    def test_no_duplicate_terms(self) -> None:
        ham = random_hamiltonian(3, 60)
        assert len(ham) == 60

    def test_seed_reproducible(self) -> None:
        np.random.seed(42)
        ham1 = random_hamiltonian(4, 10)
        np.random.seed(42)
        ham2 = random_hamiltonian(4, 10)
        assert list(ham1.keys()) == list(ham2.keys())
        assert list(ham1.values()) == list(ham2.values())

    def test_kterm_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="kterm"):
            random_hamiltonian(2, 20)
