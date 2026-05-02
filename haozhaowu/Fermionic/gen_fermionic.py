"""
Random fermionic ladder operators (Jordan–Wigner) on n qubits:
  - *_pauli.txt   Pauli string + coefficient lines (real part)
  - *.npy        dense Hamiltonian matrix (real part)
  - *.json       {"IIIXYZI": coeff, ...} real coefficients only
  - 随机量子态 |psi> 仅保存向量 .npy（不生成密度矩阵）

Indices in annihilate / create are 1-based (same convention as the Julia snippet).

Pauli 项数控制（优先级从高到低）：
  1) TARGET_PAULI_TERMS_BY_L：按 L 指定目标项数，文件名 …_L{L}_terms{N}（N 为该 L 的配置值）。
  2) TARGET_PAULI_TERMS：单个 int，每个 L 相同目标，文件名 …_terms{N}。
  3) 二者均为 None：NUM_LADDER_TERMS 固定 ladder 条数，文件名 …_ladder{N}。
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import reduce
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ================= 配置 =================
NQ = 7
LADDER_LENGTHS = (1, 2, 3, 4)  # |annihilate| = |create| = L
# 每个 L 的目标 Pauli 项数（>=）；文件名 hamiltonian_..._L{L}_terms{N}
# 文件名 …_L{L}_terms{N} 中的 N；L=1 在 nq=7 上界见 _MAX_PAULI_L1_N7（约 92），不能写 100
TARGET_PAULI_TERMS_BY_L: Optional[Dict[int, int]] = {
	1: 92,
	2: 200,
	3: 400,
	4: 1000,
}
# 若不需要按 L 区分，把 BY_L 设为 None 并改用下面单一目标（或再改为 None 走 ladder 模式）
TARGET_PAULI_TERMS: Optional[int] = None
NUM_LADDER_TERMS = 300  # 仅当 BY_L 与 TARGET_PAULI_TERMS 均为 None 时使用
MAX_LADDER_DRAWS = 2_000_000  # 按 Pauli 项数生成时的安全上限，防止达不到目标死循环
RNG_SEED = 2026
COEFF_RANGE = (-1.0, 1.0)

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "hamil_class")
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# Pauli 矩阵（与 gen_klocal 一致）
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli_dict = {"I": I, "X": X, "Y": Y, "Z": Z}


def _build_single_qubit_pauli_product_table() -> Dict[Tuple[str, str], Tuple[complex, str]]:
	"""For 2x2 Paulis: P_a P_b = phase * P_c."""
	table: Dict[Tuple[str, str], Tuple[complex, str]] = {}
	for a in pauli_dict:
		for b in pauli_dict:
			M = pauli_dict[a] @ pauli_dict[b]
			for c in pauli_dict:
				coef = np.trace(pauli_dict[c].conj().T @ M) / 2.0
				if abs(coef) > 1e-14:
					table[(a, b)] = (complex(coef), c)
					break
	return table


_SINGLE_Q_MUL = _build_single_qubit_pauli_product_table()


def _mul_pauli_strings(s: str, t: str) -> Tuple[complex, str]:
	phase = 1.0 + 0.0j
	out: List[str] = []
	for ca, cb in zip(s, t):
		p, cc = _SINGLE_Q_MUL[(ca, cb)]
		phase *= p
		out.append(cc)
	return phase, "".join(out)


def _mul_pauli_dict(d1: Dict[str, complex], d2: Dict[str, complex]) -> Dict[str, complex]:
	out: Dict[str, complex] = defaultdict(complex)
	for s1, c1 in d1.items():
		for s2, c2 in d2.items():
			ph, s = _mul_pauli_strings(s1, s2)
			out[s] += c1 * c2 * ph
	return {k: v for k, v in out.items() if abs(v) > 1e-15}


def _prod_pauli_dicts(dicts: List[Dict[str, complex]]) -> Dict[str, complex]:
	if not dicts:
		return {}
	return reduce(_mul_pauli_dict, dicts)


def _jwt_pauli_dict(i: int, is_dagger: bool, nq: int) -> Dict[str, complex]:
	"""Julia _JWT: site i is 1-based."""
	sx: List[str] = []
	sy: List[str] = []
	for q in range(1, nq + 1):
		if q < i:
			sx.append("Z")
			sy.append("Z")
		elif q == i:
			sx.append("X")
			sy.append("Y")
		else:
			sx.append("I")
			sy.append("I")
	s1 = "".join(sx)
	s2 = "".join(sy)
	if is_dagger:
		return {s1: 0.5 + 0.0j, s2: 0.5j}
	return {s1: 0.5 + 0.0j, s2: -0.5j}


def jordan_wigner_pauli_dict(
	annihilate: Iterable[int], create: Iterable[int], nq: int
) -> Dict[str, complex]:
	"""Julia: prod(_JWT.(a, false, nq)) * prod(_JWT.(a_dagger, true, nq))."""
	a = list(annihilate)
	c = list(create)
	if len(a) != len(c):
		raise ValueError("annihilate and create must have the same length")

	ident = {"I" * nq: 1.0 + 0.0j}
	if a:
		factors_a = [_jwt_pauli_dict(int(i), False, nq) for i in a]
		part_a = _prod_pauli_dicts(factors_a)
	else:
		part_a = ident
	if c:
		factors_c = [_jwt_pauli_dict(int(i), True, nq) for i in c]
		part_c = _prod_pauli_dicts(factors_c)
	else:
		part_c = ident
	return _mul_pauli_dict(part_a, part_c)


def _max_pauli_strings_l1_nq7() -> int:
	"""nq=7、L=1 时所有 c_i c_j† 的 JW 展开中出现的不同 Pauli 串个数上界。"""
	keys: set[str] = set()
	for i in range(1, 8):
		for j in range(1, 8):
			keys |= set(jordan_wigner_pauli_dict([i], [j], 7).keys())
	return len(keys)


_MAX_PAULI_L1_N7 = _max_pauli_strings_l1_nq7()


def get_pauli_matrix(pauli_string: str) -> np.ndarray:
	mats = [pauli_dict[s] for s in pauli_string]
	return reduce(np.kron, mats)


def pauli_dict_to_dense_matrix(d: Dict[str, complex], nq: int) -> np.ndarray:
	dim = 2**nq
	H = np.zeros((dim, dim), dtype=complex)
	for p_str, coeff in d.items():
		if abs(coeff) > 1e-15:
			H += coeff * get_pauli_matrix(p_str)
	return H


def _count_active_pauli_strings(d: Dict[str, complex], atol: float = 1e-9) -> int:
	"""按复系数模长计数（用于达到目标项数）；纯虚系数也算一项。"""
	return sum(1 for v in d.values() if abs(v) > atol)


def _take_real_hamiltonian_dict(d: Dict[str, complex], atol: float = 1e-9) -> Dict[str, float]:
	"""导出用：只保留实部系数；|v| 太小或 |Re v| 太小则丢弃。"""
	out: Dict[str, float] = {}
	for k, v in d.items():
		if abs(v) < atol:
			continue
		rv = float(v.real)
		if abs(rv) > atol:
			out[k] = rv
	return out


def save_pauli_txt(H_dict: Dict[str, float], filename: str) -> None:
	"""与 haozhaowu/klocal/gen_klocal.py 中 save_pauli_txt 相同格式。"""
	with open(filename, "w") as f:
		for p_str, coeff in H_dict.items():
			if abs(coeff) > 1e-9:
				f.write(f"{p_str}\n")
				f.write(f"({coeff:.16f}+0.0j)\n")


def save_json_pauli(H_dict: Dict[str, float], filename: str) -> None:
	with open(filename, "w") as f:
		json.dump(H_dict, f, indent=0, sort_keys=True)


def random_ladder_indices(nq: int, length: int, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
	"""|annihilate|=|create|=L；各自在 1..nq 内无放回抽样，彼此独立（可含不同模式，便于 Pauli 展开覆盖）。"""
	ann = list(rng.choice(np.arange(1, nq + 1), size=length, replace=False))
	cre = list(rng.choice(np.arange(1, nq + 1), size=length, replace=False))
	return ann, cre


def generate_random_state_vector(num_qubits: int, rng: np.random.Generator) -> np.ndarray:
	"""Haar-like random normalized state in computational basis (same idea as gen_klocal)."""
	dim = 2**num_qubits
	real_part = rng.normal(size=dim)
	imag_part = rng.normal(size=dim)
	psi = real_part + 1j * imag_part
	psi /= np.linalg.norm(psi)
	return psi


def build_random_fermionic_hamiltonian(
	nq: int,
	ladder_length: int,
	rng: np.random.Generator,
	*,
	target_pauli_terms: Optional[int] = None,
	num_ladder_terms: Optional[int] = None,
	max_ladder_draws: int = MAX_LADDER_DRAWS,
) -> Tuple[np.ndarray, Dict[str, float], int, int]:
	"""返回 (H 稠密实部矩阵, Pauli 导出实数字典, ladder 条数, 按 |c| 计的活跃 Pauli 串数)。"""
	if target_pauli_terms is not None and num_ladder_terms is not None:
		raise ValueError("请只设置其一：target_pauli_terms 或 num_ladder_terms")
	if target_pauli_terms is None and num_ladder_terms is None:
		raise ValueError("必须设置 target_pauli_terms 或 num_ladder_terms")

	acc: Dict[str, complex] = defaultdict(complex)
	draws = 0

	if target_pauli_terms is not None:
		if target_pauli_terms < 1:
			raise ValueError("target_pauli_terms 至少为 1")
		while True:
			n_active = _count_active_pauli_strings(dict(acc))
			if n_active >= target_pauli_terms:
				break
			if draws >= max_ladder_draws:
				raise RuntimeError(
					f"L={ladder_length} 在 {max_ladder_draws} 次 ladder 后仍只有 {n_active} 个"
					f"（按 |系数| 计）Pauli 串，无法达到目标 {target_pauli_terms}。"
					f"可降低目标或增大 MAX_LADDER_DRAWS。"
				)
			ann, cre = random_ladder_indices(nq, ladder_length, rng)
			block = jordan_wigner_pauli_dict(ann, cre, nq)
			alpha = rng.uniform(*COEFF_RANGE)
			for p, c in block.items():
				acc[p] += alpha * c
			draws += 1
	else:
		for _ in range(num_ladder_terms):
			ann, cre = random_ladder_indices(nq, ladder_length, rng)
			block = jordan_wigner_pauli_dict(ann, cre, nq)
			alpha = rng.uniform(*COEFF_RANGE)
			for p, c in block.items():
				acc[p] += alpha * c
			draws += 1

	H_dense = np.real(pauli_dict_to_dense_matrix(dict(acc), nq))
	H_pauli_real = _take_real_hamiltonian_dict(dict(acc))
	n_active = _count_active_pauli_strings(dict(acc))
	return H_dense, H_pauli_real, draws, n_active


def main() -> None:
	rng = np.random.default_rng(RNG_SEED)
	if TARGET_PAULI_TERMS_BY_L is not None:
		mode = f"per-L Pauli targets: {TARGET_PAULI_TERMS_BY_L}"
	elif TARGET_PAULI_TERMS is not None:
		mode = f"target_pauli>={TARGET_PAULI_TERMS} (all L)"
	else:
		mode = f"ladder_count={NUM_LADDER_TERMS}"
	print(f"=== Fermionic JW Hamiltonians: nq={NQ}, lengths={LADDER_LENGTHS}, {mode} ===")

	for L in LADDER_LENGTHS:
		if TARGET_PAULI_TERMS_BY_L is not None:
			if L not in TARGET_PAULI_TERMS_BY_L:
				raise KeyError(
					f"TARGET_PAULI_TERMS_BY_L 缺少键 L={L}，请为 LADDER_LENGTHS 中每个 L 配置项数"
				)
			n_terms = TARGET_PAULI_TERMS_BY_L[L]
			if NQ == 7 and L == 1 and n_terms > _MAX_PAULI_L1_N7:
				raise ValueError(
					f"nq=7、L=1 时 JW 展开最多出现 {_MAX_PAULI_L1_N7} 个不同 Pauli 串，"
					f"无法达到目标 {n_terms}。请将 L=1 改为 <= {_MAX_PAULI_L1_N7}。"
				)
			H_mat, H_dict, n_draws, n_active = build_random_fermionic_hamiltonian(
				NQ, L, rng, target_pauli_terms=n_terms
			)
			base_name = f"hamiltonian_fermionic_n{NQ}_L{L}_terms{n_terms}"
		elif TARGET_PAULI_TERMS is not None:
			H_mat, H_dict, n_draws, n_active = build_random_fermionic_hamiltonian(
				NQ, L, rng, target_pauli_terms=TARGET_PAULI_TERMS
			)
			base_name = f"hamiltonian_fermionic_n{NQ}_L{L}_terms{TARGET_PAULI_TERMS}"
		else:
			H_mat, H_dict, n_draws, n_active = build_random_fermionic_hamiltonian(
				NQ, L, rng, num_ladder_terms=NUM_LADDER_TERMS
			)
			base_name = f"hamiltonian_fermionic_n{NQ}_L{L}_ladder{NUM_LADDER_TERMS}"

		np.save(os.path.join(save_dir, f"{base_name}.npy"), H_mat.astype(np.float64))

		save_pauli_txt(H_dict, os.path.join(save_dir, f"{base_name}_pauli.txt"))

		save_json_pauli(H_dict, os.path.join(save_dir, f"{base_name}.json"))

		print(
			f"  L={L}: export {len(H_dict)} Pauli (Re coeff), {n_active} active (|c|), "
			f"{n_draws} ladder draws -> {base_name}.*"
		)

	psi = generate_random_state_vector(NQ, rng)
	vec_path = os.path.join(save_dir, f"state_fermionic_random_vector_n{NQ}.npy")
	np.save(vec_path, psi)
	print(f"  -> 随机量子态向量: {vec_path}")

	print("全部完成！")


if __name__ == "__main__":
	main()
