"""
从 *_pauli.txt（与 gen_fermionic / beh2.py 相同格式：Pauli 串一行、系数一行）
生成稠密 *.npy。

优先用 Qiskit：见文件顶部 `from qiskit.quantum_info import SparsePauliOp` 与 `_HAS_QISKIT`，
`SparsePauliOp.to_matrix`（与 h2o/beh2 稠密导出一致、快）；
若导入失败则回退到与 gen_fermionic.py 相同的 NumPy Kronecker 展开。
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
	from qiskit.quantum_info import SparsePauliOp

	_HAS_QISKIT = True
except ImportError:
	SparsePauliOp = None  # type: ignore[misc, assignment]
	_HAS_QISKIT = False

# 无 qiskit 时与 gen_fermionic.py 一致
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli_dict = {"I": I, "X": X, "Y": Y, "Z": Z}


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


def _parse_complex_coeff(line: str) -> complex:
	s = line.strip()
	if s.startswith("(") and s.endswith(")"):
		s = s[1:-1]
	return complex(s.replace(" ", ""))


def load_pauli_txt(path: str) -> List[Tuple[str, complex]]:
	with open(path, "r", encoding="utf-8") as f:
		lines = [ln.strip() for ln in f if ln.strip()]
	if len(lines) % 2 != 0:
		raise ValueError(f"{path}: 期望偶数行（Pauli 串 + 系数交替），得到 {len(lines)} 行")
	terms: List[Tuple[str, complex]] = []
	for i in range(0, len(lines), 2):
		p_str, coeff_line = lines[i], lines[i + 1]
		terms.append((p_str, _parse_complex_coeff(coeff_line)))
	return terms


def pauli_terms_to_dense(terms: List[Tuple[str, complex]]) -> Tuple[np.ndarray, str]:
	merged: Dict[str, complex] = defaultdict(complex)
	for p, c in terms:
		merged[p] += c
	pairs = [(p, complex(v)) for p, v in merged.items() if abs(v) > 1e-15]
	if not pairs:
		raise ValueError("无有效 Pauli 项")
	nq = len(pairs[0][0])
	for p, _ in pairs:
		if len(p) != nq:
			raise ValueError("Pauli 串长度不一致")

	if _HAS_QISKIT:
		H = SparsePauliOp.from_list(pairs).to_matrix(sparse=False)
		backend = "qiskit"
	else:
		H = pauli_dict_to_dense_matrix(dict(pairs), nq)
		backend = "numpy"

	if np.max(np.abs(H.imag)) < 1e-10:
		H_out = np.asarray(H.real, dtype=np.float64)
	else:
		H_out = np.asarray(H, dtype=np.complex128)
	return H_out, backend


def default_out_path(pauli_txt: str) -> str:
	if pauli_txt.endswith("_pauli.txt"):
		return pauli_txt[: -len("_pauli.txt")] + ".npy"
	return os.path.splitext(pauli_txt)[0] + ".npy"


def print_hamiltonian_state_theory(
	H: np.ndarray,
	state_path: Optional[str],
	*,
	skip: bool = False,
) -> None:
	"""打印 H 的 Hermiticity、迹、最小本征值；若给定态文件则打印范数与 <psi|H|psi>。"""
	if skip:
		return
	from scipy.sparse.linalg import LinearOperator, eigsh

	H = np.asarray(H)
	n = H.shape[0]
	print("\n========== 哈密顿量与量子态：理论量 ==========")
	if H.shape != (n, n):
		raise ValueError(f"H 应为方阵，得到 shape={H.shape}")

	if np.iscomplexobj(H):
		herm_res = np.linalg.norm(H - H.conj().T)
	else:
		herm_res = np.linalg.norm(H - H.T)
	print(f"H 与 Hermitian/对称 的 Frobenius 偏差: {herm_res:.3e}（应接近 0）")
	print(f"tr(H) = {float(np.trace(H)):.12f}")

	Hop = np.asarray(H, dtype=np.complex128 if np.iscomplexobj(H) else np.float64)

	def matvec(x: np.ndarray) -> np.ndarray:
		return Hop @ x

	lo = LinearOperator((n, n), matvec=matvec, dtype=Hop.dtype)
	w, _ = eigsh(lo, k=1, which="SA", tol=1e-8)
	e0 = float(np.real(w[0]))
	print(f"最小本征值 E0（基态能量，Hartree）= {e0:.12f}")

	if not state_path:
		print("未指定态文件，跳过 <psi|H|psi>。")
		return
	if not os.path.isfile(state_path):
		print(f"态文件不存在，跳过: {state_path}")
		return

	psi = np.load(state_path)
	psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
	if psi.shape[0] != n:
		raise ValueError(f"态维度 {psi.shape[0]} 与 H 维度 {n} 不一致")
	norm = float(np.linalg.norm(psi))
	psi_n = psi / norm if norm > 0 else psi
	e_exp = float(np.real(np.vdot(psi_n, Hop @ psi_n)))
	print(f"态文件: {state_path}")
	print(f"||psi||（载入未归一化）= {norm:.12f}（理论应为 1）")
	print(f"<psi|H|psi>（归一化后, Hartree）= {e_exp:.12f}")
	print(f"<psi|H|psi> - E0 = {e_exp - e0:.3e} Hartree（若为基态应接近 0）")


def main() -> None:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	parser = argparse.ArgumentParser(description="从 _pauli.txt 生成稠密 Hamiltonian .npy")
	parser.add_argument(
		"--pauli-txt",
		type=str,
		default=os.path.join(base_dir, "hamiltonian_BeH2_sto3g_14_pauli.txt"),
		help="输入 Pauli 文本路径",
	)
	parser.add_argument(
		"--out-npy",
		type=str,
		default=None,
		help="输出 .npy；默认去掉输入文件名中的 _pauli 后缀",
	)
	parser.add_argument(
		"--state-npy",
		type=str,
		default=os.path.join(base_dir, "state_14q.npy"),
		help="用于计算 <psi|H|psi> 等的态向量 .npy；不存在则跳过态相关输出",
	)
	parser.add_argument(
		"--skip-theory",
		action="store_true",
		help="不打印 Hermitian / E0 / 态期望值",
	)
	args = parser.parse_args()
	pauli_path = os.path.abspath(args.pauli_txt)
	out_path = os.path.abspath(args.out_npy) if args.out_npy else default_out_path(pauli_path)

	terms = load_pauli_txt(pauli_path)
	H, backend = pauli_terms_to_dense(terms)
	np.save(out_path, H)
	print(f"读取 {len(terms)} 项 Pauli 行对；稠密构造: {backend}")
	print(f"已保存: {out_path}  shape={H.shape}  dtype={H.dtype}")
	print_hamiltonian_state_theory(
		H,
		os.path.abspath(args.state_npy) if args.state_npy else None,
		skip=args.skip_theory,
	)


if __name__ == "__main__":
	main()
