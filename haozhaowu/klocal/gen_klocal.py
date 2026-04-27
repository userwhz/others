import numpy as np
import os
from functools import reduce
from math import comb


# ================= 配置 =================
n = 7
k = 4
num_terms = 1000

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "hamil_class")
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

ogm_dir = os.path.join(save_dir, "ogm_inputs")
if not os.path.exists(ogm_dir):
	os.makedirs(ogm_dir)


# Pauli 矩阵定义
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli_dict = {"I": I, "X": X, "Y": Y, "Z": Z}
ogm_pauli_map = {"I": "0", "X": "1", "Y": "2", "Z": "3"}


def get_pauli_matrix(pauli_string):
	matrices = [pauli_dict[s] for s in pauli_string]
	return reduce(np.kron, matrices)


def save_pauli_txt(H_dict, filename):
	with open(filename, "w") as f:
		for p_str, coeff in H_dict.items():
			if abs(coeff) > 1e-9:
				f.write(f"{p_str}\n")
				f.write(f"({coeff.real:.16f}+0.0j)\n")


def save_ogm_txt(H_dict, filename):
	with open(filename, "w") as f:
		for p_str, coeff in H_dict.items():
			if abs(coeff) > 1e-9:
				indices = [ogm_pauli_map[ch] for ch in p_str]
				f.write(f"{coeff.real:.16f} {' '.join(indices)}\n")


def random_k_local_pauli_string(num_qubits, locality, rng):
	p_list = ["I"] * num_qubits
	positions = rng.choice(num_qubits, size=locality, replace=False)
	for pos in positions:
		p_list[pos] = rng.choice(["X", "Y", "Z"])
	return "".join(p_list)


def build_random_k_local_hamiltonian(num_qubits, locality, term_count, rng):
	max_terms = comb(num_qubits, locality) * (3 ** locality)
	if term_count > max_terms:
		raise ValueError(
			f"term_count={term_count} 超过了 n={num_qubits}, k={locality} 时最多可生成的唯一 Pauli 项数 {max_terms}。"
		)

	dim = 2 ** num_qubits
	H_matrix = np.zeros((dim, dim), dtype=complex)
	H_dict = {}

	while len(H_dict) < term_count:
		p_str = random_k_local_pauli_string(num_qubits, locality, rng)
		if p_str in H_dict:
			continue

		coeff = rng.uniform(-1.0, 1.0)
		H_dict[p_str] = coeff
		H_matrix += coeff * get_pauli_matrix(p_str)

	return H_matrix, H_dict


def generate_random_state_vector(num_qubits, rng):
	dim = 2 ** num_qubits
	real_part = rng.normal(size=dim)
	imag_part = rng.normal(size=dim)
	psi = real_part + 1j * imag_part
	psi /= np.linalg.norm(psi)
	return psi


def main():
	print(f"=== 生成 random k-local 数据: n={n}, k={k}, terms={num_terms} ===")
	rng = np.random.default_rng()

	# 1) 生成一个随机 k-local 哈密顿量（num_terms 个 Pauli 项）
	H_matrix, H_dict = build_random_k_local_hamiltonian(n, k, num_terms, rng)
	base_name = f"hamiltonian_klocal_random_n{n}_k{k}_terms{num_terms}"

	np.save(
		os.path.join(save_dir, f"{base_name}.npy"),
		H_matrix,
	)

	save_pauli_txt(
		H_dict,
		os.path.join(save_dir, f"{base_name}_pauli.txt"),
	)

	save_ogm_txt(
		H_dict,
		os.path.join(ogm_dir, f"ogm_{base_name}.txt"),
	)

	print(f"  -> 已生成哈密顿量 (项数: {len(H_dict)})，并导出 OGM")

	# 2) 生成一个随机量子态文件
	psi = generate_random_state_vector(n, rng)
	rho = np.outer(psi, np.conjugate(psi))
	np.save(os.path.join(save_dir, f"state_klocal_random_vector_n{n}.npy"), psi)
	np.save(os.path.join(save_dir, f"state_klocal_random_rho_n{n}.npy"), rho)
	print("  -> 已生成随机量子态向量与密度矩阵")

	print("全部完成！")


if __name__ == "__main__":
	main()
