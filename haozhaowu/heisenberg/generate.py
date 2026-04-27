import numpy as np
import os
import itertools
from functools import reduce

# ================= 配置 =================
save_dir = "hamil_class"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# qubit_counts = [3, 4, 5, 6]
qubit_counts = [9]

# Pauli 矩阵定义
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def get_pauli_matrix(pauli_string):
    matrices = [pauli_dict[s] for s in pauli_string]
    return reduce(np.kron, matrices)


def save_pauli_txt(H_dict, filename):
    with open(filename, 'w') as f:
        for p_str, coeff in H_dict.items():
            if abs(coeff) > 1e-9:
                f.write(f"{p_str}\n")
                f.write(f"({coeff.real:.16f}+0.0j)\n")


# ================= 主循环 =================
for n in qubit_counts:
    print(f"\n=== 生成 {n} 比特 Heisenberg 模型 ===")
    dim = 2 ** n

    # 1. 构建哈密顿量 (字典形式便于保存txt, 矩阵形式便于保存npy)
    # H = sum(X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) + sum(h_i Z_i)

    H_matrix = np.zeros((dim, dim), dtype=complex)
    H_dict = {}

    # 耦合常数 J=1
    J = 1.0
    # 随机磁场 h in [0, 1]
    h_fields = np.random.rand(n)

    # (A) 生成相互作用项 XX, YY, ZZ
    for i in range(n - 1):  # 0 到 n-2
        for op in ['X', 'Y', 'Z']:
            # 构造 Pauli 串，例如 IXZI...
            p_list = ['I'] * n
            p_list[i] = op
            p_list[i + 1] = op
            p_str = "".join(p_list)

            # 添加到字典
            H_dict[p_str] = J

            # 添加到矩阵
            H_matrix += J * get_pauli_matrix(p_str)

    # (B) 生成磁场项 Z
    for i in range(n):
        p_list = ['I'] * n
        p_list[i] = 'Z'
        p_str = "".join(p_list)

        coeff = h_fields[i]
        H_dict[p_str] = coeff
        H_matrix += coeff * get_pauli_matrix(p_str)

    # 2. 生成基态 (Ground State) 用于测量
    # 求解本征值，取最小本征值对应的向量
    eigvals, eigvecs = np.linalg.eigh(H_matrix)
    psi = eigvecs[:, 0]  # 最小本征向量
    rho = np.outer(psi, np.conjugate(psi))

    # 3. 保存文件
    # 保存哈密顿量矩阵
    np.save(os.path.join(save_dir, f"hamiltonian_heisenberg_{n}.npy"), H_matrix)
    # 保存哈密顿量 Pauli txt
    save_pauli_txt(H_dict, os.path.join(save_dir, f"hamiltonian_heisenberg_{n}_pauli.txt"))

    # 保存量子态 (基态)
    np.save(os.path.join(save_dir, f"state_heisenberg_vector_{n}.npy"), psi)
    np.save(os.path.join(save_dir, f"state_heisenberg_rho_{n}.npy"), rho)

    print(f"  -> 已生成 Heisenberg 模型数据 (项数: {len(H_dict)})")

print("\n全部完成！")