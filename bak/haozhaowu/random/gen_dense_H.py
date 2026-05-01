import numpy as np
import os
import itertools
from functools import reduce

# ==================== 1. 基础设置 ====================
save_dir = "hamil_class"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义 Pauli 矩阵
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
pauli_labels = ['I', 'X', 'Y', 'Z']


def get_pauli_matrix(pauli_string):
    """生成 Pauli 串对应的矩阵"""
    matrices = [pauli_dict[s] for s in pauli_string]
    return reduce(np.kron, matrices)


def decompose_and_save(H, n_qubits, filename):
    """计算 Pauli 系数并保存"""
    dim = 2 ** n_qubits
    print(f"  -> 正在分解 {n_qubits} 比特哈密顿量 (共 {4 ** n_qubits} 项)...")

    # 生成所有 Pauli 组合
    pauli_combs = itertools.product(pauli_labels, repeat=n_qubits)

    with open(filename, 'w') as f:
        count = 0
        for p_tuple in pauli_combs:
            p_string = "".join(p_tuple)

            # 1. 生成 Pauli 矩阵 P
            P_matrix = get_pauli_matrix(p_string)

            # 2. 计算系数 c = Tr(H @ P) / dim
            val = np.dot(H, P_matrix).trace()
            coeff = val / dim

            # 3. 写入文件 (过滤极小值)
            # 对于稠密矩阵，通常绝大多数项都是非零的，但为了文件整洁，还是过滤掉极小的噪音
            if abs(coeff) > 1e-9:
                f.write(f"{p_string}\n")
                # 保持格式: (实部+0.0j)
                f.write(f"({coeff.real:.16f}+0.0j)\n")
                count += 1
    print(f"  -> 分解完成，保存至 {filename} (非零项: {count})")


# ==================== 2. 主循环 (3, 4, 5, 6 比特) ====================

# 你可以在这里修改要生成的比特数列表
# qubit_counts = [3, 4, 5, 6]
qubit_counts = [7]

for n in qubit_counts:
    print(f"\n=== 处理 {n} 比特系统 (稠密 Dense) ===")
    dim = 2 ** n

    # --- 生成稠密矩阵 (Dense Matrix) ---
    # np.random.randn(dim, dim) 生成一个全填充的随机矩阵
    # 如果需要复数随机矩阵，可以用: np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    # 这里保持和你稀疏矩阵一致的实数逻辑：
    H_temp = np.random.randn(dim, dim)

    # 对称化 (使其成为厄米矩阵/实对称矩阵)
    # 这一步保证了 H = H^T，物理上保证能量为实数
    H = (H_temp + H_temp.T) / 2

    # --- 保存 .npy ---
    # 文件名改为 hamiltonian_dense_{n}
    npy_name = os.path.join(save_dir, f"hamiltonian_dense_{n}.npy")
    np.save(npy_name, H)
    print(f"  -> 矩阵已保存: {npy_name}")

    # --- 保存 .txt (Pauli 分解) ---
    txt_name = os.path.join(save_dir, f"hamiltonian_dense_{n}_pauli.txt")
    decompose_and_save(H, n, txt_name)

print("\n全部稠密哈密顿量生成搞定！")