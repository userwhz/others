import numpy as np
from scipy.sparse import coo_matrix
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

    # 生成所有 Pauli 组合 (例如 4比特就是 IIII 到 ZZZZ)
    pauli_combs = itertools.product(pauli_labels, repeat=n_qubits)

    with open(filename, 'w') as f:
        count = 0
        for p_tuple in pauli_combs:
            p_string = "".join(p_tuple)

            # 1. 生成 Pauli 矩阵 P
            P_matrix = get_pauli_matrix(p_string)

            # 2. 计算系数 c = Tr(H @ P) / dim
            # 因为 H 是实对称矩阵，系数理论上是实数
            val = np.dot(H, P_matrix).trace()
            coeff = val / dim

            # 3. 写入文件 (过滤极小值)
            if abs(coeff) > 1e-9:
                f.write(f"{p_string}\n")
                # 保持你要的格式: (实部+0.0j)
                f.write(f"({coeff.real:.16f}+0.0j)\n")
                count += 1
    print(f"  -> 分解完成，保存至 {filename} (非零项: {count})")


# ==================== 2. 主循环 (4, 5, 6 比特) ====================

qubit_counts = [7]
# qubit_counts = [3，4, 5, 6]

for n in qubit_counts:
    print(f"\n=== 处理 {n} 比特系统 ===")
    dim = 2 ** n

    # --- 生成稀疏矩阵 (基于你的逻辑) ---
    # 为了让矩阵不那么空，非零元素数量设为维度的2倍 (你可以改回固定的 size=4)
    nnz = dim * 2

    rows = np.random.randint(0, dim, size=nnz)
    cols = np.random.randint(0, dim, size=nnz)
    data = np.random.randn(nnz)  # 随机生成实数

    # 创建稀疏矩阵并转为稠密
    H_temp = coo_matrix((data, (rows, cols)), shape=(dim, dim)).toarray()

    # 对称化 (变成厄米矩阵)
    H = (H_temp + H_temp.T) / 2

    # --- 保存 .npy ---
    npy_name = os.path.join(save_dir, f"hamiltonian_sparse_{n}.npy")
    np.save(npy_name, H)
    print(f"  -> 矩阵已保存: {npy_name}")

    # --- 保存 .txt (Pauli 分解) ---
    txt_name = os.path.join(save_dir, f"hamiltonian_sparse_{n}_pauli.txt")
    decompose_and_save(H, n, txt_name)

print("\n全部搞定！")