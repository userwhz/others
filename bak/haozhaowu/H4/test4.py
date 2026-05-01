import os

import numpy as np
import pickle
from collections import OrderedDict


def parse_pauli_string(pauli_str, num_qubits):
    """将泡利字符串转换为矩阵形式"""
    # 泡利矩阵定义
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # 初始化单位矩阵
    matrix = np.eye(1)

    # 对每个量子比特应用对应的泡利矩阵
    for char in pauli_str:
        matrix = np.kron(matrix, pauli_dict[char])

    return matrix


def qubit_operator_to_matrix(qubit_operator, num_qubits):
    """将QubitOperator对象转换为矩阵形式"""
    hamiltonian_matrix = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)

    # 处理QubitOperator的terms
    for pauli_term, coeff in qubit_operator.terms.items():
        # 创建泡利字符串
        pauli_string = ['I'] * num_qubits
        for qubit_idx, pauli_op in pauli_term:
            if qubit_idx < num_qubits:
                pauli_string[qubit_idx] = pauli_op

        pauli_str = ''.join(pauli_string)
        pauli_matrix = parse_pauli_string(pauli_str, num_qubits)
        hamiltonian_matrix += coeff * pauli_matrix

    return hamiltonian_matrix


def save_hamiltonian_in_attachment_format(qubit_operator, num_qubits, filename):
    """保存成附件中格式的哈密顿量"""

    # 构建泡利项字典（类似于附件格式）
    pauli_terms = OrderedDict()

    # 处理QubitOperator对象
    for pauli_term, coeff in qubit_operator.terms.items():
        # 创建泡利字符串
        pauli_string = ['I'] * num_qubits
        for qubit_idx, pauli_op in pauli_term:
            if qubit_idx < num_qubits:
                pauli_string[qubit_idx] = pauli_op

        pauli_str = ''.join(pauli_string)
        pauli_terms[pauli_str] = coeff

    # 保存为文本文件
    with open(filename, 'w') as f:
        for pauli_str, coeff in pauli_terms.items():
            f.write(f"{pauli_str}\n")
            f.write(f"({coeff.real}+{coeff.imag}j)\n")

    return pauli_terms


# 主程序
def main():
    iterm = 0.4
    # 注意：浮点数循环比较建议稍微放宽一点上限，防止精度问题导致少跑一次
    while iterm <= 4.55:
        # 1. 构造文件名
        filename = f'H4_{iterm:.1f}_sto-3g.pkl'

        # 2. 【关键修改】检查文件是否存在
        if not os.path.exists(filename):
            print(f"⚠️  文件 {filename} 不存在，跳过...")
            iterm += 0.1
            # continue  # 直接进入下一次循环

        try:
            print(f"\n>>> 开始处理: {filename}")

            # 加载pkl文件
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            print("数据键值:", data.keys())
            print("量子比特数:", data['num_qubits'])
            print("FCI能量:", data['FCI_val'])

            # 1. 转换为矩阵形式的哈密顿量
            print("1. 计算矩阵形式的哈密顿量...")
            hamiltonian_matrix = qubit_operator_to_matrix(data['qubit_hamiltonian'], data['num_qubits'])

            # 验证：计算基态能量
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)

            # 基态是能量最低的状态
            ground_state_energy = eigenvalues[0]
            ground_state_vector = eigenvectors[:, 0]

            # 密度矩阵 = |ψ⟩⟨ψ|
            density_matrix = np.outer(ground_state_vector, ground_state_vector.conj())

            # 保存 .npy 文件
            np.save(f'ground_state_H2_{iterm:.1f}_sto-3g.npy', ground_state_vector)
            np.save(f'ground_state_H2_{iterm:.1f}_sto-3g_density_matrix.npy', density_matrix)

            print(f"基态能量: {ground_state_energy}")
            print(f"FCI能量: {data['FCI_val']}")
            print(f"差异: {abs(ground_state_energy - data['FCI_val'])}")

            # 2. 保存成附件格式 (.txt)
            print("2. 保存为附件格式...")
            pauli_terms = save_hamiltonian_in_attachment_format(
                data['qubit_hamiltonian'],
                data['num_qubits'],
                f'H2_{iterm:.1f}_sto-3g_sg.txt'
            )
            print(f"保存了 {len(pauli_terms)} 个泡利项")

            # 3. 可选：保存矩阵到文件
            np.save(f'H2_{iterm:.1f}_sto-3g.npy', hamiltonian_matrix)
            print("矩阵已保存为 .npy")

        except Exception as e:
            # 捕获其他可能的错误（如文件损坏），保证循环不中断
            print(f"❌ 处理 {filename} 时发生未知错误: {e}")

        # 循环变量自增
        iterm += 0.1
        # 4. 可选：生成更详细的报告
        # generate_detailed_report(data, hamiltonian_matrix)


def generate_detailed_report(data, hamiltonian_matrix):
    """生成详细报告"""
    with open('hamiltonian_report.txt', 'w') as f:
        f.write("=== 哈密顿量分析报告 ===\n\n")
        f.write(f"系统信息:\n")
        f.write(f"- 量子比特数: {data['num_qubits']}\n")
        f.write(f"- 电子数: {data['n_elec']}\n")
        f.write(f"- 粒子数: {data['n_particles']}\n")
        f.write(f"- FCI能量: {data['FCI_val']}\n")
        f.write(f"- 精确对角化能量: {data['Precise_diagonalization_energy']}\n\n")

        f.write("哈密顿量矩阵特性:\n")
        f.write(f"- 矩阵维度: {hamiltonian_matrix.shape}\n")
        f.write(f"- 厄米性检查: {np.allclose(hamiltonian_matrix, hamiltonian_matrix.conj().T)}\n")

        eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix)
        f.write(f"- 能谱范围: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]\n")
        f.write(f"- 基态能量: {eigenvalues[0]:.6f}\n")
        f.write(f"- 第一激发态能量: {eigenvalues[1]:.6f}\n")
        f.write(f"- 能隙: {eigenvalues[1] - eigenvalues[0]:.6f}\n\n")

        f.write("主要泡利项:\n")
        # 解析qubit_hamiltonian并排序
        terms = []
        qubit_ham_str = data['qubit_hamiltonian']
        for term in qubit_ham_str.split('+'):
            term = term.strip()
            if not term:
                continue

            if '[' in term and ']' in term:
                coeff_str, pauli_part = term.split('[', 1)
                pauli_part = pauli_part.split(']', 1)[0]
                coeff = float(coeff_str.strip())
                terms.append((abs(coeff), coeff, pauli_part))
            else:
                coeff = float(term.strip())
                terms.append((abs(coeff), coeff, "I" * data['num_qubits']))

        # 按绝对值排序
        terms.sort(reverse=True)

        for abs_coeff, coeff, pauli in terms[:10]:  # 显示前10个最重要的项
            f.write(f"  {coeff:12.8f} {pauli}\n")


if __name__ == "__main__":
    main()