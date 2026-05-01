import numpy as np
from itertools import product


def hamiltonian_to_pauli_strings(H, tol=1e-10):
    """
    将哈密顿量矩阵转换为泡利字符串表示

    参数:
        H (np.ndarray): 2^n x 2^n的哈密顿量矩阵
        tol (float): 系数截断阈值

    返回:
        list: 格式为 [(pauli_str, coefficient), ...] 的列表
    """
    # 定义泡利矩阵及对应符号
    paulis = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

    # 验证矩阵维度
    n_qubits = int(np.log2(H.shape[0]))
    assert 2 ** n_qubits == H.shape[0], "矩阵维度必须是2的幂次"

    results = []

    # 遍历所有可能的泡利字符串组合
    for ops in product(paulis.keys(), repeat=n_qubits):
        # 构造张量积矩阵
        pauli_mat = paulis[ops[0]]
        for op in ops[1:]:
            pauli_mat = np.kron(pauli_mat, paulis[op])

        # 计算系数
        coeff = np.trace(pauli_mat @ H) / (2 ** n_qubits)

        # 处理复数显示
        coeff_real = np.real_if_close(coeff).item()
        if abs(coeff.imag) > tol:
            coeff = complex(coeff)
        else:
            coeff = coeff_real

        # 过滤微小系数
        if np.abs(coeff) > tol:
            # 生成泡利字符串（注意量子位顺序）
            pauli_str = ''.join(ops)[::-1]  # 调整顺序使左边对应第一个量子位
            results.append((pauli_str, coeff))

    return results


def format_output(pauli_terms):
    """格式化输出为指定样式"""
    output = []
    for pauli_str, coeff in pauli_terms:
        # 处理复数格式
        if isinstance(coeff, complex):
            real_part = f"{coeff.real:.16f}".rstrip('0').rstrip('.')
            imag_part = f"{abs(coeff.imag):.16f}".rstrip('0').rstrip('.')
            sign = '+' if coeff.imag >= 0 else '-'
            coeff_str = f"({real_part}{sign}{imag_part}j)"
        else:
            real_part = f"{coeff:.16f}".rstrip('0').rstrip('.')
            coeff_str = f"({real_part}+0j)"

        output.append(f"{pauli_str}\n{coeff_str}")
    return '\n'.join(output)


def save_hamiltonian_txt(terms, filename):
    """将结果保存到txt文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for pauli_str, coeff in terms:
            # 格式化复数显示
            if coeff.imag == 0:
                coeff_str = f"({coeff.real:.16f}+0j)"
            else:
                coeff_str = f"({coeff.real:.16f}{coeff.imag:+.16f}j)"

            # 写入文件（自动去除多余零）
            f.write(f"{pauli_str}\n")
            f.write(f"{coeff_str.replace('+0j', '+0j').rstrip('0').rstrip('.')}\n\n")


# 示例用法
if __name__ == "__main__":
    # 创建测试哈密顿量（示例）
    H = np.load('SlaterHb_8.npy')
    # 转换为泡利项
    pauli_terms = hamiltonian_to_pauli_strings(H)

    # 生成格式化输出
    formatted = format_output(pauli_terms)
    save_hamiltonian_txt(pauli_terms, "SlaterHb_8.txt")
    print(formatted)
