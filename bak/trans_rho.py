import numpy as np


def read_quantum_state(filename):
    """读取量子态文件，返回复数数组"""
    with open(filename, 'r') as f:
        # 去除空行并解析复数
        states = [complex(line.strip()) for line in f if line.strip()]
    return np.array(states, dtype=complex)


def state_to_matrix(state_vector):
    """转换为列向量形式 (N x 1 矩阵)"""
    return state_vector.reshape(-1, 1)


def state_to_density_matrix(state_vector):
    """转换为密度矩阵形式 (N x N 矩阵)"""
    return np.outer(state_vector, state_vector.conj())


if __name__ == "__main__":
    # 从文件读取量子态
    input_file = "ground_state_SlaterHb_8.txt"  # 替换为你的文件名
    psi = read_quantum_state(input_file)

    # 验证维度 (应为 2^8 = 256)
    assert len(psi) == 256, "输入态维度应为256"

    # 转换为列向量形式
    column_matrix = state_to_matrix(psi)

    # 转换为密度矩阵形式
    density_matrix = state_to_density_matrix(psi)
    np.save("ground_state_SlaterHb_8.npy",density_matrix)