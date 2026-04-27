import numpy as np
import os

# ================= 配置 =================
save_dir = "hamil_class"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

qubit_counts = [7]
# qubit_counts = [4, 5, 6]

# ================= 主循环 =================
for n in qubit_counts:
    dim = 2 ** n

    # 修正逻辑：非零元素的个数 = 比特数 n
    num_non_zero = n

    print(f"=== 正在生成 {n} 比特稀疏量子态 ===")
    print(f"  -> 维度: {dim}, 非零元素个数: {num_non_zero}")

    # 1. 初始化全零向量
    psi = np.zeros(dim, dtype=np.complex128)

    # 2. 随机选择 n 个位置作为非零值
    nonzero_indices = np.random.choice(dim, size=num_non_zero, replace=False)

    # 3. 随机复数赋值 (模拟之前的 rand * 200 逻辑)
    # 实部和虚部都在 [0, 200) 之间
    psi[nonzero_indices] = np.random.rand(num_non_zero) * 200 + 1j * np.random.rand(num_non_zero) * 200

    # 4. 归一化
    psi = psi / np.linalg.norm(psi)

    # 5. 计算密度矩阵 rho = |psi><psi|
    # 注意：这里保存完整的复数矩阵，保留相位信息
    rho = np.outer(psi, np.conjugate(psi))

    # 6. 保存文件 (.npy)
    vec_path = os.path.join(save_dir, f"state_sparse_vector_{n}.npy")
    rho_path = os.path.join(save_dir, f"state_sparse_rho_{n}.npy")

    np.save(vec_path, psi)
    np.save(rho_path, rho)

    print(f"  -> 态向量已保存: {vec_path}")
    print(f"  -> 密度矩阵已保存: {rho_path}")

print("\n全部完成！")
