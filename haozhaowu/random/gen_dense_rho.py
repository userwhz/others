import numpy as np
import os

# ================= 配置 =================
save_dir = "hamil_class"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 你可以在这里修改要生成的比特数列表
# qubit_counts = [3, 4, 5, 6]
qubit_counts = [7]

# ================= 主循环 =================
for n in qubit_counts:
    dim = 2 ** n

    print(f"=== 正在生成 {n} 比特稠密量子态 (Dense) ===")
    print(f"  -> 维度: {dim}, 所有元素均为非零随机值")

    # 1. 生成稠密复数向量
    # 使用标准正态分布 (randn) 生成实部和虚部
    # 这种方法生成的态在希尔伯特空间中是均匀分布的 (Haar Random)
    # 相比于只选几个位置，这里直接生成 dim 个复数
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)

    # 2. 归一化 (Normalization)
    # 这一步非常重要，确保 <psi|psi> = 1
    psi = psi / np.linalg.norm(psi)

    # 3. 计算密度矩阵 rho = |psi><psi|
    # 结果是一个 dim x dim 的全满矩阵
    rho = np.outer(psi, np.conjugate(psi))

    # 4. 保存文件 (.npy)
    # 文件名中的 sparse 改为 dense
    vec_path = os.path.join(save_dir, f"state_dense_vector_{n}.npy")
    rho_path = os.path.join(save_dir, f"state_dense_rho_{n}.npy")

    np.save(vec_path, psi)
    np.save(rho_path, rho)

    print(f"  -> 态向量已保存: {vec_path}")
    print(f"  -> 密度矩阵已保存: {rho_path}")

print("\n全部稠密量子态生成完成！")
