import numpy as np
import os
import qiskit_nature

# 1. 全局设置：强制使用新版对象
qiskit_nature.settings.use_pauli_sum_op = False

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

# ================= 配置区域 =================
save_dir = "hamil_class"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 扫描范围 (Å)
distances = [2.2, 2.4, 2.6, 2.8, 3.2, 3.4, 3.6, 3.8, 4.0]
summary_file = os.path.join(save_dir, "LiH_PES_exact.txt")


# ================= 核心处理函数 =================
def generate_lih_at_distance(dist):
    print(f"\n>>> 处理键长: {dist} Å")
    atom_str = f"Li 0.0 0.0 0.0; H 0.0 0.0 {dist}"

    try:
        # 1. PySCF 驱动计算
        driver = PySCFDriver(atom=atom_str, basis='sto-3g', charge=0, spin=0)
        problem = driver.run()

        # 2. 活性空间变换 (3个空间轨道 -> 6个自旋轨道 -> 6个比特)
        transformer = ActiveSpaceTransformer(
            num_electrons=2,
            num_spatial_orbitals=3
        )
        problem_reduced = transformer.transform(problem)
        hamiltonian = problem_reduced.hamiltonian.second_q_op()

        # 3. Jordan-Wigner 映射 (稳定生成 6 比特)
        mapper = JordanWignerMapper()
        final_op = mapper.map(hamiltonian)

        if hasattr(final_op, "primitive"):
            final_op = final_op.primitive

        # 4. 计算基态 (Exact Diagonalization)
        # 获取稠密矩阵
        H_matrix = final_op.to_matrix()

        # 求解特征值分解
        vals, vecs = np.linalg.eigh(H_matrix)

        # 提取基态 (能量最低对应的向量)
        ground_E = vals[0]
        ground_vec = vecs[:, 0]

        # 计算密度矩阵 rho = |psi><psi|
        ground_rho = np.outer(ground_vec, np.conj(ground_vec))

        # 5. 保存 4 个文件
        dist_str = f"{dist:.1f}"

        # --- 文件 1: Pauli Strings (.txt) ---
        txt_path = os.path.join(save_dir, f"hamiltonian_LiH_{dist_str}_pauli.txt")
        with open(txt_path, 'w') as f:
            for pauli, coeff in zip(final_op.paulis, final_op.coeffs):
                if abs(coeff) > 1e-9:
                    f.write(f"{str(pauli)}\n")
                    # 格式: (0.123+0.0j)
                    f.write(f"({coeff.real:.16f}+0.0j)\n")

        # --- 文件 2: 哈密顿量矩阵 (.npy) ---
        np.save(os.path.join(save_dir, f"hamiltonian_LiH_{dist_str}.npy"), H_matrix)

        # --- 文件 3: 状态向量 (.npy) ---
        np.save(os.path.join(save_dir, f"state_LiH_vector_{dist_str}.npy"), ground_vec)

        # --- 文件 4: 密度矩阵 (.npy) ---
        np.save(os.path.join(save_dir, f"state_LiH_rho_{dist_str}.npy"), ground_rho)

        print(f"  -> ✅ 生成成功 (6 Qubits)。基态能量: {ground_E:.6f} Ha")
        return dist, ground_E

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 处理 {dist} 时出错: {e}")
        return None, None


# ================= 主循环 =================
results = []
print("=== 开始 LiH 势能面扫描 (生成 4 个文件/键长) ===")

for d in distances:
    d_val, e_val = generate_lih_at_distance(d)
    if d_val is not None:
        results.append((d_val, e_val))

# 保存能量基准值
with open(summary_file, 'w') as f:
    f.write("Distance(A)\tEnergy(Hartree)\n")
    for d, e in results:
        f.write(f"{d:.1f}\t{e:.8f}\n")

print(f"\n全部完成！所有文件已保存在 {save_dir}/ 目录下。")