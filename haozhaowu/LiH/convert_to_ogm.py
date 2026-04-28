import os
import numpy as np


def convert_pauli_to_ogm(input_file, output_file):
    """
    将 Pauli 格式 (Pauli串 + 复数系数) 转换为 OGM 输入格式.
    Mapping: I->0, X->1, Y->2, Z->3
    """
    pauli_map = {'I': '0', 'X': '1', 'Y': '2', 'Z': '3'}

    if not os.path.exists(input_file):
        print(f"❌ [跳过] 找不到文件: {input_file}")
        return

    try:
        with open(input_file, 'r') as f:
            # 过滤空行
            lines = [line.strip() for line in f.readlines() if line.strip()]

        with open(output_file, 'w') as f_out:
            for i in range(0, len(lines), 2):
                if i + 1 >= len(lines): break

                p_str = lines[i]  # e.g., "ZIIZ"
                c_str = lines[i + 1]  # e.g., "(0.288+0j)"

                try:
                    # 提取实部系数
                    coeff_val = complex(c_str).real
                    # 映射 Pauli 字符到整数
                    indices = [pauli_map[char] for char in p_str]
                except Exception as e:
                    print(f"⚠️ 解析错误: {p_str} {c_str} -> {e}")
                    continue

                # 写入: 系数 idx1 idx2 ...
                indices_str = " ".join(indices)
                f_out.write(f"{coeff_val} {indices_str}\n")

        print(f"✅ 转换成功: {os.path.basename(output_file)}")

    except Exception as e:
        print(f"❌ 处理错误 {input_file}: {e}")


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 定义文件夹路径
    # 假设你的 hamiltonian_LiH_xxx.txt 文件在 hamil_class 文件夹下
    input_dir = "hamil_class"
    output_dir = "hamil_class/ogm_inputs"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 定义要处理的键长列表
    # 0.8 到 2.0，步长 0.2
    # numpy.arange(start, stop, step) 注意不包含 stop，所以写 2.1
    range1 = np.arange(2.2)
    # 加上 2.5 和 3.0
    distances = np.concatenate([range1, [2.2, 2.4, 2.6, 2.8, 3.2, 3.4, 3.6, 3.8, 4.0]])

    # 排序并去重，保留1位小数防止浮点误差
    distances = sorted(list(set([round(x, 1) for x in distances])))

    print(f"=== 开始转换 LiH 数据 ===")
    print(f"待处理键长: {distances}")

    for dist in distances:
        dist_str = f"{dist:.1f}"  # 格式化为 "1.8"

        # 构造文件名
        # 输入: hamiltonian_LiH_1.8_pauli.txt
        in_name = f"hamiltonian_LiH_{dist_str}_pauli.txt"
        in_path = os.path.join(input_dir, in_name)

        # 输出: ogm_LiH_1.8.txt
        out_name = f"ogm_LiH_{dist_str}.txt"
        out_path = os.path.join(output_dir, out_name)

        convert_pauli_to_ogm(in_path, out_path)

    print("\n所有 LiH 转换任务完成！")