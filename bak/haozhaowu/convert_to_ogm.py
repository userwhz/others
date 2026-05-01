import os


def convert_pauli_to_ogm(input_file, output_file):
    """
    将标准 Pauli 格式 (Pauli串 + 复数系数) 转换为 OGM 输入格式 (系数 + 整数索引).

    Mapping:
    I -> 0
    X -> 1
    Y -> 2
    Z -> 3
    """

    # 定义映射关系
    pauli_map = {
        'I': '0',
        'X': '1',
        'Y': '2',
        'Z': '3'
    }

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到输入文件 {input_file}")
        return

    try:
        with open(input_file, 'r') as f:
            # 读取所有行，去除空白字符
            # 过滤掉空行，这样即使原文件有空行分隔也能正确处理
            lines = [line.strip() for line in f.readlines() if line.strip()]

        with open(output_file, 'w') as f_out:
            # 每次步进 2 行 (一行 Pauli 串，一行系数)
            for i in range(0, len(lines), 2):
                if i + 1 >= len(lines):
                    break  # 防止最后一行不完整

                p_str = lines[i]  # 例如 "ZIIZ"
                c_str = lines[i + 1]  # 例如 "(0.2883037+0j)"

                # 1. 处理系数: 解析复数并取实部
                try:
                    # Python 的 complex() 函数可以直接解析 "(a+bj)" 格式
                    coeff_val = complex(c_str).real
                except ValueError:
                    print(f"⚠️ 警告: 无法解析系数 '{c_str}'，跳过该行。")
                    continue

                # 2. 处理 Pauli 串: 映射为数字
                # 例如 "ZIIZ" -> ["3", "0", "0", "3"]
                try:
                    indices = [pauli_map[char] for char in p_str]
                except KeyError as e:
                    print(f"⚠️ 警告: 发现非法字符 {e} 在字符串 '{p_str}' 中，跳过。")
                    continue

                # 3. 组合并写入
                # 格式: 系数 index1 index2 index3 ...
                indices_str = " ".join(indices)
                f_out.write(f"{coeff_val} {indices_str}\n")

        print(f"✅ 转换成功: {input_file} -> {output_file}")

    except Exception as e:
        print(f"❌ 处理文件 {input_file} 时发生错误: {e}")


# ==========================================
# 批量处理部分
# ==========================================
if __name__ == "__main__":
    # 配置你的文件路径
    # 假设你的文件在 hamil_class 文件夹下
    input_dir = "hamil_class"
    output_dir = "hamil_class/ogm_inputs"  # 输出到子文件夹，保持整洁

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义你要转换的文件类型 (稠密/稀疏) 和 比特数
    types = ["dense", "sparse"]
    qubits = [7]

    print("=== 开始批量转换 Pauli 文   件为 OGM 格式 ===")

    for t in types:
        for n in qubits:
            # 构造输入文件名 (和你之前生成的代码对应)
            # 例如: hamiltonian_sparse_4_pauli.txt
            in_name = f"hamiltonian_{t}_{n}_pauli.txt"
            in_path = os.path.join(input_dir, in_name)

            # 构造输出文件名 (OGM通常只要名字标识)
            # 例如: ogm_hamiltonian_sparse_4.txt
            out_name = f"ogm_hamiltonian_{t}_{n}.txt"
            out_path = os.path.join(output_dir, out_name)

            convert_pauli_to_ogm(in_path, out_path)

    print("\n所有转换任务完成！")
