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
        print(f"❌ [跳过] 找不到输入文件 {input_file}")
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

        print(f"✅ 转换成功: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")

    except Exception as e:
        print(f"❌ 处理文件 {input_file} 时发生错误: {e}")


# ==========================================
# 批量处理部分
# ==========================================
if __name__ == "__main__":
    # 1. 定义文件夹路径
    # 假设源文件在当前目录 H2 文件夹下 (或者直接在当前目录，根据实际情况修改)
    # 如果源文件在当前脚本同级目录，请设为 "."
    input_dir = "."
    output_dir = "ogm_inputs"  # 转换后的文件存放位置

    # 自动创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 定义参数范围
    start_val = 0.4
    end_val = 4.3
    step = 0.3

    print(f"=== 开始转换 H2 分子数据 ({start_val} -> {end_val}) ===")

    current_val = start_val

    # 使用 while 循环遍历
    # 注意：加上一个微小的数 (0.001) 是为了防止浮点数精度导致漏掉最后一个 4.3
    while current_val <= end_val + 0.001:
        # 【关键步骤】保留1位小数，解决浮点数精度问题
        # 这样传进去的就是 "0.7" 而不是 "0.7000000001"
        val_str = f"{current_val:.1f}"

        # 3. 构造文件名
        # 输入: H2_0.4_sto-3g_sg.txt
        in_filename = f"H2_{val_str}_sto-3g_sg.txt"
        in_path = os.path.join(input_dir, in_filename)

        # 输出: ogm_H2_0.4.txt (名字你可以自己定义，加上 ogm_ 前缀以示区别)
        out_filename = f"ogm_H2_{val_str}.txt"
        out_path = os.path.join(output_dir, out_filename)

        # 4. 执行转换
        convert_pauli_to_ogm(in_path, out_path)

        # 增加步长
        current_val += step

    print("\n所有 H2 分子文件转换完成！")