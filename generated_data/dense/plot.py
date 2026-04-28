import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================

# 1. 定义要处理的比特数
qubit_counts = [3, 4, 5, 6]

# 2. 数据文件夹
input_dir = 'processed_results'

# 3. 方法映射 (确保 dense_cmopt_res 正确，同时加上 OGM)
methods_map = {
    'dense_cmopt_res': 'Proposed (HECCM)',  # 你的新方法 (稠密)
    'ShadowGrouping': 'Shadow Grouping',
    'OverlappedGrouping': 'OGM',  # 补充 OGM
    'Derandomization': 'Derand. Shadows',
    'RandomPaulis': 'Random Paulis'
}

# 4. 样式设置
styles = {
    'dense_cmopt_res': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},  # 红色 实线
    'ShadowGrouping': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},  # 蓝色 虚线
    'OverlappedGrouping': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},  # 绿色 点划线
    'Derandomization': {'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},  # 橙色 点线
    'RandomPaulis': {'color': '#7f7f7f', 'marker': '.', 'linestyle': ':'}  # 灰色 点线
}

# 5. 【修改】输出目录改为 dense
output_dir = 'plots_dense_random'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ================= 数据读取函数 (带过滤功能) =================

def read_data_file(filepath, is_cmopt=False):
    shots = []
    errors = []

    if not os.path.exists(filepath):
        if is_cmopt:
            print(f"❌ [调试] 未找到文件: {filepath}")
        return None, None

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

            if len(lines) < 2:
                return None, None

            if is_cmopt:
                print(f"✅ [调试] 正在读取: {filepath}")

            for line in lines[1:]:
                line_content = line.strip()
                if not line_content:
                    continue

                parts = line_content.split()

                if len(parts) >= 2:
                    try:
                        s = float(parts[0])  # Shot

                        # === 过滤掉 12 和 45 ===
                        if s == 12 or s == 45:
                            continue
                        # ======================

                        raw_val = parts[1]  # Error
                        e_val = 0.0

                        try:
                            # 策略A: 纯浮点数
                            e_val = float(raw_val)
                        except ValueError:
                            # 策略B: 复数字符串
                            clean_val = raw_val.replace('(', '').replace(')', '')
                            e_val = abs(complex(clean_val))

                        if e_val > 0:
                            shots.append(s)
                            errors.append(e_val)
                    except Exception:
                        continue
    except Exception as e:
        print(f"❌ 读取错误 {filepath}: {e}")
        return None, None

    return shots, errors


# ================= 主绘图循环 =================

print(f"=== 开始绘图程序 (Dense Mode) ===")

for n in qubit_counts:
    n_str = str(n)
    print(f"\n正在处理 {n_str} 比特数据...")

    plt.figure(figsize=(10, 7), dpi=120)
    has_data = False

    for method_suffix, method_label in methods_map.items():
        # 拼接文件名
        filename = f"{n_str}_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)

        # 判断是否是你的新方法
        is_cmopt_file = (method_suffix == 'dense_cmopt_res')

        shots, errors = read_data_file(filepath, is_cmopt=is_cmopt_file)

        if shots and errors:
            has_data = True
            st = styles.get(method_suffix, {'color': 'black', 'marker': 'x', 'linestyle': '-'})

            plt.plot(shots, errors,
                     label=method_label,
                     color=st.get('color'),
                     marker=st.get('marker'),
                     linestyle=st.get('linestyle'),
                     markersize=6,
                     linewidth=1.5,
                     alpha=0.9)
        elif is_cmopt_file:
            print(f"⚠️ [警告] 在 {input_dir} 中未找到 {filename}")

    if has_data:
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel('Number of Shots ($N$)', fontsize=14)
        plt.ylabel('Estimation Error (RMSE)', fontsize=14)

        # 【修改】标题改为 Dense
        plt.title(f'Random Dense Hamiltonian Energy Error ($n={n_str}$)', fontsize=16)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(fontsize=10, loc='best', frameon=True)
        plt.tight_layout()

        # 【修改】保存文件名改为 Dense
        save_path = os.path.join(output_dir, f'Dense_Error_{n_str}qubits.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  -> 图片已生成: {save_path}")
    else:
        plt.close()
        print(f"  -> {n_str} 比特没有有效数据，跳过绘图。")

print(f"\n全部完成！")