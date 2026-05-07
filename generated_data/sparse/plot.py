import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================

# 1. 定义要处理的比特数
qubit_counts = [3, 4, 5, 6]

# 2. 数据文件夹
input_dir = 'processed_results'

# 3. 方法映射 (添加了 OverlappedGrouping)
methods_map = {
    'sparse_cmopt_res': 'Proposed (HECCM)',
    'ShadowGrouping': 'Shadow Grouping',
    'OverlappedGrouping': 'OGM',  # <--- 已确认添加
    'Derandomization': 'Derand. Shadows',
    'RandomPaulis': 'Random Paulis'
}

# 4. 样式设置 (添加了 OverlappedGrouping 的配色)
styles = {
    'sparse_cmopt_res': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},  # 红色 实线
    'ShadowGrouping': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},  # 蓝色 虚线
    'OverlappedGrouping': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},  # 绿色 点划线 <--- 新增样式
    'Derandomization': {'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},  # 橙色 点线
    'RandomPaulis': {'color': '#7f7f7f', 'marker': '.', 'linestyle': ':'}  # 灰色 点线
}

# 5. 输出目录
output_dir = 'plots_sparse_random'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ================= 数据读取函数 (带过滤功能) =================

def read_data_file(filepath, is_cmopt=False):
    shots = []
    errors = []

    if not os.path.exists(filepath):
        if is_cmopt:
            print(f"❌ [调试] 未找到文件: {filepath}")
        # 对于其他方法，如果找不到文件可以静默跳过，或者打印提示
        # print(f"  [提示] 未找到文件: {filepath}")
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

                        # === 【关键修改】过滤掉 12 和 45 ===
                        if s == 12 or s == 45:
                            continue
                        # =================================

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

print(f"=== 开始绘图程序 (已包含 OGM) ===")

for n in qubit_counts:
    n_str = str(n)
    print(f"\n正在处理 {n_str} 比特数据...")

    plt.figure(figsize=(10, 7), dpi=120)
    has_data = False

    for method_suffix, method_label in methods_map.items():
        # 拼接文件名: {n}_{suffix}.txt
        # 例如: 6_OverlappedGrouping.txt
        filename = f"{n_str}_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)

        is_cmopt_file = (method_suffix == 'sparse_cmopt_res')

        shots, errors = read_data_file(filepath, is_cmopt=is_cmopt_file)

        if shots and errors:
            has_data = True
            # 获取样式，如果找不到 key 则使用默认黑色
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
        plt.title(f'Random Sparse Hamiltonian Energy Error ($n={n_str}$)', fontsize=16)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(fontsize=10, loc='best', frameon=True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'Sparse_Error_{n_str}qubits.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  -> 图片已生成: {save_path}")
    else:
        plt.close()
        print(f"  -> {n_str} 比特没有有效数据，跳过绘图。")

print(f"\n全部完成！")
