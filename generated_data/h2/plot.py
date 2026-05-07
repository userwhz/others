import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
bond_lengths = np.arange(0.4, 4.35, 0.3)

# 1. 设置数据所在的文件夹
# 如果文件在当前目录，请改为 input_dir = '.'
input_dir = 'processed_results'

# 2. 方法映射 (添加了 OverlappedGrouping)
methods_map = {
    'H2_cmopt_res': 'Proposed (HECCM)',
    'ShadowGrouping': 'Shadow Grouping',
    'OverlappedGrouping': 'OGM',  # <--- 新增 OGM
    'AEQuO': 'AEQuO',
    'Derandomization': 'Derand. Shadows',
    'AdaptivePaulis': 'Adaptive Paulis',
    'RandomPaulis': 'Random Paulis'
}

# 3. 样式设置 (添加了 OGM 的配色)
styles = {
    'H2_cmopt_res': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},  # 红色
    'ShadowGrouping': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},  # 蓝色
    'OverlappedGrouping': {'color': '#8c564b', 'marker': 'v', 'linestyle': '-.'},  # 棕色 (新增)
    'AEQuO': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--'},  # 绿色
    'Derandomization': {'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},  # 橙色
    'AdaptivePaulis': {'color': '#9467bd', 'marker': 'x', 'linestyle': '-.'},  # 紫色
    'RandomPaulis': {'color': '#7f7f7f', 'marker': '.', 'linestyle': ':'}  # 灰色
}

output_dir = 'plots_debug'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ================= 增强版数据读取函数 =================

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
                if is_cmopt:
                    print(f"❌ [调试] 文件行数过少 ({len(lines)}行): {filepath}")
                return None, None

            if is_cmopt:
                print(f"✅ [调试] 正在读取: {filepath}")

            for i, line in enumerate(lines[1:], start=2):
                line_content = line.strip()
                if not line_content:
                    continue

                parts = line_content.split()

                if len(parts) >= 2:
                    try:
                        s = float(parts[0])  # Shot

                        # 如果需要过滤 12 和 45，可以在这里加：
                        # if s == 12 or s == 45: continue

                        raw_val = parts[1]  # Error
                        e_val = 0.0

                        try:
                            # 策略A: 纯浮点数 (cmopt)
                            e_val = float(raw_val)
                        except ValueError:
                            # 策略B: 复数
                            clean_val = raw_val.replace('(', '').replace(')', '')
                            e_val = abs(complex(clean_val))

                        if e_val > 0:
                            shots.append(s)
                            errors.append(e_val)
                        else:
                            if is_cmopt:
                                print(f"⚠️ [调试] 第{i}行 Error值 <= 0: {e_val}")

                    except Exception as loop_e:
                        if is_cmopt:
                            print(f"❌ [调试] 第{i}行解析失败: {loop_e}")
                        continue
    except Exception as e:
        print(f"❌ 读取文件严重错误 {filepath}: {e}")
        return None, None

    return shots, errors


# ================= 主绘图循环 =================

print(f"=== 开始绘图程序 (从 {input_dir} 读取) ===")

for dist in bond_lengths:
    dist_str = f"{dist:.1f}"

    plt.figure(figsize=(10, 7), dpi=120)
    has_data = False

    for method_suffix, method_label in methods_map.items():
        # 拼接文件名: 0.4_OverlappedGrouping.txt
        filename = f"{dist_str}_{method_suffix}.txt"

        # 拼接完整路径
        filepath = os.path.join(input_dir, filename)

        # 标记当前是否是在读取 cmopt 文件
        is_cmopt_file = (method_suffix == 'H2_cmopt_res')

        shots, errors = read_data_file(filepath, is_cmopt=is_cmopt_file)

        if shots and errors:
            has_data = True
            st = styles.get(method_suffix, {'color': 'black'})

            plt.plot(shots, errors,
                     label=method_label,
                     color=st.get('color'),
                     marker=st.get('marker'),
                     linestyle=st.get('linestyle'),
                     markersize=6,
                     linewidth=1.5,
                     alpha=0.9)
        elif is_cmopt_file:
            print(f"⚠️ [警告] 键长 {dist_str} 的 cmopt 数据为空或未找到！")

    if has_data:
        plt.axhline(y=0.0016, color='black', linestyle='--', linewidth=1.2, label='Chemical Accuracy')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Shots ($N$)', fontsize=14)
        plt.ylabel('Energy Estimation Error (Hartree)', fontsize=14)
        plt.title(f'$H_2$ Ground State Energy Error (Bond Length = {dist_str} Å)', fontsize=16)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(fontsize=10, loc='best', frameon=True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'H2_Error_{dist_str}.png')
        plt.savefig(save_path)
        plt.close()
    else:
        plt.close()

print(f"\n绘图结束，请检查 plots_debug 文件夹。")
