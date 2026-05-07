import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================

# 1. 定义 LiH 的键长列表 (与生成数据时一致)
bond_lengths = [0.8, 1.0, 1.2, 1.4,  1.6, 1.8, 2.0,  2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]

# 2. 数据文件夹
input_dir = 'processed_results'

# 3. 方法映射
# 注意：文件名格式为 {dist}_LiH_{method}.txt
# 所以这里的 key 需要包含 "LiH_" 前缀
methods_map = {
    'LiH_cmopt_res': 'Proposed (HECCM)',
    'LiH_ShadowGrouping': 'Shadow Grouping',
    'LiH_OverlappedGrouping': 'OGM',
    'LiH_AEQuO': 'AEQuO',
    'LiH_Derandomization': 'Derand. Shadows',
    'LiH_AdaptivePaulis': 'Adaptive Paulis',
    'LiH_RandomPaulis': 'Random Paulis'
}

# 4. 样式设置 (Key 必须与 methods_map 对应)
styles = {
    'LiH_cmopt_res': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},  # 红色
    'LiH_ShadowGrouping': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},  # 蓝色
    'LiH_OverlappedGrouping': {'color': '#8c564b', 'marker': 'v', 'linestyle': '-.'},  # 棕色
    'LiH_AEQuO': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--'},  # 绿色
    'LiH_Derandomization': {'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},  # 橙色
    'LiH_AdaptivePaulis': {'color': '#9467bd', 'marker': 'x', 'linestyle': '-.'},  # 紫色
    'LiH_RandomPaulis': {'color': '#7f7f7f', 'marker': '.', 'linestyle': ':'}  # 灰色
}

# 5. 输出目录
output_dir = 'plots_LiH'
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
                    print(f"❌ [调试] 文件行数过少: {filepath}")
                return None, None

            if is_cmopt:
                print(f"✅ [调试] 正在读取: {filepath}")

            for i, line in enumerate(lines[1:], start=2):
                line_content = line.strip()
                if not line_content:
                    continue

                parts = line_content.split()

                try:
                    s = float(parts[0])  # Shot

                    # === 过滤掉 Shot 12 和 45 ===
                    # if s == 12 or s == 45:
                    #     continue
                    # ==========================

                    raw_val = parts[1]  # Error
                    e_val = 0.0

                    try:
                        # 策略A: 纯浮点数
                        e_val = float(raw_val)
                    except ValueError:
                        # 策略B: 复数字符串 (例如 (0.001+0j))
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

print(f"=== 开始绘图程序 (LiH Mode) ===")

for dist in bond_lengths:
    dist_str = f"{dist:.1f}"
    print(f"\n正在处理键长 {dist_str} Å...")

    plt.figure(figsize=(10, 7), dpi=120)
    has_data = False

    for method_suffix, method_label in methods_map.items():
        # 拼接文件名: 0.8_LiH_Derandomization.txt
        filename = f"{dist_str}_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)

        # 标记是否为新方法 (用于调试输出)
        is_cmopt_file = (method_suffix == 'LiH_cmopt_res')

        shots, errors = read_data_file(filepath, is_cmopt=is_cmopt_file)

        if shots and errors:
            has_data = True
            # 获取样式，默认为黑色实线
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
            print(f"⚠️ [警告] 键长 {dist_str} 的 cmopt 数据为空或未找到！")

    if has_data:
        # 添加化学精度线
        plt.axhline(y=0.0016, color='black', linestyle='--', linewidth=1.2, label='Chemical Accuracy')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel('Number of Shots ($N$)', fontsize=14)
        plt.ylabel('Energy Estimation Error (Hartree)', fontsize=14)

        # 标题修改为 LiH
        plt.title(f'LiH Ground State Energy Error (Bond Length = {dist_str} Å)', fontsize=16)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(fontsize=10, loc='best', frameon=True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'LiH_Error_{dist_str}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  -> 图片已生成: {save_path}")
    else:
        plt.close()
        print(f"  -> {dist_str} Å 没有有效数据，跳过绘图。")

print(f"\n全部完成！请检查 {output_dir} 文件夹。")