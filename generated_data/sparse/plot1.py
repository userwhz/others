import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# ================= 1. 全局图表样式设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.frameon'] = False

# ================= 2. 配置区域 =================

qubit_counts = [3, 4, 5, 6]
input_dir = 'processed_results'
output_dir = 'plots_sparse_random_paper_v2'  # 新目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 【核心修改】这是你指定的横坐标点
custom_xticks = [160, 572, 2038, 7256, 25848]

methods_map = {
    'RandomPaulis': 'CS',
    'Derandomization': 'Derandomization',
    'ShadowGrouping': 'SG',
    'OverlappedGrouping': 'OGM',
    'sparse_cmopt_res': 'CCG'
}

styles = {
    'sparse_cmopt_res': {
        'color': '#EE4C2C', 'marker': 'o', 'linestyle': '-',
        'linewidth': 2.0, 'markersize': 7, 'zorder': 10, 'markeredgecolor': 'white', 'markeredgewidth': 0.8
    },
    'ShadowGrouping': {
        'color': '#0073C2', 'marker': 's', 'linestyle': '--',
        'linewidth': 1.8, 'markersize': 6, 'zorder': 5, 'alpha': 0.9
    },
    'OverlappedGrouping': {
        'color': '#20854E', 'marker': '^', 'linestyle': '-.',
        'linewidth': 1.8, 'markersize': 7, 'zorder': 4, 'alpha': 0.9
    },
    'Derandomization': {
        'color': '#E18727', 'marker': 'D', 'linestyle': (0, (3, 1, 1, 1)),
        'linewidth': 1.8, 'markersize': 5, 'zorder': 3, 'alpha': 0.9
    },
    'RandomPaulis': {
        'color': '#868686', 'marker': '.', 'linestyle': ':',
        'linewidth': 1.5, 'markersize': 4, 'zorder': 1, 'alpha': 0.7
    }
}


# ================= 3. 数据读取函数 (保持不变) =================
def read_data_file(filepath, is_cmopt=False):
    shots = []
    errors = []
    if not os.path.exists(filepath): return None, None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2: return None, None
            for line in lines[1:]:
                line_content = line.strip()
                if not line_content: continue
                parts = line_content.split()
                if len(parts) >= 2:
                    try:
                        s = float(parts[0])
                        if s == 12 or s == 45: continue
                        raw_val = parts[1]
                        e_val = float(raw_val) if '(' not in raw_val else abs(
                            complex(raw_val.replace('(', '').replace(')', '')))
                        if e_val > 0:
                            shots.append(s)
                            errors.append(e_val)
                    except:
                        continue
    except:
        return None, None
    return shots, errors


# ================= 4. 主绘图循环 =================

print(f"=== 开始生成定制化配图 ===")

for n in qubit_counts:
    n_str = str(n)
    print(f"处理 {n} 比特...")

    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300)  # 稍微调高一点高度，给旋转的标签留空间

    has_data = False
    sorted_methods = sorted(methods_map.keys(), key=lambda k: styles.get(k, {}).get('zorder', 0))

    for method_suffix in sorted_methods:
        method_label = methods_map[method_suffix]
        filename = f"{n_str}_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)
        is_cmopt_file = (method_suffix == 'sparse_cmopt_res')

        shots, errors = read_data_file(filepath, is_cmopt=is_cmopt_file)

        if shots and errors:
            has_data = True
            st = styles.get(method_suffix)
            ax.plot(shots, errors,
                    label=method_label,
                    color=st['color'],
                    marker=st['marker'],
                    linestyle=st['linestyle'],
                    linewidth=st['linewidth'],
                    markersize=st['markersize'],
                    markeredgecolor=st.get('markeredgecolor'),
                    markeredgewidth=st.get('markeredgewidth', 0),
                    zorder=st['zorder'],
                    alpha=st.get('alpha', 1.0))

    if has_data:
        # 1. 设置 Log 坐标轴
        ax.set_xscale('log')
        ax.set_yscale('log')

        # 2. 【核心修改】强制设置横坐标刻度为你指定的点
        ax.set_xticks(custom_xticks)

        # 3. 将数字格式化为普通整数（不使用科学计数法），并旋转防止重叠
        #    FuncFormatter 确保显示的是原始数字 160, 572...
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        # 旋转横坐标标签 30度，防止 25848 挤在一起
        plt.xticks(rotation=30)

        # 移除 Log 坐标轴默认的小刻度（因为我们指定了具体的点，默认的小刻度会干扰视线）
        ax.minorticks_off()

        # 4. 纵坐标标签设置 (解释 epsilon)
        # 这里的 r'...' 是 LaTeX 格式
        ax.set_ylabel(r'RMSE', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Shots ($N$)', fontsize=14, fontweight='bold')

        # 5. 网格设置
        # axis='x' 仅显示竖线，这样竖线会精准穿过你的数据点，很好看
        # axis='both' 显示网格
        ax.grid(True, axis='both', which="major", linestyle='--', alpha=0.4)

        # 图例
        ax.legend(fontsize=10.5, loc='best', frameon=False)

        plt.tight_layout()

        pdf_path = os.path.join(output_dir, f'Sparse_Error_{n}qubits.pdf')
        plt.savefig(pdf_path)
        plt.close()
        print(f"  -> 生成: {pdf_path}")
    else:
        plt.close()
        print(f"  -> {n} 比特无数据跳过")

print(f"\n全部绘图完成！")
