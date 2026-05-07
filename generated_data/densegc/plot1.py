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
output_dir = 'plots_dense_random_paper_v2'  # 输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 自定义横坐标刻度
custom_xticks = [160, 572, 2038, 7256, 25848]

methods_map = {
    'RandomPaulis': 'CS',
    'Derandomization': 'Derandomization',
    'ShadowGrouping': 'SG',
    'OverlappedGrouping': 'OGM',
    'dense_cmopt_res': 'CCG-Greedy'
}
styles = {
    # 【修改点 1】: Key 必须改成 'dense_cmopt_res'，否则你的方法会没有颜色报错
    'dense_cmopt_res': {
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


# ================= 3. 数据读取函数 =================
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

print(f"=== 开始生成 Dense 定制化配图 ===")

for n in qubit_counts:
    n_str = str(n)
    print(f"处理 {n} 比特...")

    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300)

    has_data = False
    sorted_methods = sorted(methods_map.keys(), key=lambda k: styles.get(k, {}).get('zorder', 0))

    for method_suffix in sorted_methods:
        method_label = methods_map[method_suffix]

        # 文件名构造，这里会自动寻找如 3_dense_cmopt_res.txt
        filename = f"{n_str}_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)

        # 【修改点 2】: 判断是否为你的方法
        is_cmopt_file = (method_suffix == 'dense_cmopt_res')

        shots, errors = read_data_file(filepath, is_cmopt=is_cmopt_file)

        if shots and errors:
            has_data = True
            st = styles.get(method_suffix)
            # 增加安全检查，防止 styles key 不匹配导致的 crash
            if st is None:
                print(f"⚠️ 警告: 未找到 {method_suffix} 的样式配置，将跳过绘图")
                continue

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

        # 2. 强制设置横坐标刻度
        ax.set_xticks(custom_xticks)

        # 3. 格式化横坐标数字
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        # 旋转横坐标标签
        plt.xticks(rotation=30)

        # 移除 Log 小刻度
        ax.minorticks_off()

        # 4. 纵坐标标签设置
        ax.set_ylabel(r'RMSE', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Shots ($N$)', fontsize=14, fontweight='bold')

        # 5. 网格设置
        ax.grid(True, axis='both', which="major", linestyle='--', alpha=0.4)

        # 图例
        ax.legend(
            loc='best',
            frameon=True,
            facecolor='white',
            edgecolor='0.6',
            framealpha=0.75,
            fancybox=True
        )

        plt.tight_layout()

        # 【修改点 3】: 文件名改为 Dense
        pdf_path = os.path.join(output_dir, f'Dense_Error_{n}qubits.pdf')
        plt.savefig(pdf_path)
        plt.close()
        print(f"  -> 生成: {pdf_path}")
    else:
        plt.close()
        print(f"  -> {n} 比特无数据跳过")

print(f"\n全部绘图完成！")