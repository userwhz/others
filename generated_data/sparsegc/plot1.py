import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

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
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['legend.frameon'] = False

# ================= 2. 配置区域 =================

qubit_counts = [3, 4, 5, 6]
input_dir = 'processed_results'
output_dir = 'plots_sparse_random_paper_v2'  # 新目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 【核心修改】这是你指定的横坐标点
custom_xticks = [160, 572, 2038, 7256, 25848]
show_regression = True
scatter_alpha = 0.68
scatter_size_scale = 1.25

methods_map = {
    'RandomPaulis': 'CS',
    'Derandomization': 'Derandomization',
    'ShadowGrouping': 'SG',
    'OverlappedGrouping': 'OGM',
    'sparse_cmopt_res': 'CCG-Greedy'
}

styles = {
    'sparse_cmopt_res': {
        'color': '#D62728', 'marker': 'o', 'linestyle': '-',
        'linewidth': 2.4, 'markersize': 10.5, 'zorder': 10, 'markeredgecolor': 'white', 'markeredgewidth': 1.0
    },
    'ShadowGrouping': {
        'color': '#1f77b4', 'marker': 's', 'linestyle': '--',
        'linewidth': 2.0, 'markersize': 9.5, 'zorder': 5, 'alpha': 1.0
    },
    'OverlappedGrouping': {
        'color': '#2ca02c', 'marker': '^', 'linestyle': '-.',
        'linewidth': 2.0, 'markersize': 10, 'zorder': 4, 'alpha': 1.0
    },
    'Derandomization': {
        'color': '#ff7f0e', 'marker': 'D', 'linestyle': (0, (5, 3)),
        'linewidth': 2.0, 'markersize': 8.5, 'zorder': 3, 'alpha': 1.0
    },
    'RandomPaulis': {
        'color': '#7f7f7f', 'marker': '*', 'linestyle': ':',
        'linewidth': 1.8, 'markersize': 14, 'zorder': 1, 'alpha': 1.0
    }
}


# ================= 3. 数据读取函数 (保持不变) =================
def read_data_file(filepath):
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


# ================= 4. 创建2x2子图 =================

print(f"=== 开始生成 Sparse 2x2 组合图 ===")

fig = plt.figure(figsize=(14, 10), dpi=300)
outer_gs = GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.22,
                    left=0.12, right=0.95, top=0.98, bottom=0.12)

axes = []
for i in range(2):
    row_axes = []
    for j in range(2):
        ax = fig.add_subplot(outer_gs[i, j])
        row_axes.append(ax)
    axes.append(row_axes)

sorted_methods = sorted(methods_map.keys(), key=lambda k: styles.get(k, {}).get('zorder', 0))

for idx, n in enumerate(qubit_counts):
    n_str = str(n)
    row = idx // 2
    col = idx % 2
    ax = axes[row][col]

    print(f"处理 {n} 比特...")

    has_data = False
    for method_suffix in sorted_methods:
        method_label = methods_map[method_suffix]
        filename = f"{n_str}_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)

        shots, errors = read_data_file(filepath)

        if shots and errors:
            has_data = True
            st = styles.get(method_suffix)
            if st is None:
                print(f"⚠️ 警告: 未找到 {method_suffix} 的样式配置")
                continue

            # 散点层：每种方法只画散点，不画原始折线
            ax.scatter(shots, errors,
                       marker=st['marker'],
                       s=(st['markersize'] ** 2) * scatter_size_scale,
                       facecolors=st['color'],
                       edgecolors=st.get('markeredgecolor', st['color']),
                       linewidths=st.get('markeredgewidth', 0),
                       zorder=st['zorder'] + 0.2,
                       alpha=scatter_alpha,
                       label=method_label)

            # 在log-log空间做线性回归并叠加趋势线
            if show_regression and len(shots) >= 2:
                x_arr = np.array(shots, dtype=float)
                y_arr = np.array(errors, dtype=float)
                valid = (x_arr > 0) & (y_arr > 0)
                if np.count_nonzero(valid) >= 2:
                    lx = np.log10(x_arr[valid])
                    ly = np.log10(y_arr[valid])
                    slope, intercept = np.polyfit(lx, ly, 1)
                    x_fit = np.linspace(lx.min(), lx.max(), 100)
                    y_fit = 10 ** (slope * x_fit + intercept)
                    ax.plot(10 ** x_fit, y_fit,
                            color=st['color'],
                            linestyle='-',
                            linewidth=2.0,
                            alpha=1.0,
                            zorder=max(st['zorder'] - 1, 0),
                            label='_nolegend_')
                    print(f"  [回归] {n} qubits | {method_label}: log10(y) = {slope:.4f}*log10(x) + {intercept:.4f}")

    if has_data:
        # 坐标轴设置：x对数，y改为log2
        ax.set_xscale('log')
        ax.set_yscale('log', base=2)

        # 横坐标恢复为之前固定刻度
        ax.set_xticks(custom_xticks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        # 纵轴使用log2刻度，主刻度显示2^k，次刻度用sqrt(2)细分
        ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=2.0, labelOnlyBase=False))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=2.0, subs=(np.sqrt(2),)))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # 标签
        ax.set_xlabel(r'Shots ($N$)', fontsize=16, fontweight='normal')
        ax.set_ylabel(r'RMSE', fontsize=16, fontweight='normal')

        # 网格
        ax.grid(True, axis='both', which='major', linestyle='-', alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)

        # log坐标下保留少量留白，避免最上方曲线贴边
        ax.margins(y=0.03)

        # 每个子图右上角都有方法图例
        local_handles, local_labels = ax.get_legend_handles_labels()
        ax.legend(local_handles, local_labels,
                  loc='upper right',
                  fontsize=11,
                  frameon=False,
                  handlelength=2.2,
                  handletextpad=0.8,
                  columnspacing=0.6,
                  ncol=1,
                  title_fontsize=11)
    else:
        ax.text(0.5, 0.5, f'{n} qubits: No Data',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xscale('log')

# 调整布局
plt.subplots_adjust(bottom=0.08, top=0.97)

# (a)(b)(c)(d)：放在每个子图“纵轴最上方并稍微偏左”
for idx, ax in enumerate([axes[0][0], axes[0][1], axes[1][0], axes[1][1]]):
    bbox = ax.get_position()  # figure 坐标系
    subplot_label = chr(97 + idx)  # a, b, c, d
    fig.text(
        bbox.x0 - 0.015,
        bbox.y1 + 0.002,
        f'({subplot_label})',
        fontsize=16,
        fontweight='bold',
        family='monospace',
        ha='right',
        va='bottom',
        transform=fig.transFigure
    )

pdf_path = os.path.join(output_dir, 'Sparse_Error_2x2_Combined.pdf')
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.12, dpi=300, format='pdf')
plt.close()
print(f"  -> 生成: {pdf_path}")

print(f"\n✅ 2x2 组合图生成完成！")
print(f"   • (a)(b)(c)(d) 在每个子图左上外侧")
print(f"   • 每个子图右上角都有 Methods 图例")
