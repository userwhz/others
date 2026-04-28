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

# 1. 定义键长列表
bond_lengths = [round(x, 1) for x in np.arange(1.0, 2.8, 0.2)]

# 2. 设置数据文件夹
input_dir = 'processed_results'

# 3. 目标 Shot 数
target_shot = 25848

# 4. 输出目录
output_dir = 'plots_LiH_summary'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 4.1 图例模式
# 可选: 'outside'（图外右侧）或 'inside_box'（图内白底半透明框）
legend_mode = 'inside_box'
legend_inside_alpha = 0.75

# 5. 方法映射
methods_map = {
    'cmopt_res': 'CCG-Greedy ',
    'cmopt_res_surrogateconvex': 'CCG-Convex',
    'OverlappedGrouping': 'OGM',
    'ShadowGrouping': 'SG',
    'Derandomization': 'Derandomization',
    'RandomPaulis': 'CS'
}

# 6. 样式设置
styles = {
    'cmopt_res': {
        'color': '#EE4C2C', 'marker': 'o', 'linestyle': '-',
        'linewidth': 2.4, 'markersize': 8.5, 'zorder': 10, 'markeredgecolor': 'white'
    },
    'cmopt_res_surrogateconvex': {
        'color': '#CC79A7', 'marker': 'h', 'linestyle': '-',
        'linewidth': 2.4, 'markersize': 8.8, 'zorder': 9, 'markeredgecolor': 'white'
    },
    'OverlappedGrouping': {
        'color': '#20854E', 'marker': '^', 'linestyle': '-.',
        'linewidth': 2.0, 'markersize': 8.2, 'zorder': 4
    },
    'ShadowGrouping': {
        'color': '#0073C2', 'marker': 's', 'linestyle': '--',
        'linewidth': 2.0, 'markersize': 7.8, 'zorder': 5
    },
    'Derandomization': {
        'color': '#E18727', 'marker': 'D', 'linestyle': ':',
        'linewidth': 2.0, 'markersize': 7.2, 'zorder': 3
    },
    'RandomPaulis': {
        'color': '#868686', 'marker': 'x', 'linestyle': ':',
        'linewidth': 2.0, 'markersize': 7.2, 'zorder': 1
    }
}


# ================= 3. 数据读取函数 =================

def get_rmse_at_shot(filepath, target_n):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 2: continue
                try:
                    shot = float(parts[0])
                    if abs(shot - target_n) < 100:
                        raw_err = parts[1]
                        clean_err = raw_err.replace('(', '').replace(')', '')
                        rmse = abs(complex(clean_err))
                        return rmse
                except ValueError:
                    continue
        return None
    except Exception as e:
        print(f"  ❌ 读取出错 {filepath}: {e}")
        return None


# ================= 4. 主绘图程序 =================

print(f"=== 开始绘制 LiH Scaling 图 (Shot={target_shot}) ===")

fig, ax = plt.subplots(figsize=(8.2, 5), dpi=300)

for method_suffix, method_label in methods_map.items():
    x_data = []
    y_data = []

    for dist in bond_lengths:
        dist_str = f"{dist:.1f}"
        filename = f"{dist_str}_LiH_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)

        rmse = get_rmse_at_shot(filepath, target_shot)

        if rmse is not None:
            x_data.append(dist)
            y_data.append(rmse)
        else:
            if method_suffix == 'cmopt_res':
                print(f"⚠️ [警告] 缺失数据: {filename}")

    if x_data:
        st = styles.get(method_suffix, {'color': 'black'})
        ax.plot(x_data, y_data,
                label=method_label,
                color=st.get('color'),
                marker=st.get('marker'),
                linestyle=st.get('linestyle'),
                linewidth=st.get('linewidth'),
                markersize=st.get('markersize'),
                markeredgecolor=st.get('markeredgecolor'),
                zorder=st.get('zorder', 1))

# === 图表装饰 (核心修改部分) ===

ax.set_xlabel(r'Bond Length ($\AA$)', fontsize=12, fontweight='bold')
ax.set_ylabel(r'RMSE', fontsize=12, fontweight='bold')

ax.set_yscale('linear')
ax.grid(True, which="major", linestyle='--', alpha=0.4)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))

if legend_mode == 'inside_box':
    # 图内图例（无边框）
    ax.legend(
        loc='best',
        frameon=False
    )
    fig.tight_layout()
else:
    # 图例移到图外右侧，完全避免遮挡数据
    ax.legend(
        fontsize=10,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False
    )
    # 为右侧外置图例预留空间
    fig.tight_layout(rect=[0, 0, 0.8, 1])

pdf_path = os.path.join(output_dir, f'LiH_PES_Error_Shot{target_shot}.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"✅ 图片已生成: {pdf_path}")
plt.close()
