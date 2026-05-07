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
bond_lengths = [round(x, 1) for x in np.arange(0.4, 2.2, 0.3)]

# 2. 设置数据文件夹
input_dir = 'processed_results'

# 3. 目标 Shot 数
target_shot = 25848

# 4. 输出目录
output_dir = 'plots_H2_summary'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 5. 方法映射
methods_map = {
    'H2_cmopt_res': 'CCG-Greedy', 
    'H2_cmopt_res_surrogateconvex_hf_prior': 'CCG-Convex',
    'OverlappedGrouping': 'OGM',
    'ShadowGrouping': 'SG',
    'Derandomization': 'Derandomization',
    'RandomPaulis': 'CS'
}
# 6. 样式设置
styles = {
    'H2_cmopt_res': {
        'color': '#EE4C2C', 'marker': 'o', 'linestyle': '-',
        'linewidth': 2.0, 'markersize': 7, 'zorder': 10, 'markeredgecolor': 'white'
    },
    'H2_cmopt_res_surrogateconvex_hf_prior': {
        'color': '#C44E52', 'marker': 'P', 'linestyle': '-',
        'linewidth': 2.0, 'markersize': 6.5, 'zorder': 9, 'markeredgecolor': 'white'
    },
    'OverlappedGrouping': {
        'color': '#20854E', 'marker': '^', 'linestyle': '-.',
        'linewidth': 1.5, 'markersize': 7, 'zorder': 4
    },
    'ShadowGrouping': {
        'color': '#0073C2', 'marker': 's', 'linestyle': '--',
        'linewidth': 1.5, 'markersize': 6, 'zorder': 5
    },
    'Derandomization': {
        'color': '#E18727', 'marker': 'D', 'linestyle': ':',
        'linewidth': 1.5, 'markersize': 5, 'zorder': 3
    },
    'RandomPaulis': {
        'color': '#868686', 'marker': 'x', 'linestyle': ':',
        'linewidth': 1.5, 'markersize': 5, 'zorder': 1
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
                    if abs(shot - target_n) < 10:
                        raw_err = parts[1]
                        clean_err = raw_err.replace('(', '').replace(')', '')
                        rmse = abs(complex(clean_err))
                        return rmse
                except ValueError:
                    continue
        return None
    except Exception as e:
        return None

# ================= 4. 主绘图程序 =================

print(f"=== 开始绘制 H2 Scaling 图 (Shot={target_shot}) ===")

# 【修改点1】：使用 layout='constrained' 实现自适应布局
# 尺寸设为 (8, 5) 这是一个比较标准且稍微宽一点的比例，适合大多数论文排版
fig, ax = plt.subplots(figsize=(8, 5), dpi=300, layout='constrained')

for method_suffix, method_label in methods_map.items():
    x_data = []
    y_data = []

    for dist in bond_lengths:
        dist_str = f"{dist:.1f}"
        filename = f"{dist_str}_{method_suffix}.txt"
        filepath = os.path.join(input_dir, filename)

        if not os.path.exists(filepath):
            continue

        rmse = get_rmse_at_shot(filepath, target_shot)

        if rmse is not None:
            x_data.append(dist)
            y_data.append(rmse)

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

# === 图表装饰 ===

ax.set_xlabel(r'Bond Length ($\AA$)', fontsize=14, fontweight='bold')
ax.set_ylabel(r'RMSE (N = 25848)', fontsize=14, fontweight='bold')

# 化学精度线
ax.axhline(y=0.0016, color='#444444', linestyle='--', linewidth=1.2, alpha=0.8, zorder=0,
           label=r'Chemical Accuracy ')

ax.set_yscale('log')
ax.grid(True, which="major", linestyle='--', alpha=0.4)

# 【修改点2】：使用 loc='best' 让程序自动寻找不挡线的空白位置
# 并开启半透明背景 (framealpha)，万一真的有一点点重叠也能看清后面的线
ax.legend(fontsize=10.5, loc='best', frameon=True, framealpha=0.8, edgecolor='none')

# 【修改点3】：稍微增加 Y 轴上方的留白，给图例腾出更多“最佳位置”的选择空间
ax.margins(y=0.1)

# 注意：使用了 layout='constrained' 后，不需要再调用 plt.tight_layout()

pdf_path = os.path.join(output_dir, f'H2_Bond_Scaling_RMSE_at_{target_shot}.pdf')
plt.savefig(pdf_path)
print(f"✅ 图片已生成: {pdf_path}")
plt.close()