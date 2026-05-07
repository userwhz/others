from __future__ import annotations

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# ================= 1. 全局图表样式设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelpad'] = 5.0
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['legend.frameon'] = False

# ================= 2. 配置区域 =================

qubit_counts = [7]

custom_xticks = [572, 2038, 3845, 7256, 13695, 25848]

x_min = 1700
show_regression = True
scatter_alpha = 0.68
scatter_size_scale = 1.25

methods_map = {
    'RandomPaulis': 'CS',
    'Derandomization': 'Derandomization',
    'ShadowGrouping': 'SG',
    'OverlappedGrouping': 'OGM',
    'cmopt_res': 'CCG-Greedy'
}

styles = {
    'cmopt_res': {
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


def read_data_file(filepath):
    shots = []
    errors = []
    if not os.path.exists(filepath):
        return None, None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None, None
            for line in lines[1:]:
                line_content = line.strip()
                if not line_content:
                    continue
                parts = line_content.split()
                if len(parts) >= 2:
                    try:
                        s = float(parts[0])
                        raw_val = parts[1]
                        e_val = float(raw_val) if '(' not in raw_val else abs(
                            complex(raw_val.replace('(', '').replace(')', '')))
                        if e_val > 0:
                            shots.append(s)
                            errors.append(e_val)
                    except Exception:
                        continue
    except Exception:
        return None, None
    return shots, errors


def resolve_cmopt_res_filename(input_dir: str, explicit: str | None) -> str:
    """各子目录对应 fermionic_n7_L1_res … L4_res；未指定时在 input_dir 下自动匹配唯一文件。"""
    if explicit:
        return explicit
    matches = sorted(glob.glob(os.path.join(input_dir, 'fermionic_n7_L*_res.txt')))
    if len(matches) == 1:
        return os.path.basename(matches[0])
    if len(matches) > 1:
        raise ValueError(f"{input_dir}: 存在多个 fermionic_n7_L*_res.txt: {matches}")
    return 'fermionic_n7_L3_res.txt'


file_map_template = {
    'RandomPaulis': 'fermionic_RandomPaulis.txt',
    'Derandomization': 'fermionic_Derandomization.txt',
    'ShadowGrouping': 'fermionic_ShadowGrouping.txt',
    'OverlappedGrouping': 'fermionic_OverlappedGrouping.txt',
    'cmopt_res': None,  # 由 resolve_cmopt_res_filename 填入
}


def generate_figure(
    input_dir: str,
    output_dir: str,
    *,
    pdf_basename: str = 'Klocal_Error_7qubits.pdf',
    cmopt_res_filename: str | None = None,
) -> str:
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmopt_name = resolve_cmopt_res_filename(input_dir, cmopt_res_filename)
    file_map = dict(file_map_template)
    file_map['cmopt_res'] = cmopt_name

    print(f"=== fermionic 图: input={input_dir}（cmopt: {cmopt_name}）")
    fig, ax = plt.subplots(figsize=(9.0, 6.1), dpi=300)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.14)

    sorted_methods = sorted(methods_map.keys(), key=lambda k: styles.get(k, {}).get('zorder', 0))

    for idx, n in enumerate(qubit_counts):
        n_str = str(n)
        print(f"处理 {n} 比特...")

        has_data = False

        for method_suffix in sorted_methods:
            method_label = methods_map[method_suffix]
            filename = file_map.get(method_suffix, f"{n_str}_{method_suffix}.txt")
            filepath = os.path.join(input_dir, filename)

            shots, errors = read_data_file(filepath)

            if shots and errors:
                has_data = True
                st = styles.get(method_suffix)
                if st is None:
                    print(f"⚠️ 警告: 未找到 {method_suffix} 的样式配置")
                    continue

                ax.scatter(shots, errors,
                           marker=st['marker'],
                           s=(st['markersize'] ** 2) * scatter_size_scale,
                           facecolors=st['color'],
                           edgecolors=st.get('markeredgecolor', st['color']),
                           linewidths=st.get('markeredgewidth', 0),
                           zorder=st['zorder'] + 0.2,
                           alpha=scatter_alpha,
                           label=method_label)

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
            ax.set_xscale('log')
            ax.set_yscale('log', base=2)
            ax.set_xlim(left=x_min)
            ax.set_facecolor('#fcfcfc')

            ax.set_xticks(custom_xticks)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())

            ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0))
            ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=2.0, labelOnlyBase=False))
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=2.0, subs=(np.sqrt(2),)))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            ax.set_xlabel(r'Shots ($N$)', fontsize=16, fontweight='normal')
            ax.set_ylabel(r'RMSE', fontsize=16, fontweight='normal')

            ax.grid(True, axis='both', which='major', linestyle='--', alpha=0.18, linewidth=0.6)
            ax.set_axisbelow(True)

            ax.margins(x=0.03, y=0.04)

            local_handles, local_labels = ax.get_legend_handles_labels()
            ax.legend(local_handles, local_labels,
                      loc='upper right',
                      fontsize=11,
                      frameon=True,
                      facecolor='white',
                      edgecolor='#d9d9d9',
                      framealpha=0.92,
                      fancybox=True,
                      handlelength=2.2,
                      handletextpad=0.8,
                      columnspacing=0.6,
                      ncol=1,
                      title_fontsize=11)
        else:
            ax.text(0.5, 0.5, f'{n} qubits: No Data',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xscale('log')

    pdf_path = os.path.join(output_dir, pdf_basename)
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.12, dpi=300, format='pdf')
    plt.close()
    print(f"  -> 生成: {pdf_path}")
    print(f"✅ fermionic 7-qubit 图完成（{output_dir}）\n")
    return pdf_path


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='fermionic RMSE vs shots（processed_results → PDF）')
    parser.add_argument('--input-dir', default=None, help='含 fermionic_*.txt 的目录')
    parser.add_argument('--output-dir', default=None, help='PDF 输出目录')
    parser.add_argument('--pdf-name', default='Klocal_Error_7qubits.pdf', help='输出 PDF 文件名')
    parser.add_argument(
        '--cmopt-file',
        default=None,
        help='CCG-Greedy 对应文件名（默认在 input 目录下自动匹配 fermionic_n7_L*_res.txt）',
    )
    parser.add_argument(
        '--regenerate-all',
        action='store_true',
        help='重绘子目录 1–4 下 processed_results',
    )
    args = parser.parse_args()

    if args.regenerate_all:
        cases = [
            '1/processed_results',
            '2/processed_results',
            '3/processed_results',
            '4/processed_results',
        ]
        for rel_in in cases:
            inp = os.path.join(script_dir, rel_in)
            parent = os.path.basename(os.path.dirname(inp))
            outp = os.path.join(script_dir, parent, 'plots_klocal_paper_v2')
            if not os.path.isdir(inp):
                print(f"⚠️ 跳过（目录不存在）: {inp}")
                continue
            generate_figure(inp, outp, pdf_basename=args.pdf_name, cmopt_res_filename=args.cmopt_file)
        return

    input_dir = os.path.abspath(args.input_dir or os.path.join(script_dir, 'processed_results'))
    output_dir = os.path.abspath(args.output_dir or os.path.join(script_dir, 'plots_klocal_paper_v2'))
    generate_figure(
        input_dir,
        output_dir,
        pdf_basename=args.pdf_name,
        cmopt_res_filename=args.cmopt_file,
    )


if __name__ == '__main__':
    main()
