import numpy as np
import os


def trans(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件未找到: {input_file}")

    with open(input_file, "r") as f:
        lines = f.readlines()

    # 第一行为真实值 (Ground Truth)
    # 注意：有时候 header 可能带有 # 号，或者纯数字，这里做个简单清洗
    header_str = lines[0].strip().replace('#', '')
    expect = complex(header_str).real

    # 提取数据 (从第二行开始)
    # data 的形状是 (N_runs, N_steps)
    data = [list(map(float, line.strip().split())) for line in lines[1:]]
    data = np.array(data)

    # -----------------------------------------------------------
    # 【关键修改】采样次数列表
    # 必须与你生成数据时(Julia/Python)设置的 shots 列表完全一致
    # 之前商定的新列表是: [160, 320, 640, 1280, 2560, 5120, 10240]
    # 如果你跑的是旧数据，请改回原来的 [12, 45, ...]
    # -----------------------------------------------------------
    list_N = [572, 2038, 3845, 7256, 13695, 25848]

    # 安全检查：防止数据列数和 Shot 数不匹配
    if data.shape[1] != len(list_N):
        print(f"⚠️ 警告: 数据列数({data.shape[1]}) 与 list_N长度({len(list_N)}) 不匹配！")

    # 计算每一列的 RMSE
    errors = []
    for i, N in enumerate(list_N):
        if i < data.shape[1]:
            values = data[:, i]  # 获取第 i 列 (对应第 N 个 shot 设置)

            # RMSE 计算公式: sqrt( mean( (est - true)^2 ) )
            sum_err = np.sum((values - expect) ** 2)
            error = np.sqrt(sum_err / len(values))
            errors.append(error)
        else:
            errors.append(0.0)  # 填充占位

    # 保存结果
    with open(output_file, "w") as f:
        # 写入真实值 (格式: (0.123+0j))
        f.write(f"({expect}+0j)\n")
        # 写入误差
        for N, error in zip(list_N, errors):
            f.write(f"{N}   ({error}+0j)\n")


if __name__ == "__main__":
    # 当前目录下的 klocal 输出文件
    input_dir = "."
    output_dir = "processed_results"  # 结果保存位置

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 需要处理的当前 klocal 文件
    methods = [
        # "AdaptivePaulis",
        # "AEQuO",
        "Derandomization",
        "RandomPaulis",
        "ShadowGrouping",
        "OverlappedGrouping",
    ]

    # 逐个处理当前 klocal 目录下的文件
    for method in methods:
        input_filename = f"klocal_{method}_energies.txt"
        input_path = os.path.join(input_dir, input_filename)

        output_filename = f"klocal_{method}.txt"
        output_path = os.path.join(output_dir, output_filename)

        print(f"正在处理: method={method}")

        try:
            trans(input_path, output_path)
            print(f"  ✅ 成功生成: {output_path}")
        except FileNotFoundError:
            print(f"  ❌ 未找到文件: {input_path}")
        except Exception as e:
            print(f"  ⚠️ 处理出错: {e}")

    print("-" * 50)
