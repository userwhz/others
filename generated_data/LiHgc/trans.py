import numpy as np
import os


def trans(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    # 防止文件为空或只有一行的情况导致报错
    if not lines:
        return

    # 第一行通常是真实能量值 (Ground Truth)
    # 处理可能的复数格式 (a+bj) 或纯实数
    try:
        expect_str = lines[0].strip().replace('(', '').replace(')', '')
        expect = complex(expect_str).real
    except ValueError:
        return

    # 提取数据
    # 从第二行开始提取数据
    data = []
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        try:
            row = list(map(float, line.split()))
            data.append(row)
        except ValueError:
            continue

    data = np.array(data)

    # 采样次数列表 (请确保这与你 Julia 模拟时的 Shots 一致)
    # LiH 实验使用的是这组:
    list_N = [12, 45, 160, 572, 2038, 7256, 25848]

    # 计算每一列的误差
    errors = []

    # 增加一个检查，防止数据列数少于 list_N 的长度导致索引越界
    num_columns = data.shape[1] if data.ndim > 1 else 0

    for i, N in enumerate(list_N):
        if i < num_columns:
            values = data[:, i]  # 获取当前采样次数对应的列数据
            # RMSE 计算公式
            sum_err = np.sum((values - expect) ** 2)
            error = np.sqrt(sum_err / len(values))
            errors.append(error)
        else:
            # 如果数据不足，填充 0 或者跳过
            errors.append(0.0)

    # 保存结果为文本文件
    with open(output_file, "w") as f:
        # 写入真实值
        f.write(f"({expect}+0j)\n")
        # 写入每个采样次数对应的误差
        for N, error in zip(list_N, errors):
            f.write(f"{N}   ({error}+0j)\n")


if __name__ == "__main__":
    # 0. 设置输出文件夹路径
    output_dir = "processed_results"

    # 如果文件夹不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 结果将保存在: {output_dir}")

    # 1. 定义 LiH 的键长参数列表
    # 范围1: 0.8 到 2.0，步长 0.2
    range1 = np.arange(0.8, 4.1, 0.2)
    # 范围2: 2.5 和 3.0
    range2 = []

    # 合并并处理精度 (保留1位小数)
    raw_vals = np.concatenate([range1, range2])
    params = sorted(list(set([round(x, 1) for x in raw_vals])))

    print(f"📋 待处理参数列表: {params}")

    # 2. 定义方法列表
    methods = [
        "AdaptivePaulis",
        "AEQuO",
        "Derandomization",
        "RandomPaulis",
        "ShadowGrouping",
        "OverlappedGrouping"
    ]

    # 3. 批量循环处理
    for param in params:
        for method in methods:
            # -----------------------------------------------------------
            # 【关键修改】文件名格式匹配
            # 目标格式: 0.8_LiH_Derandomization_energies.txt
            # -----------------------------------------------------------
            input_file = f"./{param}_LiH_{method}_energies.txt"

            # 输出文件名: 0.8_LiH_Derandomization.txt
            file_name = f"{param}_LiH_{method}.txt"
            output_file = os.path.join(output_dir, file_name)

            # 打印提示方便检查
            print(f"正在转换: param={param}, method={method}")
            # print(f"  输入: {input_file}")

            # 执行转换函数
            try:
                trans(input_file, output_file)
                # print(f"  ✅ 生成: {output_file}")
            except FileNotFoundError:
                print(f"  ⚠️ [跳过] 未找到文件: {input_file}")
            except Exception as e:
                print(f"  ❌ [错误] 处理异常: {e}")

        print("-" * 50)  # 分隔不同param的处理结果

    print("\n🎉 所有转换任务完成！")