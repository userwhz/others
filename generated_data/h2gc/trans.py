import numpy as np
import os


def trans(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    # 防止文件为空或只有一行的情况导致报错
    if not lines:
        return

    expect = complex(lines[0].strip())  # 第一行为真实值，转换为复数
    expect = expect.real  # 取实部

    # 提取数据
    # 从第二行开始提取数据
    data = [list(map(float, line.strip().split())) for line in lines[1:]]
    data = np.array(data)  # 转换为 NumPy 数组

    # 采样次数列表
    list_N = [12, 45, 160, 572, 2038, 7256, 25848]

    # 计算每一列的误差
    errors = []

    # 增加一个检查，防止数据列数少于 list_N 的长度导致索引越界
    num_columns = data.shape[1] if data.ndim > 1 else 0

    for i, N in enumerate(list_N):
        if i < num_columns:
            values = data[:, i]  # 获取当前采样次数对应的列数据
            # print(values) # 可以注释掉以减少控制台输出
            sum_err = np.sum((values - expect) ** 2)  # 计算误差平方和
            error = np.sqrt(sum_err / len(values))  # 计算均方根误差 (RMSE)
            errors.append(error)
        else:
            # 如果数据不足，填充 0 或者跳过，防止报错
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
        print(f"创建文件夹: {output_dir}")

    # 1. 定义那个变化的参数 (占位符)
    start_value = 0.4
    end_value = 4.3
    step = 0.3

    # 计算循环次数
    num_steps = int((end_value - start_value) / step) + 1

    for i in range(num_steps):
        param = start_value + i * step
        # 由于浮点数精度问题，可以四舍五入到1位小数
        param = round(param, 1)

        # 2. 定义方法列表
        methods = [
            # "AdaptivePaulis",
            # "AEQuO",
            # "Derandomization",
            # "RandomPaulis",
            "ShadowGrouping",
            "OverlappedGrouping"
        ]

        # 3. 批量循环处理
        for method in methods:
            # 自动拼接输入路径: e.g. ../0.4_sto3g_AdaptivePaulis_energies.txt
            # 假设输入文件在上一级目录
            input_file = f"./{param}_sto3g_{method}_energies.txt"

            # 自动拼接输出路径: e.g. processed_results/0.4_AdaptivePaulis.txt
            file_name = f"{param}_{method}.txt"
            output_file = os.path.join(output_dir, file_name)

            # 打印提示方便检查
            print(f"正在转换: param={param}, method={method}")
            print(f"  输入: {input_file}")
            print(f"  输出: {output_file}")

            # 执行转换函数
            try:
                trans(input_file, output_file)
            except FileNotFoundError:
                print(f"  ⚠️ 跳过: 找不到文件 {input_file}")
            except Exception as e:
                print(f"  ❌ 错误: 处理 {input_file} 时发生异常: {e}")

        print("-" * 50)  # 分隔不同param的处理结果