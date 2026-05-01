import numpy as np
import pickle


# 简单版本，专注于处理QubitOperator
def main_simple():
    # 加载pkl文件
    with open('../H6/H6_3.4_sto-3g.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data)
    print("数据键值:", data.keys())
    print("qubit_hamiltonian类型:", type(data['qubit_hamiltonian']))

    # 直接打印QubitOperator内容
    qubit_op = data['qubit_hamiltonian']

    if hasattr(qubit_op, 'terms'):
        print("QubitOperator项:")
        for term, coeff in qubit_op.terms.items():
            print(f"  {term}: {coeff}")

    # 尝试转换为字符串
    qubit_str = str(qubit_op)
    print("\n字符串表示:")
    print(qubit_str[:500] + "..." if len(qubit_str) > 500 else qubit_str)


# 先运行简单版本查看数据结构
main_simple()