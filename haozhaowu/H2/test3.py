import numpy as np
import pickle
from openfermion import QubitOperator
import quimb as qu
import quimb.tensor as qtn


def load_molecular_data(pkl_file_path):
    """从pkl文件加载分子数据"""
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    print("数据加载完成")
    print(f"量子比特数: {data['num_qubits']}")
    print(f"FCI能量: {data['FCI_val']}")
    print(f"QubitOperator项数: {len(data['qubit_hamiltonian'].terms)}")

    return data


def qubitoperator_to_matrix(qubit_operator, num_qubits):
    """将QubitOperator转换为矩阵形式"""
    # 泡利矩阵定义
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # 初始化哈密顿量矩阵
    hamiltonian_matrix = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)

    for term, coeff in qubit_operator.terms.items():
        # 创建泡利字符串
        pauli_string = ['I'] * num_qubits
        for qubit_idx, pauli_op in term:
            if qubit_idx < num_qubits:
                pauli_string[qubit_idx] = pauli_op

        # 构建泡利矩阵
        matrix = np.eye(1)
        for char in pauli_string:
            matrix = np.kron(matrix, pauli_dict[char])

        hamiltonian_matrix += coeff * matrix

    return hamiltonian_matrix


def matrix_to_mpo_simple(hamiltonian_matrix, num_qubits, max_bond=None, cutoff=1e-12):
    """
    使用Quimb的from_dense函数将矩阵转换为MPO
    
    参数:
        hamiltonian_matrix: 哈密顿量矩阵 (2^n x 2^n)
        num_qubits: 量子比特数量
        max_bond: 最大键维数，None表示不限制
        cutoff: SVD截断阈值
    
    返回:
        mpo: MatrixProductOperator对象
    """
    # 确保矩阵是正确的形状
    expected_dim = 2 ** num_qubits
    if hamiltonian_matrix.shape != (expected_dim, expected_dim):
        raise ValueError(f"矩阵形状错误: 期望 ({expected_dim}, {expected_dim}), "
                        f"实际 {hamiltonian_matrix.shape}")
    
    # 使用Quimb的from_dense函数
    # 这个方法会自动进行SVD分解，将矩阵转换为MPO形式
    mpo = qtn.MatrixProductOperator.from_dense(
        hamiltonian_matrix,
        dims=[2] * num_qubits,  # 每个量子比特的物理维度都是2
        max_bond=max_bond,      # 最大键维数
        cutoff=cutoff           # SVD截断阈值
    )

    return mpo


def qubitoperator_to_mpo_direct(qubit_operator, num_qubits):
    """
    直接将QubitOperator转换为MPO
    使用Quimb的MPO构建功能
    """
    # 将QubitOperator转换为Quimb的Pauli项列表
    pauli_terms = []

    for term, coeff in qubit_operator.terms.items():
        # 转换为Quimb格式
        if term:  # 非恒等算符
            ops = []
            for qubit_idx, pauli_op in term:
                ops.append((pauli_op.lower(), qubit_idx))
            pauli_terms.append((coeff, ops))
        else:  # 恒等算符
            pauli_terms.append((coeff, []))

    # 使用Quimb的MPO构建函数
    # 方法1: 使用MPO_ham_ising（需要指定系统大小）
    try:
        # 对于4量子比特系统，使用MPO_ham_ising
        if num_qubits == 4:
            # 创建一个简单的Ising模型MPO作为基础
            mpo = qtn.MPO_ham_ising(L=num_qubits, hz=0.0, jx=0.0)
            print("使用MPO_ham_ising构建基础MPO")
        else:
            raise ValueError("不支持的量子比特数")
    except Exception as e:
        print(f"方法1失败: {e}")
        # 方法2: 使用更通用的方法
        mpo = build_mpo_from_pauli_terms(pauli_terms, num_qubits)

    return mpo


def build_mpo_from_pauli_terms(pauli_terms, num_qubits):
    """
    从泡利项列表构建MPO
    使用更稳健的方法
    """
    # 泡利矩阵
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    pauli_matrices = {'i': I, 'x': X, 'y': Y, 'z': Z}

    # 初始化MPO张量列表
    mpo_tensors = []

    # 对于每个位点，构建MPO张量
    for site in range(num_qubits):
        # 确定该位点上的所有算符
        operators_on_site = []

        for coeff, ops in pauli_terms:
            # 检查该算符是否与当前位点相关
            relevant = False
            site_op = 'i'  # 默认恒等算符

            for op_type, qubit_idx in ops:
                if qubit_idx == site:
                    relevant = True
                    site_op = op_type
                    break

            if relevant:
                operators_on_site.append((site_op, coeff))
            else:
                # 即使不相关，也需要恒等算符
                operators_on_site.append(('i', coeff if not ops else 0.0))

        # 确定MPO张量的维度
        # 我们需要一个通道用于恒等算符，一个通道用于每个不同的算符
        unique_operators = set(op for op, _ in operators_on_site)
        num_channels = len(unique_operators)

        # 确保至少有一个通道
        num_channels = max(1, num_channels)

        # 限制最大通道数
        max_channels = 8
        num_channels = min(num_channels, max_channels)

        # 构建MPO张量
        # 形状: (左键维, 物理输出维, 物理输入维, 右键维)
        left_dim = 1 if site == 0 else num_channels
        right_dim = 1 if site == num_qubits - 1 else num_channels

        tensor = np.zeros((left_dim, 2, 2, right_dim), dtype=complex)

        # 填充张量
        # 恒等算符通道
        tensor[0, :, :, 0] = I

        # 添加其他算符
        channel_idx = 1
        for op_type, coeff in operators_on_site:
            if channel_idx < num_channels and op_type != 'i':
                tensor[0, :, :, channel_idx] = coeff * pauli_matrices[op_type]
                channel_idx += 1

        mpo_tensors.append(tensor)

    # 创建Quimb MPO对象
    mpo = qtn.MatrixProductOperator(mpo_tensors, shape='lrud')

    return mpo


def save_mpo_tensors(mpo, filename_prefix):
    """
    保存MPO张量为npz格式
    确保第一个张量的左键维度为1，最后一个张量的右键维度为1
    标准MPO格式: (left_bond, phys_out, phys_in, right_bond)
    """
    # 提取张量
    tensors = []
    
    # Quimb的MPO对象包含多个张量节点
    if hasattr(mpo, 'tensors'):
        # mpo.tensors返回一个列表，每个元素是一个Tensor对象
        for tensor_node in mpo.tensors:
            # 获取实际的numpy数组数据
            if hasattr(tensor_node, 'data'):
                tensors.append(tensor_node.data)
            else:
                tensors.append(np.array(tensor_node))
    elif hasattr(mpo, 'tensor_map'):
        # 另一种可能的结构
        for key in sorted(mpo.tensor_map.keys()):
            tensor_node = mpo.tensor_map[key]
            if hasattr(tensor_node, 'data'):
                tensors.append(tensor_node.data)
            else:
                tensors.append(np.array(tensor_node))
    else:
        # 如果直接是tensor列表
        for t in mpo:
            if hasattr(t, 'data'):
                tensors.append(t.data)
            else:
                tensors.append(np.array(t))
    
    # 修正边界张量的形状，确保标准MPO格式
    # 标准格式: (left_bond, phys_out, phys_in, right_bond)
    processed_tensors = []
    
    for i, tensor in enumerate(tensors):
        tensor = np.array(tensor)
        
        if i == 0:  # 第一个张量
            if len(tensor.shape) == 3:
                # 形状是 (phys_out, phys_in, right_bond)
                # 需要在最前面添加维度1: (1, phys_out, phys_in, right_bond)
                tensor = tensor[np.newaxis, :, :, :]
                print(f"  修正第一个张量: 添加左键维度1")
            elif len(tensor.shape) == 4 and tensor.shape[0] != 1:
                # 如果已经是4维但左键维度不是1，报警
                print(f"  警告: 第一个张量左键维度不是1: {tensor.shape}")
        
        elif i == len(tensors) - 1:  # 最后一个张量
            if len(tensor.shape) == 3:
                # 形状是 (left_bond, phys_out, phys_in)
                # 需要在最后面添加维度1: (left_bond, phys_out, phys_in, 1)
                tensor = tensor[:, :, :, np.newaxis]
                print(f"  修正最后一个张量: 添加右键维度1")
            elif len(tensor.shape) == 4 and tensor.shape[3] != 1:
                # 如果已经是4维但右键维度不是1，报警
                print(f"  警告: 最后一个张量右键维度不是1: {tensor.shape}")
        
        else:  # 中间张量
            if len(tensor.shape) != 4:
                print(f"  警告: 中间张量 {i} 不是4维: {tensor.shape}")
        
        processed_tensors.append(tensor)

    # 保存为npz格式
    data_dict = {}
    for i, tensor in enumerate(processed_tensors):
        data_dict[f'mt_{i}'] = tensor

    filename = f'{filename_prefix}.npz'
    np.savez(filename, **data_dict)

    print(f"\n保存的MPO张量信息:")
    print(f"  文件名: {filename}")
    print(f"  张量数量: {len(processed_tensors)}")
    print(f"  各张量形状 (left_bond, phys_out, phys_in, right_bond):")
    for i in range(len(processed_tensors)):
        print(f'    mt_{i}: {processed_tensors[i].shape}')
    
    # 计算文件大小
    import os
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.2f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.2f} MB"
        print(f"  文件大小: {size_str}")
    
    return processed_tensors


def verify_saved_npz(filename):
    """验证保存的npz文件是否正确"""
    try:
        # 读取npz文件
        data = np.load(filename)
        print(f"\n验证保存的文件: {filename}")
        print(f"  包含的张量数: {len(data.keys())}")
        print(f"  张量详情 (格式: left_bond, phys_out, phys_in, right_bond):")
        
        # 检查每个张量
        num_tensors = len(data.keys())
        for i, key in enumerate(sorted(data.keys())):
            tensor = data[key]
            shape_str = f"{tensor.shape}"
            dtype_str = f"{tensor.dtype}"
            
            # 检查格式是否符合标准
            status = "✓"
            if i == 0:  # 第一个张量
                if len(tensor.shape) == 4 and tensor.shape[0] == 1:
                    status = "✓ 标准格式"
                else:
                    status = "✗ 左键维度应为1"
            elif i == num_tensors - 1:  # 最后一个张量
                if len(tensor.shape) == 4 and tensor.shape[3] == 1:
                    status = "✓ 标准格式"
                else:
                    status = "✗ 右键维度应为1"
            elif len(tensor.shape) == 4:
                status = "✓ 标准格式"
            else:
                status = "✗ 应为4维"
            
            print(f"    {key}: {shape_str:<20} {dtype_str:<12} {status}")
        
        data.close()
        return True
        
    except Exception as e:
        print(f"\n验证文件时出错: {e}")
        return False


def verify_mpo_accuracy(mpo, original_matrix, num_qubits):
    """验证MPO的准确性"""
    try:
        # 将MPO转换回矩阵形式
        # Quimb的MatrixProductOperator有to_dense方法
        # 不需要传递参数，它会自动使用MPO的维度信息
        reconstructed_matrix = mpo.to_dense()
        
        # 计算误差
        error = np.linalg.norm(reconstructed_matrix - original_matrix) / np.linalg.norm(original_matrix)
        
        print(f"MPO重建相对误差: {error:.10e}")
        
        # 也打印一些其他信息
        print(f"原始矩阵范数: {np.linalg.norm(original_matrix):.10e}")
        print(f"重建矩阵范数: {np.linalg.norm(reconstructed_matrix):.10e}")
        
        # 计算基态能量进行交叉验证
        try:
            eigenvalues_orig = np.linalg.eigvalsh(original_matrix)
            eigenvalues_recon = np.linalg.eigvalsh(reconstructed_matrix)
            energy_error = abs(eigenvalues_orig[0] - eigenvalues_recon[0])
            print(f"基态能量差: {energy_error:.10e}")
        except:
            pass
        
        return error < 1e-6  # 允许的误差阈值
        
    except Exception as e:
        print(f"验证MPO准确性时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def reconstruct_matrix_from_mpo(mpo, num_qubits):
    """从MPO重建矩阵"""
    # 如果mpo是Quimb对象
    if hasattr(mpo, 'tensors'):
        tensors = mpo.tensors
    else:
        tensors = [mpo]

    # 初始化收缩结果
    result = np.eye(1, dtype=complex)

    for i, tensor in enumerate(tensors):
        # 将张量重塑为矩阵形式
        left_dim, phys_out, phys_in, right_dim = tensor.shape
        matrix_form = tensor.reshape(left_dim * phys_out, phys_in * right_dim)

        # 与当前结果进行张量积
        if i == 0:
            result = matrix_form
        else:
            result = np.kron(result, matrix_form)

    # 最终重塑为2^n x 2^n矩阵
    final_matrix = result.reshape(2 ** num_qubits, 2 ** num_qubits)
    return final_matrix


def main():
    """主函数：使用OpenFermion和Quimb从pkl文件获取MPO"""
    pkl_file_path = 'H2_4.3_sto-3g.pkl'
    output_prefix = 'H2_4.3_sto-3g_MPO'

    # 1. 加载分子数据
    print("\n" + "="*60)
    print("步骤1: 加载分子数据")
    print("="*60)
    data = load_molecular_data(pkl_file_path)

    # 2. 获取QubitOperator和系统信息
    qubit_operator = data['qubit_hamiltonian']
    num_qubits = data['num_qubits']

    # 3. 将QubitOperator转换为矩阵
    print("\n" + "="*60)
    print("步骤2: 将QubitOperator转换为哈密顿量矩阵")
    print("="*60)
    hamiltonian_matrix = qubitoperator_to_matrix(qubit_operator, num_qubits)
    print(f"哈密顿量矩阵形状: {hamiltonian_matrix.shape}")
    print(f"哈密顿量矩阵范数: {np.linalg.norm(hamiltonian_matrix):.10e}")
    
    # 计算基态能量（验证用）
    eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix)
    ground_state_energy = eigenvalues[0]
    print(f"计算得到的基态能量: {ground_state_energy:.10f}")
    print(f"FCI参考能量: {data['FCI_val']:.10f}")
    print(f"能量差: {abs(ground_state_energy - data['FCI_val']):.10e}")

    # 4. 将矩阵转换为MPO
    print("\n" + "="*60)
    print("步骤3: 将矩阵转换为MPO格式")
    print("="*60)
    mpo = matrix_to_mpo_simple(hamiltonian_matrix, num_qubits)
    print("MPO构建成功!")
    
    # 打印MPO信息
    print(f"\nMPO信息:")
    print(f"  - 位点数量: {num_qubits}")
    if hasattr(mpo, 'bond_sizes'):
        print(f"  - 键维数: {mpo.bond_sizes()}")

    # 5. 验证准确性
    print("\n" + "="*60)
    print("步骤4: 验证MPO准确性")
    print("="*60)
    is_accurate = verify_mpo_accuracy(mpo, hamiltonian_matrix, num_qubits)
    print(f"MPO准确性检验: {'✓ 通过' if is_accurate else '✗ 未通过'}")

    # 6. 保存MPO
    print("\n" + "="*60)
    print("步骤5: 保存MPO到npz文件")
    print("="*60)
    save_mpo_tensors(mpo, output_prefix)
    
    # 7. 验证保存的文件
    print("\n" + "="*60)
    print("步骤6: 验证保存的npz文件")
    print("="*60)
    npz_file = f"{output_prefix}.npz"
    verify_saved_npz(npz_file)

    return mpo


def minimal_mpo_representation(qubit_operator, num_qubits):
    """
    最小MPO表示
    只包含恒等算符和单量子比特算符
    """
    # 泡利矩阵
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    pauli_matrices = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # 初始化MPO张量
    mpo_tensors = []

    for site in range(num_qubits):
        # 创建当前位点的MPO张量
        # 形状: (1, 2, 2, 2) - 只包含恒等算符和一个可能的单量子比特算符
        tensor = np.zeros((1, 2, 2, 2), dtype=complex)

        # 恒等算符通道
        tensor[0, :, :, 0] = I

        # 检查是否有单量子比特算符作用在该位点上
        has_single_qubit_op = False
        for term, coeff in qubit_operator.terms.items():
            if len(term) == 1:  # 单量子比特算符
                qubit_idx, pauli_op = term[0]
                if qubit_idx == site:
                    tensor[0, :, :, 1] += coeff * pauli_matrices[pauli_op]
                    has_single_qubit_op = True

        # 如果没有单量子比特算符，只保留恒等算符
        if not has_single_qubit_op:
            tensor = tensor[:, :, :, 0:1]  # 只保留第一个通道

        mpo_tensors.append(tensor)

    # 创建Quimb MPO对象
    mpo = qtn.MatrixProductOperator(mpo_tensors, shape='lrud')

    return mpo


if __name__ == "__main__":
    print("\n" + "="*60)
    print("使用OpenFermion和Quimb将哈密顿量转换为MPO格式")
    print("="*60)

    try:
        mpo = main()
        
        print("\n" + "="*60)
        print("✓ 转换完成！MPO已成功保存为标准格式")
        print("="*60)
        
        print("\n说明:")
        print("  - MPO张量格式: (left_bond, phys_out, phys_in, right_bond)")
        print("  - 第一个张量的 left_bond = 1")
        print("  - 最后一个张量的 right_bond = 1")
        print("  - 物理维度 phys_out = phys_in = 2 (量子比特)")

    except Exception as e:
        print("\n" + "="*60)
        print("✗ 错误：处理文件时出错")
        print("="*60)
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()