'''

Created Date: Tue Nov 12 2023

Revised Date: Tue Dec 9 2023

Revised Date: Mon Mar 24 2025

Author: Qiming Ding, Huiyuan Wang, Yukun Zhang

Description: H2 631g

Copyright (c) 2023

'''
import os
import time
from functools import reduce, partial
from itertools import product
import concurrent.futures
from typing import Union, List, Optional, Tuple, Any
from scipy.linalg import eigh
import datetime
import itertools
from pathlib import Path
import functools
import tqdm
import pickle
import cvxpy as cp
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
np.set_printoptions(suppress=True, precision=6, threshold=np.inf)
from joblib import Memory, Parallel, delayed

def measure_execution_time(func):
    """
    Decorator function to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result

    return wrapper

def is_invalid_index(i: int, j: int, k: int, l: int, num_spin_orbitals: int) -> bool:
    """
    Check if the given indices are invalid based on the number of spin orbitals.

    Args:
        i: The first index.
        j: The second index.
        k: The third index.
        l: The fourth index.
        num_spin_orbitals: The total number of spin orbitals.

    Returns:
        True if the indices are invalid, False otherwise.
    """
    valid_term = (
        (i < num_spin_orbitals // 2 and j < num_spin_orbitals // 2 and k < num_spin_orbitals // 2 and l < num_spin_orbitals // 2) or
        (i < num_spin_orbitals // 2 and j >= num_spin_orbitals // 2 and k < num_spin_orbitals // 2 and l >= num_spin_orbitals // 2) or
        (i < num_spin_orbitals // 2 and j >= num_spin_orbitals // 2 and k >= num_spin_orbitals // 2 and l < num_spin_orbitals // 2) or
        (i >= num_spin_orbitals // 2 and j < num_spin_orbitals // 2 and k < num_spin_orbitals // 2 and l >= num_spin_orbitals // 2) or
        (i >= num_spin_orbitals // 2 and j < num_spin_orbitals // 2 and k >= num_spin_orbitals // 2 and l < num_spin_orbitals // 2) or
        (i >= num_spin_orbitals // 2 and j >= num_spin_orbitals // 2 and k >= num_spin_orbitals // 2 and l >= num_spin_orbitals // 2)
    )
    return not valid_term

def check_two_rdm_standard(two_rdm, num_spin_orbitals):
    for i in range(num_spin_orbitals):
        for j in range(num_spin_orbitals):
            for k in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    if is_invalid_index(i, j, k, l, num_spin_orbitals) and two_rdm[i, j, k, l] != 0:
                        return False
    return True

def kron_to_ikjl(two_rdm_ref_ij_kl):
    """
    将一个经过压缩或重塑为二维数组的四阶张量转换回其原始的四维形式，
    并将其元素从"ij-kl"格式排列改为"ik-jl"格式。

    输入:
    - two_rdm_ref_ij_kl (numpy.ndarray): 一个二维数组，表示一个被压缩或重塑的四阶张量，
                                          形状为(num_spin_orbitals^2, num_spin_orbitals^2)，
                                          其中 num_spin_orbitals 是自旋轨道的数量。

    输出:
    - numpy.ndarray: 一个四维张量，其元素根据"ik-jl"格式排列，
                     形状为(num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals)。
    """

    # 计算自旋轨道的数量
    num_spin_orbitals = int(np.sqrt(two_rdm_ref_ij_kl.shape[0]))

    # 初始化一个新的四维张量来存储重排后的数据
    two_rdm_ref_ikjl = np.ndarray((num_spin_orbitals,) * 4)

    # 遍历每个索引，重新排列张量的元素
    for i in range(num_spin_orbitals):
        for k in range(num_spin_orbitals):
            for j in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    # 重新排列为"ik-jl"格式
                    two_rdm_ref_ikjl[i, k, j, l] = two_rdm_ref_ij_kl[i*num_spin_orbitals + j, k*num_spin_orbitals + l].value

    return two_rdm_ref_ikjl

def ikjl_to_kron(two_rdm_ref_ikjl):
    """
    将一个按照"ik-jl"格式排列的四阶张量转换为按照"ij-kl"格式排列的二维数组。

    输入:
    - two_rdm_ref_ikjl (numpy.ndarray): 一个四维张量，按照"ik-jl"格式排列，
                                         形状为(num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals)，
                                         其中 num_spin_orbitals 是自旋轨道的数量。

    输出:
    - numpy.ndarray: 一个二维数组，按照"ij-kl"格式排列，
                     形状为(num_spin_orbitals^2, num_spin_orbitals^2)。
    """

    # 获取自旋轨道的数量
    num_spin_orbitals = two_rdm_ref_ikjl.shape[0]

    # 初始化一个新的二维数组来存储重排后的数据
    two_rdm_ref_ij_kl = np.ndarray((num_spin_orbitals**2,)*2)

    # 遍历每个索引，重新排列张量的元素
    for i in range(num_spin_orbitals):
        for k in range(num_spin_orbitals):
            for j in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    # 重新排列为"ij-kl"格式
                    two_rdm_ref_ij_kl[i*num_spin_orbitals + j, k*num_spin_orbitals + l] = two_rdm_ref_ikjl[i, k, j, l]

    return two_rdm_ref_ij_kl

def kron_to_ijkl(two_rdm_ref_ij_kl):
    return np.einsum('ikjl->ijkl',kron_to_ikjl(two_rdm_ref_ij_kl))

def expr_to_val (M_ikjl):
    return np.apply_along_axis(lambda arr: np.array([expr.value for expr in arr]), M_ikjl.ndim-1,M_ikjl)

def validate_inputs(two_rdm, num_particles, num_spin_orbitals):
    """
    验证输入的有效性，包括two_rdm的维度和旋轨道数是否符合条件。
    """
    assert two_rdm.ndim == 4, 'two_rdm given is not a fourth-order tensor'
    assert num_spin_orbitals >= num_particles, 'the number of available orbitals should be no less than the number of fermions'

def prepare_hamiltonian(one_body_coefficients, two_body_coefficients, two_rdm):
    """
    准备哈密顿量的一体和二体系数。
    """
    h1_ij = np.array(one_body_coefficients) if one_body_coefficients is not None else None
    h2_ijkl = np.array(two_body_coefficients) if two_body_coefficients is not None else None
    if h2_ijkl is not None:
        assert h2_ijkl.shape == two_rdm.shape, 'the shape of h2_ijkl should be the same as two_rdm'
    return h1_ij, h2_ijkl

def check_tensor_convention(two_rdm):
    """
    检查并调整two_rdm的索引约定。
    """
    num_particles_ref = np.einsum('ijij', two_rdm)
    flag_ijkl = num_particles_ref > 0
    num_particles_ref = (1 + np.sqrt(1 + 4 * abs(num_particles_ref))) / 2
    print(f"Initially {num_particles_ref} particles measured")

    if not flag_ijkl:
        two_rdm = np.einsum('ijlk->ijkl', two_rdm)
        print("Index convention: ijlk. Changing the input tensor convention from ijlk->ijkl.")
        print("The given two-rdm tensor is of index ijlk, i.e.a two electron wavefunction 1,2 would be encoded as 1,2,2,1")
        print("This function would not change the convention between input and output tensor")
    else:
        print("Index convention: ijkl. Not changing the input tensor convention.")
        print("The given two-rdm tensor is of index ijkl, i.e. a two electron wavefunction 1,2 would be encoded as 1,2,1,2")
        print("This function would not change the convention between input and output tensor")         
    return two_rdm, flag_ijkl
def to_kron_with_constraint(M_ikjl, constraints, pos_def=True):
    ##return the two dimensional alias of the matrix with equality and positive definiteness constraint
    offset= M_ikjl.shape[0]
    ## group the (i,j) as a composite index and (k,l) as a composite index
    M_ij_kl = cp.Variable((offset**2, offset**2))
    assert M_ij_kl.is_matrix(), 'should convert to a two dimensional cvxpy matrix'
    
    for i in range (offset):
        for k in range(offset):
            for j in range(offset):
                for l in range (offset):
                      constraints.append(M_ij_kl[i*offset+j][k*offset+l] == M_ikjl[i,k][j,l])
    if pos_def:
        constraints.append(M_ij_kl>>0)
        
    return M_ij_kl, constraints


def symmetrize_2rdm (M, constraints):   
    if type(M)==np.ndarray:
        offset= M.shape[0]
    if type(M)==cp.Variable:
        offset= int(np.sqrt(M.shape[0]))  
    for i in range (offset):
        for k in range(offset):
            for j in range(offset):
                for l in range (offset):
                    if type(M)==np.ndarray:
                        ## M is forth order tensor in ikjl index
                            if i<j : constraints.append(M[i,k][j,l]== - M[j,k][i,l])
                            if k<l : constraints.append(M[i,k][j,l]== - M[i,l][j,k])
                            if (i,j)<(k,l): constraints.append(M[i,k][j,l]== M[k,i][l,j])
                    if type(M)==cp.Variable:
                        ## M is a kronecker product two-dimensional matrix in (i*offset+j, k*offset+l) index
                            if i<j :constraints.append(M[i*offset+j][k*offset+l]== - M[j*offset+i][k*offset+l])
                            if k<l :constraints.append(M[i*offset+j][k*offset+l]== - M[i*offset+j][l*offset+k])
                            if (i,j)<(k,l): constraints.append(M[i*offset+j][k*offset+l]== M[k*offset+l][i*offset+j])                       
    return M,constraints
def check_2rdm_properties(tensor):
    offset = tensor.shape[0]
    for i in range(offset):
        for j in range(offset):
            for k in range(offset):
                for l in range(offset):
                    # Check Hermitian property
                    if tensor[i, j, k, l] != tensor[k, l, i, j]:
                        print(f"不满足厄米性质", (i, j, k, l), (k, l, i, j))

                    # Check antisymmetry property
                    if i != j and tensor[i, j, k, l] != -tensor[j, i, k, l]:
                        print(f"不满足反对称性质", {(i, j, k, l)}, {(j, i, k, l)})
                    if k != l and tensor[i, j, k, l] != -tensor[i, j, l, k]:
                        print(f"不满足反对称性质", {(i, j, k, l)}, {(i, j, l, k)})
    print(f"=================================================================")
    return "满足"
def optimize(two_rdm, num_particles, one_body_coefficients=None, two_body_coefficients=None, epsilon=None, max_iters=20000, verbose=True, accuracy=1e-10):
    
    # 检查张量顺序和Pauli排斥原理
    two_rdm = np.array(two_rdm)
    num_spin_orbitals = two_rdm.shape[0]
    constraints = []

    # 验证输入
    validate_inputs(two_rdm, num_particles, num_spin_orbitals)

    # 检查是否需要最小化能量
    flag_to_minimize_E = bool(two_body_coefficients is not None and one_body_coefficients is not None)

    # 准备哈密顿量
    h1_ij, h2_ijkl = prepare_hamiltonian(one_body_coefficients, two_body_coefficients, two_rdm)
    
    # 检查张量约定
    two_rdm, flag_ijkl = check_tensor_convention(two_rdm)

    # 转换张量
    
    is_standard = check_two_rdm_standard(two_rdm, num_spin_orbitals)
    
    print(f"two_rdm is {is_standard}")
    
    two_rdm_ref_ikjl = np.einsum('ijkl->ikjl', two_rdm)
    
    
    print(np.shape(two_rdm_ref_ikjl))
    
    two_rdm_ref_ij_kl = cp.Constant(ikjl_to_kron(two_rdm_ref_ikjl))

    # 生成 Kronecker Delta 张量
    
    kd_ijkl = np.einsum('ij,kl->ijkl', np.eye(num_spin_orbitals), np.eye(num_spin_orbitals))
    
    two_rdm_ref_ikjl_echo = np.einsum('ijkl->ikjl',two_rdm)
    
    two_rdm_ref_ij_kl_echo = ikjl_to_kron(two_rdm_ref_ikjl_echo)

    ##prepare in cvxpy constant form
    ##kronecker delta ijkl
    
    kd_ijkl= np.einsum('ij,kl->ijkl',np.eye(num_spin_orbitals),np.eye(num_spin_orbitals))
    
    ##define an 2d nparray of 2d cvxpy variable object in the index of ik jl respectively
    ##since cvxpy does not support variable with dimensions greater than 2
    
    var_ls= []
    for i in range (num_spin_orbitals):
        temp=[]
        for k in range (num_spin_orbitals):
            temp.append(cp.Variable((num_spin_orbitals, num_spin_orbitals)))
        var_ls.append(temp)
        
    two_rdm_new_ikjl= np.array(var_ls, dtype=object)
    
    print(np.shape(two_rdm_new_ikjl))
    
    for i in range(num_spin_orbitals):
        for k in range(num_spin_orbitals):
            for j in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    # 如果参考矩阵在(i, k, j, l)位置的值为0，则在优化变量矩阵中设置相应的约束
                    if i == j or l == k or is_invalid_index(i, j, k, l, num_spin_orbitals):
                        # print(two_rdm_ref_ikjl[i][k][j][l])
                        # print(two_rdm_new_ikjl[i][k][j][l])
                        constraints.append(two_rdm_new_ikjl[i][k][j][l] == cp.Constant(0))
                    # if i == j or l == k or invalid(i, j, k, l, num_spin_orbitals):
                    #    constraints.append(two_rdm_new_ikjl[i][k][j][l] == cp.Constant(0))
                        
    ##tracing to give one_rdm_new # Test_one_RDM = np.einsum('prrq', FCI_two_RDM)np.isclose(two_rdm_ref_ikjl[i][k][j][l], 0, atol=1e-8):
                       # constraints.append(two_rdm_new_ikjl[i][k][j][l] == cp.Constant(0))
                    # 您之前的条件可以保留，或者根据需要进行修改
                    # elif
    
    one_rdm_new=  1/(num_particles-1)*(np.vectorize(cp.trace)(two_rdm_new_ikjl))                        
    one_rdm_ref=  1/(num_particles-1)*np.trace(two_rdm_ref_ikjl)
    
    ## compute the Q-matrix in Q_ikjl 
    ## Q_ikjl = 2rdm_ikjl + kd_ijkl -kd_ilkj - 1rdm_ij kd_kl -1rdm_kl kd_ij + 1rdm_jk kd_il + 1rdm_il kd_jk
    
    kd_ij_1rdm_kl = np.ndarray((num_spin_orbitals,)*4,dtype=object)
    
    for i in range (num_spin_orbitals):
        for j in range (num_spin_orbitals):
            for k in range (num_spin_orbitals):
                for l in range (num_spin_orbitals):
                    if i==j:
                        kd_ij_1rdm_kl[i,j,k,l]= one_rdm_new[k,l]
                    else:
                        kd_ij_1rdm_kl[i,j,k,l]= cp.Constant(0)
                        
    Q_ikjl = np.ndarray((num_spin_orbitals,)*4,dtype=object)
    for i in range (num_spin_orbitals):
        for k in range (num_spin_orbitals):
            for j in range (num_spin_orbitals):
                for l in range (num_spin_orbitals):
                        Q_ikjl[i,k,j,l]= two_rdm_new_ikjl[i,k][j,l] + kd_ijkl[j,l,i,k]-kd_ijkl[i,l,j,k]\
                            - kd_ij_1rdm_kl[j,l,i,k]-kd_ij_1rdm_kl[i,k,j,l]+kd_ij_1rdm_kl[i,l,j,k]+kd_ij_1rdm_kl[j,k,i,l]
                            
    ##compute the G-matrix in G_ikjl
    ##G_ikjl = kd_kl_1rdm_ij - 2rdm_iljk
    
    G_ikjl = np.ndarray((num_spin_orbitals,)*4,dtype=object)
    for i in range (num_spin_orbitals):
        for k in range (num_spin_orbitals):
            for j in range (num_spin_orbitals):
                for l in range (num_spin_orbitals):
                        G_ikjl[i,k,j,l]= kd_ij_1rdm_kl[j,l,i,k]-two_rdm_new_ikjl[i,k][l,j]

    ##symmetry constraints
    ##symmetrizing 2rdm should automatically enforce symmetry of 1rdm, P, and Q
    
    two_rdm_new_ikjl, constraints = symmetrize_2rdm(two_rdm_new_ikjl, constraints)

    ##Positivity constraints
                
    two_rdm_new_ij_kl, constraints = to_kron_with_constraint(two_rdm_new_ikjl, constraints)
   
    G_ij_kl, constraints = to_kron_with_constraint(G_ikjl, constraints)
    Q_ij_kl, constraints = to_kron_with_constraint(Q_ikjl, constraints)
    
    ##Trace constraints
    
    ##Tr{2rdm}=N*(N-1) this should automatically enforce the trace of 1rdm, P, and Q
    
    constraints.append(cp.trace(two_rdm_new_ij_kl)==num_particles*(num_particles-1))
    
    ##set up the objective
    ##to minimize the Frobenius norm between the optimized 2rdm and the reference 2rdm 
    
    if flag_to_minimize_E:
        if epsilon is not None:
            if np.isscalar(epsilon):
                # epsilon 是一个标量
                constraints.append(cp.norm(two_rdm_new_ij_kl - two_rdm_ref_ij_kl, 'fro') <= epsilon)                
            elif epsilon.shape == two_rdm_new_ij_kl.shape:                
                n_dim = two_rdm_new_ij_kl.shape[0]
                for i in range(n_dim):
                    for j in range(n_dim):
                        if epsilon[i][j] != 0 and np.any(two_rdm_ref_ij_kl_echo[i][j] != 0):
                            epsilon_ij_kl = cp.Constant(np.abs(epsilon[i][j]))
                            two_rdm_ref_ij_kl_echo_new = cp.Constant(two_rdm_ref_ij_kl_echo[i][j])                                     
                            constraints.append(cp.norm(two_rdm_new_ij_kl[i,j] - two_rdm_ref_ij_kl_echo_new,1) <= epsilon_ij_kl)
            else:
                # epsilon 的维度既不是标量也不与 two_rdm_new_ij_kl 一致
                raise ValueError("epsilon 的维度不正确。它必须是一个标量或与 two_rdm_new_ij_kl 的维度一致。")
                               
        h2_ikjl=np.einsum('ijkl->ikjl',h2_ijkl)
        
        h2_ij_kl=cp.Constant(ikjl_to_kron(h2_ikjl))
        
        E = np.trace(np.matmul(h1_ij,one_rdm_new))+ 0.5*cp.trace(h2_ij_kl@two_rdm_new_ij_kl)
        
        # E = np.trace(np.matmul(h1_ij,one_rdm_new))+0.5*np.trace(np.matmul(h2_ij_kl,two_rdm_new_ij_kl))
        objective = cp.Minimize(E)
        
        print("the objective is to find the lowest eigenvalue subject to n-representability and closeness to measurements")
    else:
        objective= cp.Minimize(cp.norm(two_rdm_new_ij_kl - two_rdm_ref_ij_kl, 'fro'))
        print("the objective is the find the nearest n-representable state")
    ##check if all constraints are legal
    
    assert np.all(list(map(lambda c: c.is_dcp(),constraints))), 'all constraints should be disciplined convex programming'
    print('all constraints checked to be disciplined convex programming')

    ##check if the objective is legal
    assert objective.is_dcp(), 'the objective should be disciplined convex programming'
    print('objective checked to be disciplined convex programming')

    ##create and solve the problem
    problem= cp.Problem(objective, constraints)
    
    try:
        problem.solve(verbose=verbose, solver=cp.SCS, eps=accuracy, max_iters=max_iters)
    except Exception as e:
        print(e)
        
    # 获取最优值
    optimal_value = problem.value
    
    ##summarize data
    two_rdm_res_ijkl= kron_to_ijkl(two_rdm_new_ij_kl)
    
    one_rdm_res_ij= expr_to_val(one_rdm_new)
    G_ijkl = kron_to_ijkl(G_ij_kl)
    Q_ijkl = kron_to_ijkl(Q_ij_kl)
    
    if not flag_ijkl:
        two_rdm_res_ijkl=np.einsum('ijkl->ijlk',two_rdm_res_ijkl)
        G_ijkl=np.einsum('ijkl->ijlk',G_ijkl)
        Q_ijkl=np.einsum('ijkl->ijlk',Q_ijkl)
        
        # print("please note that all forth order tensor is output in ijlk convention, not changing the convention of the input tensor.")
        
    else:
        print("please note that all forth order tensor is output in ijkl convention, not changing the convention of the input tensor.")
        
    if flag_to_minimize_E:
        if verbose:
            return E.value,two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl
        else:
            return E.value,two_rdm_res_ijkl, one_rdm_res_ij
        
    if verbose:
        return two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl, optimal_value
    else:
        return two_rdm_res_ijkl, one_rdm_res_ij, optimal_value
    
def legalize_then_optimize(two_rdm, num_particles, one_body_coefficients, two_body_coefficients, verbose=True, max_iters=30000, accuracy=1e-7, epsilon=None):
    """
    先合法化（legalize）二阶还原密度矩阵（2-RDM），然后进行优化。

    参数:
    two_rdm -- 实验得到的二阶还原密度矩阵
    num_particles -- 粒子数
    one_body_coefficients -- 单体系数
    two_body_coefficients -- 二体系数
    verbose -- 是否显示详细信息
    max_iters -- 最大迭代次数
    accuracy -- 精度
    epsilon -- 用于优化的参数

    返回:
    优化后的二阶还原密度矩阵及相关信息
    """
    two_rdm_legal, _, _, _, _ = optimize(two_rdm, num_particles, one_body_coefficients = None, two_body_coefficients = None, max_iters=max_iters, verbose=verbose, accuracy=accuracy)
    
    print("============================== Legalization was completed in legalize_then_optimize function===============================")
    
    E,two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl = optimize(two_rdm_legal, num_particles, one_body_coefficients, two_body_coefficients, epsilon=epsilon, verbose=verbose, max_iters=max_iters, accuracy=accuracy)
    
    return E,two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl

def legalized(two_rdm, num_particles, one_body_coefficients, two_body_coefficients, verbose=True, max_iters=30000, accuracy=1e-7, epsilon=None):
    """
    合法化二阶还原密度矩阵，并返回合法化后的矩阵和其他相关信息。

    参数:
    two_rdm -- 实验得到的二阶还原密度矩阵
    num_particles -- 粒子数
    one_body_coefficients -- 单体系数
    two_body_coefficients -- 二体系数
    verbose -- 是否显示详细信息
    max_iters -- 最大迭代次数
    accuracy -- 精度

    返回:
    合法化后的二阶还原密度矩阵、epsilon 和优化后的能量值
    """
    two_rdm_legal, _, _, _, eps = optimize(two_rdm, num_particles, one_body_coefficients=None,two_body_coefficients=None, max_iters=max_iters, verbose=verbose, accuracy=accuracy)
    
    print(f"合法化后的在优化的two_rdm与two_rdm_legal的差{np.linalg.norm(two_rdm_legal - two_rdm)}，准备检查，理想值是{eps}")
    
    print("============================== Legalization was completed in legalize function===============================")
    
    E, two_rdm_res_ijkl, _, _, _ = optimize(two_rdm_legal, num_particles, one_body_coefficients, two_body_coefficients, epsilon=eps, verbose=verbose, max_iters=max_iters, accuracy=accuracy)
         
    print(f"合法化后的在优化的two_rdm_res_ijkl与two_rdm_legal的差{np.linalg.norm(two_rdm_legal - two_rdm_res_ijkl)}，准备检查，理想值是0")
    
    print(f"合法化后的在优化的two_rdm与two_rdm_res_ijkl的差{np.linalg.norm(two_rdm_res_ijkl - two_rdm)}，准备检查，理想值是0")
    
    return two_rdm_res_ijkl, eps, E


def find_optimal_eps(molecule, prob_1, prob_2, d, shots_num, ground_state_energy, c, num_particles, accuracy, one_body_coefficients, two_body_coefficients, experimental_2rdm):

    step_size = 0.1
    threshold = 1e-4
    iteration_info = []

    # 初始合法化和优化
    two_rdm_legal, current_eps, current_E = legalized(experimental_2rdm, num_particles, one_body_coefficients, two_body_coefficients, verbose=True, max_iters=30000, accuracy=1e-7)
    
    print(f"current_eps_init is {current_eps}, and energy is {current_E }.")
    
    iteration_info.append({'d': d, 'current_eps_now': current_eps, 'current_eps': current_eps, 'E': current_E, 'E_thm': ground_state_energy - c, 'iterations': 0})

    while True:
        E, _, _, _, _ = optimize(two_rdm_legal, num_particles, one_body_coefficients, two_body_coefficients, epsilon=current_eps, accuracy=accuracy)
        current_eps_now = current_eps

        if E + c < ground_state_energy:
            if step_size <= threshold:
                break
            else:
                current_eps -= step_size
                step_size /= 10
        else:
            current_eps += step_size
            
        iteration_info.append({'d': d, 'current_eps_now': current_eps_now, 'current_eps': current_eps, 'E': E, 'E_thm': ground_state_energy - c, 'iterations': len(iteration_info)})
    
    iteration_info_df = pd.DataFrame(iteration_info)
    print(iteration_info_df)

    # 确保变量是字符串并且适合用于文件命名
    molecule = str(molecule)  # 如果molecule是其他类型，确保转换为字符串
    prob_1 = str(prob_1)  # 同样保证prob_1是字符串
    prob_2 = str(prob_2)  # 同理
    d = str(d)
    shots_num = str(shots_num)

    # 保存结果为 CSV 文件
    file_name_csv = f"iteration_info_eps_{molecule}_{prob_1}_{prob_2}_{d}_{shots_num}.csv"
    iteration_info_df.to_csv(file_name_csv, index=True)
    print("CSV文件已创建:", file_name_csv)

    # 保存结果为 pkl 文件
    file_path = f"iteration_info_eps_{molecule}_{prob_1}_{prob_2}_{d}_{shots_num}.pkl"  # 为pkl文件定义路径
    with open(file_path, 'wb') as file:
        pickle.dump(iteration_info_df, file)
    print("文件已创建:", file_path)

    return iteration_info_df['current_eps'].iloc[0], iteration_info_df['current_eps'].iloc[-1]

def process_single_case(molecule, prob_1, prob_2, d, shots_num, accuracy):
    
    print(f"Now finding optimal eps, parameters are : {prob_1},{prob_2},{d},{shots_num}")

    result_data = {}
    
    d_str = f"{d:.1f}" 
    current_dir = os.getcwd() # 获取当前工作目录的字符串 (e.g., '/home/user/project')
    input_filename = f"{molecule}_{d}_sto-3g.pkl"
    full_input_path = os.path.join(current_dir, input_filename) # 安全地拼接路径
    
    # 3. 在 open() 中使用构建好的完整路径
    print(f"Attempting to load file: {full_input_path}")
    try:
        with open(full_input_path, 'rb') as file:
            H2_d = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found. Skipping this distance.")
        print(f"Searched for file at: {full_input_path}")
        return f"Failed for d={d:.2f}"

    try:
        with open(os.path.join(current_dir, f'{molecule}_d{d_str}_one_rdm_list.pkl'), 'rb') as f:
            loaded_one_rdm_list = pickle.load(f)
        with open(os.path.join(current_dir, f'{molecule}_d{d_str}_two_rdm_list.pkl'), 'rb') as f:
            loaded_two_rdm_list = pickle.load(f)
         
        two_body_coefficients = H2_d['two_body_coefficients']
        one_body_coefficients = H2_d['one_body_coefficients']
        
        constant = H2_d['constant']
        FCI_val = H2_d['FCI_val']
        num_particles = H2_d['n_particles']
        Precise_diagonalization_energy = H2_d['Precise_diagonalization_energy']
        two_D_no_noisy = loaded_two_rdm_list[2]
        one_D_no_noisy = loaded_one_rdm_list[2]
        two_D_noisy = loaded_two_rdm_list[3]
        one_D_noisy = loaded_one_rdm_list[3]
        energy_no_noisy = np.sum(one_body_coefficients * one_D_no_noisy) + 0.5 * np.sum(two_body_coefficients * two_D_no_noisy) + constant
        energy_noisy = np.sum(one_body_coefficients * one_D_noisy) + 0.5 * np.sum(two_body_coefficients * two_D_noisy) + constant
        

        print("++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++")
        
        two_body_coefficients = np.einsum('ijkl->ijlk', H2_d['two_body_coefficients'])
        experimental_2rdm_list = []
        # experimental_2rdm_list.append(loaded_two_rdm_list[3])
        experimental_2rdm_list.append(np.einsum('ijlk->ijkl', loaded_two_rdm_list[3]))
        
        print(experimental_2rdm_list[0].ndim)
        
        print("++++++++++++++++++++++++++++")

        eps_max = 10000
        
        E_thm_max, two_rdm_res_ijkl_max, one_rdm_res_ij_max, G_ijkl_max, Q_ijkl_max = legalize_then_optimize(experimental_2rdm_list[0], num_particles, one_body_coefficients, two_body_coefficients, accuracy=accuracy, epsilon = eps_max)
        print("++++++++++++++++++++++++++++") 
        print(E_thm_max)
        
        
        Convert_two_rdm = np.einsum('ijlk->ijkl', loaded_two_rdm_list[0])
        # print(two_rdm_res_ijkl_max)
        difference = Convert_two_rdm - two_rdm_res_ijkl_max

        # 计算差的 Frobenius 范数
        norm_difference = np.linalg.norm(difference)

        print("Norm of the difference:", norm_difference)
        
        
        energy = np.sum(one_body_coefficients * one_rdm_res_ij_max) + 0.5 * np.sum(two_body_coefficients * two_rdm_res_ijkl_max) + constant


        result_data['one_rdm_res_ij_max'] = one_rdm_res_ij_max
        result_data['two_rdm_res_ijkl_max'] = two_rdm_res_ijkl_max
        result_data['E_thm_max'] = E_thm_max + constant  
        
        
        print(energy)
        
        print(result_data['E_thm_max'])
        
        print("++++++++++++++++++++++++++++")
        
        current_eps_init, current_eps = find_optimal_eps(molecule, prob_1, prob_2, d, shots_num, 
            Precise_diagonalization_energy,
            constant,
            num_particles,
            accuracy,
            one_body_coefficients,
            two_body_coefficients,
            experimental_2rdm_list[0]
        )
        
        # 更新 result_data 字典
        result_data['leg_eps'] = current_eps_init
        result_data['eps_optimal'] = current_eps
        print("++++++++++++++++++++++++++++")
        
        print("--- Loading delta value from file ---")
        try:
            with open(os.path.join(current_dir, f'{molecule}_d{d_str}_df_2_rdm.pkl'), 'rb') as f:
                loaded_df_2_rdm = pickle.load(f)
            
            delta = loaded_df_2_rdm["CIRC"]['WGF']
            print(f"Successfully loaded delta = {delta:.6f}")
            print(f"The threshold value current_eps_init = {current_eps_init:.6f}")

            # 2. 初始化循环变量
            k = 1  # delta 的倍数，从1开始
            cdr_results_list = [] # 创建一个列表来存储每次循环的结果

            print("\n--- Searching for the smallest k and running one calculation ---")

            while True: # 启动一个无限循环
                # 检查是否满足条件
                if (k * delta) > current_eps:
                # 条件满足，这就是我们要找的 k
                    print(f"\nCondition met at k={k}:")
                    print(f"(k * delta) = {k * delta:.6f} > {current_eps:.6f} (True)")
        
                    eps_CDR = k * delta
                    print(f"Setting epsilon for legalize_then_optimize to {eps_CDR:.6f}")

                    # 运行核心计算函数
                    E_thm_CDR, two_rdm_res_ijkl_CDR, one_rdm_res_ij_CDR, G_ijkl_CDR, Q_ijkl_CDR = legalize_then_optimize(
                    experimental_2rdm_list[0], 
                    num_particles, 
                    one_body_coefficients, 
                    two_body_coefficients, 
                    accuracy=accuracy, 
                    epsilon=eps_CDR
                        )
        
                    # 保存结果
                    final_result = {
                        'k': k,
                        'eps_CDR': eps_CDR,
                        'E_thm_CDR': E_thm_CDR + constant,
                        'one_rdm_res_ij_CDR': one_rdm_res_ij_CDR,
                        'two_rdm_res_ijkl_CDR': two_rdm_res_ijkl_CDR
                    }
                    cdr_results_list.append(final_result)
        
                    print(f"Calculation finished. E_thm_CDR = {final_result['E_thm_CDR']:.8f}")
        
                    # 完成计算后，跳出循环
                    break
        
                else:
                # 条件不满足，继续寻找
                    print(f"Searching... k={k}, (k * delta) = {k * delta:.6f} <= {current_eps:.6f}")
                    k += 1 # k自增，准备下一次判断

            # 循环结束后的最终检查
            print(f"\nLoop finished. Final check for k={k}:")
            print(f"Condition: (k * delta) = {k * delta:.6f} < {current_eps_init:.6f} (False)")

            # 4. 将存储了所有循环结果的列表存入主 result_data 字典
            result_data['cdr_iterations'] = cdr_results_list
            
        except FileNotFoundError:
            print(f"Error: Could not find the file to load delta. Skipping the CDR loop.")
        except KeyError:
            print(f"Error: Keys 'CIRC' or 'THM' not found in the loaded data. Skipping the CDR loop.")

        print("++++++++++++++++++++++++++++") 
        # 初始化 E_thm_CF
  
        return result_data
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    return None
if __name__ == "__main__":
    current_time_start = datetime.datetime.now()
    
    print(f"Now starting the execution of VQE + VRDM, time is: {current_time_start}")
    
    # 固定参数
    molecule = 'H2'
    basis= "sto-3g"
    d_list = []
    current_value = 0.4

    while current_value <= 1.2:
        d_list.append(round(current_value, 1))
        current_value += 0.1

    while current_value <= 3.5:
        d_list.append(round(current_value, 1))
        current_value += 0.3

    print(d_list)
    
    for d in d_list:
        prob_1 = 0.001
        prob_2 = 0.01
    
        shots_num = 2 ** 15
        num_particles = 4  
        accuracy = 1e-7

        # 调用处理函数
        data_result = process_single_case(molecule, prob_1, prob_2, d, shots_num, accuracy)
        
        filename = f"data_result_{molecule}_{d}_{basis}.pkl"

        # 使用pickle保存字典
        with open(filename, 'wb') as file:
            pickle.dump(data_result, file)

    print(f"All data are saved as {filename}")
    current_time_end = datetime.datetime.now()
    print(f"Now ending VRDM, Total time is: {current_time_end - current_time_start}")