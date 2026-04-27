'''

Created Date: Tue Nov 15 2023

Revised Date: Tue Aug 26 2024

Revised Date: Sat Jul 21 2025

Author: Qiming Ding, Yukun Zhang

Description: H4 Dissociation  New qiskit 1.2 Qiskit Algorithms 0.3.0 Qiskit Aer 0.15.0 只计算替换门的方法


Copyright (c) 2025

'''

# Standard library imports
import time
import os
# Scientific computing and data handling libraries
import numpy as np
import pandas as pd
import datetime
import pickle
import itertools
from itertools import combinations, product
import functools
from functools import reduce
from joblib import Memory, Parallel, delayed
from typing import Union, List, Optional, Tuple, Any

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.quantum_info import Clifford
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, SGate


from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit_aer import AerSimulator, AerError
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.primitives import EstimatorV2 
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import L_BFGS_B,COBYLA
from qiskit_algorithms import VQE
from qiskit_algorithms.gradients import FiniteDiffEstimatorGradient,ParamShiftEstimatorGradient

from qiskit.exceptions import QiskitError

import openfermion as of

import mitiq
from mitiq import cdr, Observable, PauliString
from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
from mitiq.interface.mitiq_qiskit import qiskit_utils

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

def to_density_matrix(ground_state_wavefunction):
    """
    Convert a given quantum state (either a state vector or a density matrix) 
    into its density matrix representation.

    Parameters:
        ground_state_wavefunction (numpy.ndarray): The quantum state to be converted.
            It can be provided as a state vector (1D array) or already as a density 
            matrix (2D square matrix).

    Returns:
        numpy.ndarray: The density matrix representation of the input state.
            If the input is a state vector, it returns the corresponding density matrix.
            If the input is already a density matrix, it returns it unchanged.

    Raises:
        ValueError: If the input is not a valid quantum state, i.e., it's not a 
            1D vector (state vector) or a 2D square matrix (density matrix).
    """
    # Check if the input is a 1D array (pure state vector)
    if ground_state_wavefunction.ndim == 1:
        # If it's a state vector, convert it to a density matrix using the outer product
        # ρ = |ψ⟩⟨ψ| where ψ is the state vector
        density_matrix = np.outer(ground_state_wavefunction, np.conj(ground_state_wavefunction))
        return density_matrix
    
    # Check if the input is already a 2D array (potential density matrix)
    elif ground_state_wavefunction.ndim == 2:
        # Ensure it's a square matrix (valid density matrix must be square)
        if ground_state_wavefunction.shape[0] != ground_state_wavefunction.shape[1]:
            raise ValueError("Input must be a square matrix to represent a valid density matrix.")
        # If it's a valid square matrix, return it directly
        return ground_state_wavefunction
    
    else:
        # Raise an error if the input is neither a 1D vector nor a 2D square matrix
        raise ValueError("Input must be either a 1D state vector or a 2D square matrix.")
    
def calculate_frobenius_norm_difference(theoretical_data, experimental_data):
    """
    Calculate the Frobenius norm difference between theoretical and experimental data.
    This function computes the Frobenius norm, which measures the overall difference 
    between two arrays (matrices or tensors), as well as returning the absolute 
    element-wise differences between the input arrays.

    Parameters:
        theoretical_data (numpy.ndarray): Theoretical data represented as a Numpy array, 
                                          can be a matrix (2D) or tensor (4D).
        experimental_data (numpy.ndarray): Experimental data represented as a Numpy array, 
                                           must have the same shape as theoretical_data.

    Returns:
        float: The Frobenius norm difference, representing the overall magnitude of the 
               difference between the two input arrays.
        numpy.ndarray: A tensor of the absolute element-wise differences between 
                      theoretical_data and experimental_data.

    Raises:
        ValueError: If the input arrays do not have the same shape or the dimensionality
                    is neither 2D nor 4D.
    """
    # Ensure both input arrays have the same shape
    if theoretical_data.shape != experimental_data.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Handle 2D (matrix) or 4D (tensor) input arrays
    if theoretical_data.ndim == 2:
        # If the input is a 2D matrix, calculate the Frobenius norm directly using np.linalg.norm
        norm_difference = np.linalg.norm(theoretical_data - experimental_data, 'fro')
    elif theoretical_data.ndim == 4:
        # If the input is a 4D tensor, manually calculate the Frobenius norm

        # Step 1: Compute the difference between theoretical and experimental tensors
        difference = theoretical_data - experimental_data
        
        # Step 2: Square each element of the difference tensor
        squared_difference = np.square(difference)
        
        # Step 3: Sum all the squared elements to get the sum of squares
        sum_of_squares = np.sum(squared_difference)
        
        # Step 4: Take the square root of the sum of squares to obtain the Frobenius norm
        norm_difference = np.sqrt(sum_of_squares)

        # Print intermediate results for comparison of different methods (commented out)
        # Uncomment if you want to compare various methods of calculating the Frobenius norm
        '''
        # Method 2: Directly using numpy's built-in norm function
        norm_difference2 = np.linalg.norm(theoretical_data - experimental_data)
        print("Method 2 - numpy.linalg.norm on difference:", norm_difference2)

        # Method 3: Using numpy.einsum to efficiently compute the Frobenius norm
        norm_difference3 = np.sqrt(np.einsum('ijkl,ijkl', difference, difference))
        print("Method 3 - numpy.einsum:", norm_difference3)

        # Method 4: Flatten the tensor and calculate the norm on the flattened array
        norm_difference4 = np.linalg.norm(difference.ravel())
        print("Method 4 - Flattening and numpy.linalg.norm:", norm_difference4)
        '''
    else:
        # Raise an error if the input is neither a 2D matrix nor a 4D tensor
        raise ValueError("Input arrays must be either two-dimensional (matrix) or four-dimensional (tensor).")

    # Calculate the absolute element-wise difference tensor
    # This represents the magnitude of the difference at each corresponding element
    element_wise_difference = np.abs(theoretical_data - experimental_data)

    # Return the Frobenius norm and the element-wise difference tensor
    return norm_difference.real, element_wise_difference


def noisy_model_set(prob_1, prob_2):
    """
    Create a noise model with depolarizing errors for 1-qubit and 2-qubit gates.
    If both prob_1 and prob_2 are zero, the noise model is set to None.

    Parameters:
        prob_1 (float): Depolarizing error probability for 1-qubit gates.
        prob_2 (float): Depolarizing error probability for 2-qubit gates.

    Returns:
        NoiseModel or None: The noise model with the specified depolarizing errors.
    """
    if prob_1 == 0 and prob_2 == 0:
        noise_model = None
    else:
        error_1 = depolarizing_error(prob_1, 1)
        error_2 = depolarizing_error(prob_2, 2)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error_1, 
            ['u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 't', 'tdg']
        )    
        noise_model.add_all_qubit_quantum_error(
            error_2, 
            ['swap', 'cx', 'cy', 'cz', 'csx', 'cp', 'cu', 'cu1', 'cu2', 'cu3', 'rxx', 'ryy', 'rzz', 'rzx', 'ecr']
        )
    return noise_model

def build_noisy_estimator(prob_1, prob_2, shots_num, device="CPU"):
    """
    Build a noisy quantum estimator using the specified noise model, and attempt
    to use GPU if the 'device' parameter is set to 'GPU'. If GPU initialization fails,
    fallback to CPU.

    Parameters:
        prob_1 (float): Depolarizing error probability for 1-qubit gates.
        prob_2 (float): Depolarizing error probability for 2-qubit gates.
        shots_num (int): Number of shots to use for simulation.
        device (str): 'GPU' or 'CPU'. If 'GPU' is selected, the code will attempt to 
                      initialize the estimator on GPU.

    Returns:
        Tuple: (noise_model, noisy_estimator)
    """
    noise_model = None
    noisy_estimator = None

    if prob_1 != 0 or prob_2 != 0:
        # Create the noise model with the specified error probabilities
        noise_model = noisy_model_set(prob_1, prob_2)
        try:
            # Attempt to initialize the AerEstimator with GPU if specified
            if device.upper() == "GPU":
                print("Attempting to initialize GPU for noisy estimator...")
                noisy_estimator = AerEstimator(
                    backend_options={"method": "density_matrix", "device": "GPU", "noise_model": noise_model},
                    run_options={"shots": shots_num}
                )
            else:
                # Use CPU by default if no GPU is requested
                print("Using CPU for noisy estimator...")
                noisy_estimator = AerEstimator(
                    backend_options={"method": "density_matrix", "noise_model": noise_model},
                    run_options={"shots": shots_num}
                )
        except AerError as e:
            # If GPU initialization fails, fallback to CPU and log the error
            print("Failed to initialize GPU estimator, falling back to CPU:", str(e))
            noisy_estimator = AerEstimator(
                backend_options={"method": "density_matrix", "noise_model": noise_model},
                run_options={"shots": shots_num}
            )

    return noise_model, noisy_estimator

def build_noisy_estimatorV2(prob_1, prob_2, shots_num, device="CPU"):
    """
    Build a noisy quantum estimator using EstimatorV2, and attempt to use GPU 
    if the 'device' parameter is set to 'GPU'. If GPU initialization fails, fallback to CPU.

    Parameters:
        prob_1 (float): Depolarizing error probability for 1-qubit gates.
        prob_2 (float): Depolarizing error probability for 2-qubit gates.
        shots_num (int): Number of shots to use for simulation.
        device (str): 'GPU' or 'CPU'. If 'GPU' is selected, the code will attempt to 
                      initialize the estimator on GPU.

    Returns:
        Tuple: (noise_model, noisy_estimator)
    """
    if prob_1 != 0 or prob_2 != 0:
        # Create the noise model with the specified error probabilities
        noise_model = noisy_model_set(prob_1, prob_2)

        # Set the backend device (GPU or CPU)
        method = "density_matrix"
        backend_device = "GPU" if device.upper() == "GPU" else "CPU"

        # Initialize the simulator with noise model
        try:
            if backend_device == "GPU":
                print("Attempting to initialize GPU for noisy estimator...")
                simulator = AerSimulator(method=method, device="GPU", noise_model=noise_model)
            else:
                print("Using CPU for noisy estimator...")
                simulator = AerSimulator(method=method, noise_model=noise_model)

            # Create EstimatorV2 using the simulator as the backend
            noisy_estimatorV2 = EstimatorV2.from_backend(
                backend=simulator,
                options={
                    "backend_options": {"noise_model": noise_model},
                    "run_options": {"shots": shots_num},
                    "default_precision": 0.01  # Adjust precision as needed
                }
            )
        except AerError as e:
            # If GPU initialization fails, fallback to CPU and log the error
            print("Failed to initialize GPU estimator, falling back to CPU:", str(e))
            simulator = AerSimulator(method=method, noise_model=noise_model)
            noisy_estimatorV2 = EstimatorV2.from_backend(
                backend=simulator,
                options={
                    "backend_options": {"noise_model": noise_model},
                    "run_options": {"shots": shots_num},
                    "default_precision": 0.01
                }
            )

    return noisy_estimatorV2

@measure_execution_time
def Gene_Qiskit_VQE_hamiltonian(n_qubits, qubit_hamiltonian):
    """
    Generate a Qiskit-compatible Hamiltonian for VQE (Variational Quantum Eigensolver)
    from a given qubit Hamiltonian.

    Parameters:
        n_qubits (int): The number of qubits in the system.
        qubit_hamiltonian (QubitOperator): The qubit Hamiltonian in terms of Pauli operators.
    
    Returns:
        SparsePauliOp: A Qiskit SparsePauliOp object representing the Hamiltonian.

    Description:
        The function converts a Hamiltonian, represented as a dictionary of Pauli terms 
        and coefficients, into a Qiskit SparsePauliOp, which can be used for quantum 
        simulations and optimizations like VQE. The qubit Hamiltonian is expected to 
        have terms of Pauli strings (e.g., X, Y, Z), and each term is translated into 
        a corresponding Pauli string operator in Qiskit.

    Example:
        A Hamiltonian term like 'X0 Y1 Z2' is converted into a SparsePauliOp representation.

    """
    paulis = []  # List to store the Pauli strings for each term
    coeffs = []  # List to store the coefficients for each Pauli string

    # Loop through each term in the qubit Hamiltonian (terms are Pauli operators and coefficients)
    for term, coeff in qubit_hamiltonian.terms.items():
        modes = [0] * n_qubits  # Initialize a list to store Pauli modes for each qubit (I, X, Y, Z)

        # For each Pauli operator in the term, assign its corresponding integer mode
        for qubit, op in term:
            if op == "X":
                modes[qubit] = 1  # X Pauli operator
            elif op == "Y":
                modes[qubit] = 2  # Y Pauli operator
            elif op == "Z":
                modes[qubit] = 3  # Z Pauli operator

        # Construct the Pauli string from the list of modes, in reversed order for correct indexing
        pauli_str = "".join(["I", "X", "Y", "Z"][mode] for mode in reversed(modes))

        # Append the Pauli string and the corresponding coefficient to their respective lists
        paulis.append(pauli_str)
        coeffs.append(coeff)

    # Combine Pauli strings and their coefficients into a list of tuples
    pauli_op = [(pauli, weight) for pauli, weight in zip(paulis, coeffs)]

    # Convert the list of Pauli terms into a SparsePauliOp (Qiskit representation)
    hamiltonian_qiskit = SparsePauliOp.from_list(pauli_op)

    # Print the number of qubits for reference/debugging
    print(f"Number of qubits: {hamiltonian_qiskit.num_qubits}")

    return hamiltonian_qiskit  # Return the constructed Qiskit Hamiltonian

def run_vqe(hamiltonian_qiskit, ansatz, optimizer, estimator):
    """
    Generic function to run the Variational Quantum Eigensolver (VQE) algorithm.

    Parameters:
    -----------
    hamiltonian_qiskit : OperatorBase
        The Hamiltonian of the system in Qiskit's format (e.g., SparsePauliOp).
    
    ansatz : QuantumCircuit
        The ansatz quantum circuit used for the VQE algorithm.
    
    optimizer : Optimizer
        The classical optimizer used to minimize the expectation value of the Hamiltonian.
    
    estimator : Estimator
        The estimator object used to calculate the expectation values of the Hamiltonian 
        for given parameters (supports both noiseless and noisy estimators).

    Returns:
    --------
    result : VQEResult
        The result object containing details of the VQE optimization process (e.g., optimal energy).
    
    parameters_list : list
        A list of parameter values collected during the VQE optimization process.
    
    circ : QuantumCircuit
        The optimal circuit after the VQE optimization with parameters assigned.
    
    estimator : Estimator
        The estimator used in the VQE, returned for further reuse or analysis.
    
    optimal_parameters : ndarray
        The optimal parameters obtained from the VQE optimization process.
    """
    # Store intermediate results such as iteration count, parameter values, and energy values
    counts = []
    values = []
    parameters_list = []
    std_list = []

    # Callback function to store intermediate VQE results
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        parameters_list.append(parameters)
        std_list.append(std)
        # Uncomment the following line for more detailed output during optimization
        # print(f"Iteration {eval_count}: Energy = {mean:.8f}, Parameters = {parameters}")

    # Using the ParamShiftEstimatorGradient for gradient calculation (parameter shift rule)
    gradient = ParamShiftEstimatorGradient(estimator)
    
    # Initialize the VQE algorithm with estimator, ansatz, optimizer, and gradient
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, gradient=gradient, callback=store_intermediate_result)
    
    # Set the initial parameters for the ansatz (either zeros or random values)
    vqe.initial_point = np.zeros(ansatz.num_parameters)
    # Alternatively, you could use random initial points:
    # vqe.initial_point = np.random.uniform(low=-np.pi, high=np.pi, size=ansatz.num_parameters)

    # Run the VQE algorithm to find the minimum eigenvalue of the Hamiltonian
    result = vqe.compute_minimum_eigenvalue(hamiltonian_qiskit)

    # Get the optimal quantum circuit with the best-found parameters
    circ = result.optimal_circuit.assign_parameters(result.optimal_parameters)
    optimal_parameters = result.optimal_parameters

    # Print the final optimal energy found by the VQE
    print(f"VQE Optimal Energy: {result.optimal_value.real:.8f}")

    return result, parameters_list, circ, estimator, optimal_parameters

def create_ansatz(num_qubits, n_particles):
    """
    Create the Hartree-Fock initial state and UCCSD (Unitary Coupled Cluster with Single and Double excitations) ansatz.

    Parameters:
    -----------
    num_qubits : int
        The number of qubits in the quantum system, which corresponds to the number of orbitals in the problem.
    
    n_particles : int
        The total number of particles (electrons) in the quantum system.

    Returns:
    --------
    ansatz : UCCSD
        The UCCSD ansatz initialized with the Hartree-Fock state as the reference state.
    """
    # Initialize the qubit mapper (Jordan-Wigner mapping) for mapping fermions to qubits
    mapper = JordanWignerMapper()
    
    # Number of particles split between spin-up and spin-down electrons
    num_particles = [n_particles // 2, n_particles // 2]
    
    # Create the Hartree-Fock initial state as the starting point for the ansatz
    hf = HartreeFock(num_spatial_orbitals=num_qubits // 2, 
                     num_particles=num_particles, 
                     qubit_mapper=mapper)
    
    # Define the UCCSD ansatz with the Hartree-Fock state as the reference
    ansatz = UCCSD(num_spatial_orbitals=num_qubits // 2, 
                   num_particles=num_particles, 
                   qubit_mapper=mapper,
                   initial_state=hf, 
                   generalized=False,  # Set to True if you want to use generalized excitations
                   preserve_spin=True)  # Preserve the total spin symmetry in the system

    return ansatz

def UCCSD_VQE_noiseless(hamiltonian_qiskit, num_qubits, n_particles, max_iterations=1000):
    """
    Runs noiseless UCCSD VQE.

    Parameters:
    -----------
    hamiltonian_qiskit : OperatorBase
        The Hamiltonian of the system in Qiskit's format.
    
    num_qubits : int
        The number of qubits in the system.
    
    n_particles : int
        The number of particles in the system.
    
    max_iterations : int, optional
        Maximum number of iterations for the optimizer (default is 1000).

    Returns:
    --------
    result : VQEResult
        The result object from the VQE algorithm.
    """
    # Create the ansatz (UCCSD with Hartree-Fock initial state)
    ansatz = create_ansatz(num_qubits, n_particles)
    
    # Define the classical optimizer (COBYLA in this case)
    optimizer = COBYLA(maxiter=max_iterations)
    
    # Use the default noiseless estimator
    estimator = Estimator()
    # estimator = StatevectorEstimator()

    # Run the VQE algorithm using the provided Hamiltonian, ansatz, optimizer, and estimator
    return run_vqe(hamiltonian_qiskit, ansatz, optimizer, estimator)

def UCCSD_VQE_noisy(hamiltonian_qiskit, num_qubits, n_particles, noisy_estimator, max_iterations=1000):
    """
    Runs noisy UCCSD VQE with the specified noise model.

    Parameters:
    -----------
    hamiltonian_qiskit : OperatorBase
        The Hamiltonian of the system in Qiskit's format.
    
    num_qubits : int
        The number of qubits in the system.
    
    n_particles : int
        The number of particles in the system.
    
    noisy_estimator : Estimator
        A noisy estimator created externally and passed into the function.
    
    max_iterations : int, optional
        Maximum number of iterations for the optimizer (default is 1000).

    Returns:
    --------
    result : VQEResult
        The result object from the noisy VQE algorithm.
    """
    # Create the ansatz (UCCSD with Hartree-Fock initial state)
    ansatz = create_ansatz(num_qubits, n_particles)
    
    # Define the classical optimizer (COBYLA in this case)
    optimizer = COBYLA(maxiter=max_iterations)

    # Run the VQE algorithm using the provided Hamiltonian, ansatz, optimizer, and noisy estimator
    return run_vqe(hamiltonian_qiskit, ansatz, optimizer, noisy_estimator)

def get_one_rdm_reduced_indices(num_qubits: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(num_qubits) for j in range(i, num_qubits)]

def get_one_rdm_term(i, j, num_qubits, circ, estimator):
    """
    Calculate a specific element of the one-body reduced density matrix (1-RDM)
    for the given indices in a quantum system.

    Parameters:
    -----------
    i, j : int
        Indices for the 1-RDM element.
    
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit
        The quantum circuit used to prepare the quantum state for measurement.
    
    estimator : Estimator
        The estimator object (either noiseless or noisy) used to compute expectation values.

    Returns:
    --------
    tuple : (i, j, float)
        A tuple containing the indices (i, j) and the calculated value of the 1-RDM element.
    """
    # Check if the indices are out of bounds for the system size
    if i >= num_qubits or j >= num_qubits:
        # Return 0 if indices are invalid
        return i, j, 0.0
    
    # Create the fermionic operator for the 1-RDM element (creation at i, annihilation at j)
    fermi_term = FermionicOp({f"+_{i} -_{j}": 1}, num_spin_orbitals=num_qubits)
    
    # Map the fermionic operator to a qubit operator using the Jordan-Wigner transformation
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Initialize the total expectation value
    total_expectation_value = 0.0
    
    # Loop over Pauli terms in the mapped qubit operator
    for pauli_op, coeff in qubit_term.label_iter():
        # Create a SparsePauliOp for each Pauli term
        single_pauli_term = SparsePauliOp(pauli_op)
        
        # Use the estimator to compute the expectation value of the Pauli term
        result = estimator.run(circuits=[circ], observables=[single_pauli_term]).result()
        
        # Multiply the expectation value by the coefficient and accumulate it
        expectation_value = np.real(coeff * result.values[0])
        total_expectation_value += expectation_value

    # Output the calculated 1-RDM element for debugging
    print(f"Calculated 1-RDM element ({i}, {j}) is {total_expectation_value}")
    
    # Return the indices and the calculated value
    return i, j, total_expectation_value

@measure_execution_time
def get_one_rdm(num_qubits, circ=None, estimator=None):
    """
    Calculate the full one-body reduced density matrix (1-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.
    
    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    one_rdm : np.ndarray
        A 2D numpy array representing the one-body reduced density matrix.
    """
    # Ensure that an estimator is provided for the calculation
    if estimator is None:
        raise ValueError("Estimator must be provided for 1-RDM calculation.")
        
    # Initialize an empty matrix for the 1-RDM
    one_rdm = np.zeros((num_qubits, num_qubits), dtype=complex)
    
    # Partial application of the function to fix the constant parameters
    func = functools.partial(get_one_rdm_term, num_qubits=num_qubits, circ=circ, estimator=estimator)
    
    # Generate all possible index combinations (i, j) for the 1-RDM
    indices = [(i, j) for i in range(num_qubits) for j in range(num_qubits)]
    
    # Parallel computation of all 1-RDM elements (set n_jobs according to your system)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j) for i, j in indices)

    # Fill the 1-RDM matrix with the computed results
    for result in results:
        i, j, temp = result
        one_rdm[i, j] = temp
        one_rdm[j, i] = temp
    return one_rdm

def is_invalid_index(i: int, j: int, k: int, l: int, num_qubits: int) -> bool:
    """
    Check if the given indices are invalid based on the number of spin orbitals.

    Args:
        i: The first index.
        j: The second index.
        k: The third index.
        l: The fourth index.
        num_qubits: The total number of spin orbitals.

    Returns:
        True if the indices are invalid, False otherwise.
    """
    valid_term = (
        (i < num_qubits // 2 and j < num_qubits // 2 and k < num_qubits // 2 and l < num_qubits // 2) or
        (i < num_qubits // 2 and j >= num_qubits // 2 and k < num_qubits // 2 and l >= num_qubits // 2) or
        (i < num_qubits // 2 and j >= num_qubits // 2 and k >= num_qubits // 2 and l < num_qubits // 2) or
        (i >= num_qubits // 2 and j < num_qubits // 2 and k < num_qubits // 2 and l >= num_qubits // 2) or
        (i >= num_qubits // 2 and j < num_qubits // 2 and k >= num_qubits // 2 and l < num_qubits // 2) or
        (i >= num_qubits // 2 and j >= num_qubits // 2 and k >= num_qubits // 2 and l >= num_qubits // 2)
    )
    return not valid_term

def get_two_rdm_reduced_indices(num_qubits: int) -> List[Tuple[int, int, int, int]]:
    unique_indices = []
    for i, j, k, l in itertools.product(range(num_qubits), repeat=4):
        if (i, j, k, l) not in unique_indices and (k, l, i, j) not in unique_indices and \
           (j, i, k, l) not in unique_indices and (i, j, l, k) not in unique_indices:
            unique_indices.append((i, j, k, l))
    return unique_indices

def get_total_two_rdm_terms(num_qubits: int) -> int:
    """
    Calculate the total number of RDM terms to be computed.

    Args:
        num_qubits: The total number of spin orbitals.
        num_particles: The total number of particles.

    Returns:
        The total number of RDM terms.
    """
    unique_indices = get_two_rdm_reduced_indices(num_qubits)
    
    valid_terms = 0

    for i, j, k, l in unique_indices:
        if not is_invalid_index(i, j, k, l, num_qubits):
            valid_terms += 1

    return valid_terms

def get_two_rdm_term(i, j, k, l, num_qubits, circ, estimator):
    """
    Calculate a specific element of the two-body reduced density matrix (2-RDM) 
    for given indices in a quantum system.

    Parameters:
    -----------
    i, j, k, l : int
        Indices for the 2-RDM element.
    
    num_qubits : int
        The number of qubits (spin orbitals) in the system.
    
    circ : QuantumCircuit
        The quantum circuit used to prepare the quantum state for measurement.
    
    estimator : Estimator
        The estimator object (either noiseless or noisy) used to compute expectation values.

    Returns:
    --------
    tuple : (i, j, k, l, float)
        A tuple containing the indices (i, j, k, l) and the calculated value of the 2-RDM element.
    """
    # Check if the given indices are invalid for the current system size
    if is_invalid_index(i, j, k, l, num_qubits):
        # If invalid, return the indices and a value of 0.0
        return i, j, k, l, 0.0
    
    # Construct the fermionic operator for the 2-RDM element
    # Fermionic operator format: f"+_{i} +_{j} -_{k} -_{l}"
    fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

    # Use Jordan-Wigner mapping to convert the fermionic operator to a qubit operator
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Initialize total expectation value to accumulate the results
    total_expectation_value = 0.0

    # Loop over Pauli terms in the qubit operator
    for pauli_op, coeff in sorted(qubit_term.label_iter()):
        # Create a SparsePauliOp for each individual Pauli term
        single_pauli_term = SparsePauliOp(pauli_op, coeffs=1)
        
        # Use the estimator to compute the expectation value of the Pauli term
        result = estimator.run(circuits=[circ], observables=[single_pauli_term]).result()
        
        # Multiply the expectation value by the coefficient and accumulate it
        expectation_value = np.real(coeff * result.values[0])
        total_expectation_value += expectation_value

    # Store the final calculated value of the 2-RDM element
    temp = total_expectation_value
    
    # Output the calculated 2-RDM element for debugging purposes
    print(f"Calculated 2-RDM element ({i}, {j}, {k}, {l}) is {temp}")
    
    # Return the indices and the calculated 2-RDM element
    return i, j, k, l, temp

@measure_execution_time
def get_two_rdm(num_qubits, circ=None, estimator=None):
    """
    Calculate the full two-body reduced density matrix (2-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.
    
    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure an estimator is provided, as it's necessary for the calculation
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")
    
    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)
    unique_indices = get_two_rdm_reduced_indices(num_qubits)
    
    # Use functools.partial to fix the constant parameters (num_qubits, circ, estimator)
    func = functools.partial(get_two_rdm_term, num_qubits=num_qubits, circ=circ, estimator=estimator)

    # Use joblib's Parallel to compute 2-RDM elements in parallel (can adjust n_jobs for parallelism)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j, k, l) for i, j, k, l in unique_indices)

    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Fill the 2-RDM matrix with the calculated values, including symmetries
    for result in results:
        i, j, k, l, temp = result
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm

def generate_pauli_terms(unique_indices, num_qubits):
    """
    生成所有需要的 Pauli 算符及其系数，并建立 RDM 元素与 Pauli 项的映射。

    Parameters:
    -----------
    unique_indices : list of tuples
        需要计算的 RDM 索引组合。

    num_qubits : int
        系统中的量子比特数量。

    Returns:
    --------
    pauli_dict : dict
        Pauli 算符及其累积系数的字典。

    rdm_pauli_terms : dict
        RDM 元素与对应的 Pauli 项及系数的映射。

    """
    pauli_dict = {}
    rdm_pauli_terms = {}
    mapper = JordanWignerMapper()

    for idx in unique_indices:
        i, j, k, l = idx

        # 检查索引的有效性
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # 构建费米子算符
        fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

        # 映射到量子比特算符
        qubit_term = mapper.map(fermi_term)

        # 遍历量子比特算符的 Pauli 项
        for pauli_op, coeff in qubit_term.to_list():
            # 更新 Pauli 项的系数
            if pauli_op in pauli_dict:
                pauli_dict[pauli_op] += coeff
            else:
                pauli_dict[pauli_op] = coeff

            # 记录 RDM 元素与 Pauli 项的关系
            if idx in rdm_pauli_terms:
                rdm_pauli_terms[idx].append((pauli_op, coeff))
            else:
                rdm_pauli_terms[idx] = [(pauli_op, coeff)]

    return pauli_dict, rdm_pauli_terms

def compute_expectation_values(pauli_dict, circ, estimator):
    """
    并行计算所有唯一 Pauli 算符的期望值。

    Parameters:
    -----------
    pauli_dict : dict
        Pauli 算符及其累积系数的字典。

    circ : QuantumCircuit
        用于制备量子态的量子电路。

    estimator : Estimator
        用于计算期望值的估计器对象。

    Returns:
    --------
    pauli_expectation_dict : dict
        Pauli 算符及其期望值的字典。

    """
    # 提取所有唯一的 Pauli 算符
    unique_pauli_ops = list(pauli_dict.keys())

    # 将 Pauli 算符转换为 SparsePauliOp 对象
    sparse_pauli_ops = [SparsePauliOp(pauli_op) for pauli_op in unique_pauli_ops]

    # 定义计算单个 Pauli 项期望值的函数
    def compute_expectation(pauli_op):
        result = estimator.run(circuits=[circ], observables=[pauli_op]).result()
        return result.values[0]

    # 并行计算期望值
    expectation_values = Parallel(n_jobs=-1)(
        delayed(compute_expectation)(pauli_op) for pauli_op in sparse_pauli_ops
    )

    # 构建 Pauli 算符与期望值的映射
    pauli_expectation_dict = {
        pauli_op: value for pauli_op, value in zip(unique_pauli_ops, expectation_values)
    }

    return pauli_expectation_dict

def get_two_rdm_term_Parallel(i, j, k, l, num_qubits,pauli_expectation_dict, circ, estimator):
        # Check if the given indices are invalid for the current system size
    if is_invalid_index(i, j, k, l, num_qubits):
        # If invalid, return the indices and a value of 0.0
        return i, j, k, l, 0.0
    
    # Construct the fermionic operator for the 2-RDM element
    # Fermionic operator format: f"+_{i} +_{j} -_{k} -_{l}"
    fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

    # Use Jordan-Wigner mapping to convert the fermionic operator to a qubit operator
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Initialize total expectation value to accumulate the results
    total_expectation_value = 0.0

    # Loop over Pauli terms in the qubit operator
    for pauli_op, coeff in sorted(qubit_term.label_iter()):
        expectation_value = coeff * pauli_expectation_dict[pauli_op]
        total_expectation_value += expectation_value

    # Store the final calculated value of the 2-RDM element
    temp = np.real(total_expectation_value)
    
    # Output the calculated 2-RDM element for debugging purposes
    print(f"Calculated 2-RDM element with Parallel after GPU Parallel 3th way, ({i}, {j}, {k}, {l}) is {temp}")
    # Return the indices and the calculated 2-RDM element
    return i, j, k, l, temp

@measure_execution_time
def get_two_rdm_Parallel3(num_qubits, circ=None, estimator=None):
    """
    Calculate the full two-body reduced density matrix (2-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.
    
    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure an estimator is provided, as it's necessary for the calculation
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")
    
    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)

    # 获取需要计算的 RDM 索引组合
    unique_indices = get_two_rdm_reduced_indices(num_qubits)

    # 生成 Pauli 项和对应关系
    pauli_dict, rdm_pauli_terms = generate_pauli_terms(unique_indices, num_qubits)
    # print(pauli_dict)
    # print(rdm_pauli_terms)

    # 计算 Pauli 项的期望值
    pauli_expectation_dict = compute_expectation_values(pauli_dict, circ, estimator)
    
    print(pauli_expectation_dict)
    
    # Use functools.partial to fix the constant parameters (num_qubits, circ, estimator)
    func = functools.partial(get_two_rdm_term_Parallel, num_qubits=num_qubits, pauli_expectation_dict = pauli_expectation_dict, circ=circ, estimator=estimator)

    # Use joblib's Parallel to compute 2-RDM elements in parallel (can adjust n_jobs for parallelism)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j, k, l) for i, j, k, l in unique_indices)

    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Fill the 2-RDM matrix with the calculated values, including symmetries
    for result in results:
        i, j, k, l, temp = result
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm

def assemble_two_rdm(unique_indices, num_qubits, pauli_expectation_dict, rdm_pauli_terms):
    """
    使用期望值和预处理信息，组装完整的 2-RDM。

    Parameters:
    -----------
    unique_indices : list of tuples
        需要计算的 RDM 索引组合。

    num_qubits : int
        系统中的量子比特数量。

    pauli_expectation_dict : dict
        Pauli 算符及其期望值的字典。

    rdm_pauli_terms : dict
        RDM 元素与对应的 Pauli 项及系数的映射。

    Returns:
    --------
    two_rdm : np.ndarray
        计算得到的二体约化密度矩阵。

    """    
    # 初始化 2-RDM 矩阵
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # 遍历所有 RDM 元素
    for idx in unique_indices:
        i, j, k, l = idx

        # 跳过无效索引
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # 初始化当前 RDM 元素的总期望值
        total_expectation_value = 0.0

        # 获取对应的 Pauli 项和系数
        pauli_terms = rdm_pauli_terms.get(idx, [])

        # 计算总期望值
        for pauli_op, coeff in pauli_terms:
            expectation_value = coeff * pauli_expectation_dict[pauli_op]
            total_expectation_value += expectation_value

        # 获取实部
        temp = np.real(total_expectation_value)
        
        # Output the calculated 2-RDM element for debugging purposes
        print(f"Calculated 2-RDM element with Parallel after GPU Parallel 2nd way, ({i}, {j}, {k}, {l}) is {temp}")
        
        # 填充 2-RDM 矩阵，考虑对称性和反对称性
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # 对称性
        two_rdm[j, i, k, l] = -temp      # 在 i, j 索引上的反对称性
        two_rdm[i, j, l, k] = -temp      # 在 k, l 索引上的反对称性

    return two_rdm

@measure_execution_time
def get_two_rdm_Parallel2(num_qubits, circ=None, estimator=None):
    """
    计算量子系统的二体约化密度矩阵（2-RDM）。

    Parameters:
    -----------
    num_qubits : int
        系统中的量子比特数量。

    circ : QuantumCircuit, optional
        用于制备量子态的量子电路。

    estimator : Estimator
        用于计算期望值的估计器对象。

    Returns:
    --------
    two_rdm : np.ndarray
        二体约化密度矩阵。

    """
    # 确保提供了估计器
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")

    # 获取需要计算的 RDM 索引组合
    unique_indices = get_two_rdm_reduced_indices(num_qubits)

    # 生成 Pauli 项和对应关系
    pauli_dict, rdm_pauli_terms = generate_pauli_terms(unique_indices, num_qubits)
    
    # print(pauli_dict)
    
    # print(rdm_pauli_terms)

    # 计算 Pauli 项的期望值
    pauli_expectation_dict = compute_expectation_values(pauli_dict, circ, estimator)
    
    # print(pauli_expectation_dict)
    
    # 组装 2-RDM
    
    two_rdm = assemble_two_rdm(unique_indices, num_qubits, pauli_expectation_dict, rdm_pauli_terms)

    return two_rdm

@measure_execution_time
def get_two_rdm_Parallel(num_qubits, circ=None, estimator=None):
    """
    Calculate the full two-body reduced density matrix (2-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.

    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.

    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure an estimator is provided, as it's necessary for the calculation
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")

    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)
    unique_indices = get_two_rdm_reduced_indices(num_qubits)

    # Initialize a dictionary to store Pauli terms and their total coefficients
    pauli_dict = {}

    # Initialize a dictionary to map RDM elements to their corresponding Pauli terms and coefficients
    rdm_pauli_terms = {}

    mapper = JordanWignerMapper()

    # Preprocessing: Generate Pauli terms for all RDM elements
    for idx in unique_indices:
        i, j, k, l = idx

        # Check if the given indices are invalid for the current system size
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # Construct the fermionic operator for the 2-RDM element
        fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

        # Map the fermionic operator to a qubit operator
        qubit_term = mapper.map(fermi_term)

        # For each Pauli term in the qubit operator, accumulate its coefficient
        for pauli_op, coeff in qubit_term.to_list():
            if pauli_op in pauli_dict:
                pauli_dict[pauli_op] += coeff
            else:
                pauli_dict[pauli_op] = coeff

            # Map RDM elements to their corresponding Pauli terms and coefficients
            if idx in rdm_pauli_terms:
                rdm_pauli_terms[idx].append((pauli_op, coeff))
            else:
                rdm_pauli_terms[idx] = [(pauli_op, coeff)]

    # Get the unique Pauli operators
    unique_pauli_ops = list(pauli_dict.keys())
    # print(unique_pauli_ops)
    # Convert Pauli operators to SparsePauliOp
    sparse_pauli_ops = [SparsePauliOp(pauli_op) for pauli_op in unique_pauli_ops]
    # print(sparse_pauli_ops)
    # Create a list of coefficients corresponding to the Pauli operators
    pauli_coeffs = [pauli_dict[pauli_op] for pauli_op in unique_pauli_ops]
    # print(pauli_coeffs)
    
    # Parallel computation of expectation values
    def compute_expectation(pauli_op):
        result = estimator.run(circuits=[circ], observables=[pauli_op]).result()
        return result.values[0]

    # Compute expectation values in parallel
    expectation_values = Parallel(n_jobs=-1)(
        delayed(compute_expectation)(pauli_op) for pauli_op in sparse_pauli_ops
    )

    # Create a dictionary to map Pauli operators to their expectation values
    pauli_expectation_dict = {
        pauli_op: value for pauli_op, value in zip(unique_pauli_ops, expectation_values)
    }
    print(pauli_expectation_dict)
    
    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Post-processing: Compute RDM elements using the precomputed expectation values
    for idx in unique_indices:
        i, j, k, l = idx

        # Skip invalid indices
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # Initialize total expectation value for this RDM element
        total_expectation_value = 0.0

        # Retrieve the Pauli terms and coefficients for this RDM element
        pauli_terms = rdm_pauli_terms.get(idx, [])

        # Sum over the Pauli terms
        for pauli_op, coeff in pauli_terms:
            expectation_value = coeff * pauli_expectation_dict[pauli_op]
            total_expectation_value += expectation_value

        # Store the calculated value in the 2-RDM matrix
        temp = np.real(total_expectation_value)
        
        # Output the calculated 2-RDM element for debugging purposes
        print(f"Calculated 2-RDM element with Parallel after GPU Parallel 1st way, ({i}, {j}, {k}, {l}) is {temp}")
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm



def get_one_rdm_term_wavefunction(i, j, num_qubits, ground_state_wavefunction):
    """
    Calculate a specific element of the one-body reduced density matrix (1-RDM) 
    using the ground state wavefunction.

    Parameters:
    -----------
    i, j : int
        Indices for the 1-RDM element.
    
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    tuple : (i, j, float)
        A tuple containing the indices (i, j) and the calculated value of the 1-RDM element.
    """
    # Check if the indices are out of bounds for the system size
    if i >= num_qubits or j >= num_qubits:
        # Return 0 if indices are invalid
        return i, j, 0.0
    
    # Create the fermionic operator for the 1-RDM element (creation at i, annihilation at j)
    fermi_term = FermionicOp({f"+_{i} -_{j}": 1}, num_spin_orbitals=num_qubits)
    
    # Map the fermionic operator to a qubit operator using the Jordan-Wigner transformation
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Compute the expectation value using the ground state wavefunction
    # If ground_state_wavefunction is a density matrix, we use the trace
    # Otherwise, it's assumed to be a state vector, and we compute ⟨ψ|O|ψ⟩
    qubit_matrix = qubit_term.to_matrix()
    
    if ground_state_wavefunction.ndim == 2:
        # Assume it's a density matrix (mixed state), calculate trace(ρO)
        temp = np.trace(ground_state_wavefunction @ qubit_matrix)
    else:
        # Assume it's a pure state, calculate ⟨ψ|O|ψ⟩
        temp = ground_state_wavefunction.conj().T @ qubit_matrix @ ground_state_wavefunction
    
    # Take the real part of the result
    temp = np.real(temp)
    
    print(f"Calculated 1-RDM element with wavefunction ({i}, {j}) is {temp}")
    
    return i, j, temp

@measure_execution_time
def get_one_rdm_wavefunction(num_qubits, ground_state_wavefunction):
    """
    Calculate the full one-body reduced density matrix (1-RDM) using the ground state wavefunction.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    one_rdm : np.ndarray
        A 2D numpy array representing the one-body reduced density matrix.
    """
    # Ensure the ground state wavefunction is provided
    if ground_state_wavefunction is None:
        raise ValueError("Ground state wavefunction must be provided.")
    
    # Initialize an empty matrix for the 1-RDM
    one_rdm = np.zeros((num_qubits, num_qubits), dtype=complex)
    
    # Use functools.partial to fix constant parameters for the 1-RDM element calculation
    func = functools.partial(get_one_rdm_term_wavefunction, num_qubits=num_qubits, ground_state_wavefunction=ground_state_wavefunction)
    
    # Generate all possible index combinations (i, j) for the 1-RDM
    indices = [(i, j) for i in range(num_qubits) for j in range(num_qubits)]
    
    # Parallel computation of all 1-RDM elements (set n_jobs according to your system)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j) for i, j in indices)

    # Fill the 1-RDM matrix with the computed results
    for result in results:
        i, j, temp = result
        one_rdm[i, j] = temp
        one_rdm[j, i] = temp
    return one_rdm

def get_two_rdm_term_wavefunction(i, j, k, l, num_qubits, ground_state_wavefunction):
    """
    Calculate a specific element of the two-body reduced density matrix (2-RDM)
    using the ground state wavefunction.

    Parameters:
    -----------
    i, j, k, l : int
        Indices for the 2-RDM element.
    
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    tuple : (i, j, k, l, float)
        A tuple containing the indices (i, j, k, l) and the calculated value of the 2-RDM element.
    """
    # Check if the indices are invalid based on system size or symmetry
    if is_invalid_index(i, j, k, l, num_qubits):
        return i, j, k, l, 0.0
    
    # Create the fermionic operator for the 2-RDM element (two creation, two annihilation operators)
    fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)
    
    # Map the fermionic operator to a qubit operator using the Jordan-Wigner transformation
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Convert the qubit operator to a matrix
    qubit_matrix = qubit_term.to_matrix()
    
    # Compute the expectation value using the ground state wavefunction
    if ground_state_wavefunction.ndim == 2:
        # If it's a density matrix, calculate Tr(ρO)
        temp = np.trace(ground_state_wavefunction @ qubit_matrix)
    else:
        # If it's a pure state, calculate ⟨ψ|O|ψ⟩
        temp = ground_state_wavefunction.conj().T @ qubit_matrix @ ground_state_wavefunction

    # Take the real part of the result
    temp = np.real(temp)
    
    print(f"Calculated 2-RDM element with wavefunction ({i}, {j}, {k}, {l}) is {temp}")
    
    return i, j, k, l, temp

@measure_execution_time
def get_two_rdm_wavefunction(num_qubits, ground_state_wavefunction):
    """
    Calculate the full two-body reduced density matrix (2-RDM) using the ground state wavefunction.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure that the ground state wavefunction is provided
    if ground_state_wavefunction is None:
        raise ValueError("Ground state wavefunction must be provided.")
    
    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)
    unique_indices = get_two_rdm_reduced_indices(num_qubits)
    
    # Use functools.partial to fix the constant parameters (num_qubits, ground_state_wavefunction)
    func = functools.partial(get_two_rdm_term_wavefunction, num_qubits=num_qubits, ground_state_wavefunction=ground_state_wavefunction)

    # Use parallel computing to compute 2-RDM elements
    results = Parallel(n_jobs=-1)(delayed(func)(i, j, k, l) for i, j, k, l in unique_indices)

    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Fill the 2-RDM matrix with the computed results, including symmetries
    for result in results:
        i, j, k, l, temp = result
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm


def create_gate_circuit(gate: str) -> QuantumCircuit:
    """
    Create and return a quantum circuit with a single qubit and apply the specified gate.

    Parameters:
    -----------
    gate (str): The name of the gate to apply (e.g., 'x', 'y', 'z', 'h', 's').

    Returns:
    --------
    QuantumCircuit: A quantum circuit with the specified gate applied.
    """
    circuit = QuantumCircuit(1)
    # Dynamically apply the specified gate using getattr
    getattr(circuit, gate)(0)
    return circuit

def find_closest_gate(original_gate: QuantumCircuit) -> object:
    """
    Find the closest Clifford gate to the given original gate by comparing the process fidelity.

    Parameters:
    -----------
    original_gate (QuantumCircuit): The quantum circuit representing the original gate.

    Returns:
    --------
    object: The closest Clifford gate's class instance.
    """
    # Dictionary of available Clifford gates
    gate_classes = {'h': HGate, 'x': XGate, 'y': YGate, 'z': ZGate, 's': SGate}
    
    max_fidelity = 0
    closest_gate_name = None

    # Compare each Clifford gate by calculating the process fidelity
    for gate_name, gate_class in gate_classes.items():
        gate_circuit = create_gate_circuit(gate_name)
        fidelity = process_fidelity(Operator(original_gate), Operator(gate_circuit))
        
        # Track the gate with the highest fidelity
        if fidelity > max_fidelity:
            max_fidelity = fidelity
            closest_gate_name = gate_name

    # Return the closest gate instance
    return gate_classes[closest_gate_name]()

def replace_gates_optimal(original_circuit, optimal_parameters_noiseless):
    """
    Replace parameterized gates in the original circuit with fixed Clifford gates based on fidelity
    and optimal noiseless parameters.

    Parameters:
    -----------
    original_circuit : QuantumCircuit
        The original quantum circuit to process.

    optimal_parameters_noiseless : list
        The optimal parameters for noiseless simulations, used for identifying gate replacements.

    Returns:
    --------
    QuantumCircuit or None: The new quantum circuit with gates replaced, or None if replacement failed.
    """
    # Ensure the circuit only uses the specified gate set
    gate_set = {instruction.operation.name for instruction in original_circuit.data}
    if not gate_set.issubset({'cx', 'h', 's', 'rz'}):
        # Transpile the circuit to use only the allowed gates
        original_circuit = transpile(original_circuit, basis_gates=['cx', 'h', 's', 'rz'])

    # Count the number of each type of gate in the original circuit
    gate_counts = {gate: sum(1 for instruction in original_circuit.data if instruction.operation.name == gate)
                   for gate in ['cx', 'h', 's', 'rz', 'x', 'y']}
    
    # Print gate counts for the original circuit
    for gate, count in gate_counts.items():
        print(f"Number of '{gate}' gates before replacing gates: {count}")

    # Count the number of parameterized gates in the original circuit
    original_num_param_gates = sum(1 for instruction in original_circuit.data
                                   if any(isinstance(param, ParameterExpression) for param in instruction.operation.params))
    print(f"Number of parameterized gates before replacing gates: {original_num_param_gates}")

    # Create a new quantum circuit with the same number of qubits
    new_circuit = QuantumCircuit(original_circuit.num_qubits)

    replaced_gates = 0
    print(optimal_parameters_noiseless)

    # Iterate through the instructions of the original circuit
    for instruction in original_circuit.data:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        # Check if any of the instruction's parameters are parameter expressions
        if any(isinstance(param, ParameterExpression) for param in instr.params):
            # Replace the parameterized gate with the closest Clifford gate
            new_circuit.append(SGate(), [qarg for qarg in qargs])
            replaced_gates += 1
        else:
            # Otherwise, keep the original gate as is
            new_circuit.append(instr, qargs, cargs)

    print(f"Total parameter gates replaced: {replaced_gates}")

    # Print gate counts for the new circuit
    new_gate_counts = {gate: sum(1 for instruction in new_circuit.data if instruction.operation.name == gate)
                       for gate in ['cx', 'h', 's', 'rz', 'x', 'y', 'z']}
    
    for gate, count in new_gate_counts.items():
        print(f"Number of '{gate}' gates after replacing gates: {count}")

    # Check if the conditions for gate replacement are met
    single_gates = {'x', 'y', 'z', 'h', 's'}
    original_single_gates = sum(1 for instruction in original_circuit.data if instruction.operation.name in single_gates)
    new_circuit_single_gates = sum(1 for instruction in new_circuit.data if instruction.operation.name in single_gates)

    if original_single_gates + original_num_param_gates == new_circuit_single_gates:
        print("Gate replacement successful.")
        return new_circuit
    else:
        print("Error: Gate replacement unsuccessful.")
        return None

def check_clifford_and_get_stabilizer(circuit: QuantumCircuit) -> str:
    """
    Check if the given quantum circuit is a Clifford circuit.
    If it is a Clifford circuit, return its stabilizers.
    If it is not a Clifford circuit, return a message indicating so.

    Parameters:
    -----------
    circuit : QuantumCircuit
        The quantum circuit to check for Clifford property.
        
    Returns:
    --------
    str
        A message indicating whether the circuit is a Clifford circuit or not,
        and if it is, returns the stabilizer group of the circuit.
    """
    try:
        # Convert the quantum circuit to a Clifford object
        cliff = Clifford(circuit)
        
        # Get the stabilizers of the Clifford circuit
        stabilizer = cliff.to_labels(mode="S")  # "S" mode returns stabilizers in label form
        
        # Return a message with the stabilizers
        return f"This is a Clifford circuit. The stabilizers are: {stabilizer}"
    
    except QiskitError as e:
        # Catch the error if the circuit is not a Clifford circuit
        return f"This is not a Clifford circuit. Error: {str(e)}"

    
def RDM_with_given_wavefunction(num_qubits: int, ground_state_wavefunction: np.ndarray):
    """
    Calculate both the one-body and two-body reduced density matrices (RDMs)
    using a given ground state wavefunction.

    Parameters:
    -----------
    num_qubits : int
        The number of qubits (spin orbitals) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    tuple : (np.ndarray, np.ndarray)
        The one-body RDM and two-body RDM calculated from the given wavefunction.
    """
    # Compute the one-body reduced density matrix using the ground state wavefunction
    one_RDM_with_given_wavefunction = get_one_rdm_wavefunction(num_qubits, ground_state_wavefunction)
    
    # Compute the two-body reduced density matrix using the ground state wavefunction
    two_RDM_with_given_wavefunction = get_two_rdm_wavefunction(num_qubits, ground_state_wavefunction)
    
    return one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction

def RDM_with_given_circuit(num_qubits: int, circuit: QuantumCircuit, estimator: Estimator):
    """
    Calculate both the one-body and two-body reduced density matrices (RDMs)
    using a given quantum circuit and estimator.

    Parameters:
    -----------
    num_qubits : int
        The number of qubits (spin orbitals) in the system.
    
    circuit : QuantumCircuit
        The quantum circuit used to prepare the quantum state for measurement.
    
    estimator : Estimator
        The estimator object (either noiseless or noisy) used to compute expectation values.

    Returns:
    --------
    tuple : (np.ndarray, np.ndarray)
        The one-body RDM and two-body RDM calculated from the given quantum circuit.
    """
    # Compute the two-body reduced density matrix using the quantum circuit and estimator
    two_rdm_given_circuit = get_two_rdm(num_qubits, circ=circuit, estimator=estimator)
    
    # Compute the one-body reduced density matrix using the quantum circuit and estimator
    one_rdm_given_circuit = get_one_rdm(num_qubits, circ=circuit, estimator=estimator)
    
    return one_rdm_given_circuit, two_rdm_given_circuit

def VQE_RDM_thm_noisy_RG(molecule: str, d:str, num_qubits: int, n_particles: int, prob_1: float, prob_2: float, shots_num: int):
    """
    执行无噪声和有噪声的VQE，并返回计算得到的1-RDM和2-RDM。

    参数:
    molecule : str
        分子名称
    d : float
        分子距离
    num_qubits : int
        量子比特数
    n_particles : int
        粒子数
    prob_1 : float
        1-比特门的噪声概率
    prob_2 : float
        2-比特门的噪声概率
    shots_num : int
        模拟中的shots数量

    返回:
    tuple: (1-RDM和2-RDM的多组结果)
    """
    _, noisy_estimator = build_noisy_estimator(prob_1=prob_1, prob_2=prob_2, shots_num=shots_num)

    # 文件名
    current_dir = os.getcwd() # 获取当前工作目录的字符串 (e.g., '/home/user/project')
    input_filename = f"{molecule}_{d}_sto-3g.pkl"
    full_input_path = os.path.join(current_dir, input_filename) # 安全地拼接路径
    try:
        with open(full_input_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found. Skipping this distance.")
        print(f"Searched for file at: {full_input_path}")
        return f"Failed for d={d:.2f}"


    qubit_hamiltonian = data['qubit_hamiltonian']
    n_particles = data['n_particles']

    # 生成无噪声和有噪声的 ansatz 和 VQE 结果
    hamiltonian_qiskit = Gene_Qiskit_VQE_hamiltonian(num_qubits, qubit_hamiltonian)
    
    result_noiseless, _, circ_noiseless, estimator_noiseless, optimal_parameters_noiseless = UCCSD_VQE_noiseless(
        hamiltonian_qiskit=hamiltonian_qiskit, num_qubits=num_qubits, n_particles=n_particles, max_iterations=5000)
    print(result_noiseless)
    
    result_noisy, _, circ_noisy, _, _ = UCCSD_VQE_noisy(
        hamiltonian_qiskit=hamiltonian_qiskit, num_qubits=num_qubits, n_particles=n_particles, noisy_estimator=noisy_estimator, max_iterations=5000)
    print(result_noisy)
    
    two_rdm_noisy_Parallel = get_two_rdm_Parallel(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    two_rdm_noisy_Parallel2 = get_two_rdm_Parallel2(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    two_rdm_noisy_Parallel3 = get_two_rdm_Parallel3(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    
    # 生成并优化 ansatz
    circ_rgo = replace_gates_optimal(circ_noiseless, optimal_parameters_noiseless)
    # 基态波函数生成并计算RDM
    RG_ground_state_wavefunction = Statevector(circ_rgo).data
    RG_ground_state_wavefunction = to_density_matrix(RG_ground_state_wavefunction)
    
    one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction = RDM_with_given_wavefunction(num_qubits, RG_ground_state_wavefunction)
    
    one_rdm_given_circuit, two_rdm_given_circuit = RDM_with_given_circuit(num_qubits, circ_rgo, noisy_estimator)


    # 使用无噪声和有噪声量子电路的 RDM 结果
    ground_state_wavefunction_noiseless = Statevector(circ_noiseless).data
    ground_state_wavefunction_noiseless = to_density_matrix(ground_state_wavefunction_noiseless)
    
    two_rdm_noiseless = get_two_rdm(num_qubits, circ=circ_noiseless, estimator = estimator_noiseless)

    
    two_rdm_thm = get_two_rdm_wavefunction(num_qubits, ground_state_wavefunction=ground_state_wavefunction_noiseless)
    two_rdm_noisy = get_two_rdm(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    two_rdm_noisy_Parallel = get_two_rdm_Parallel(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    one_rdm_noiseless = get_one_rdm(num_qubits, circ=circ_noiseless, estimator = estimator_noiseless)
    one_rdm_thm = get_one_rdm_wavefunction(num_qubits, ground_state_wavefunction=ground_state_wavefunction_noiseless)
    one_rdm_noisy = get_one_rdm(num_qubits, circ=circ_noisy, estimator=noisy_estimator)

    return (one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction, 
            one_rdm_given_circuit, two_rdm_given_circuit,
            one_rdm_noiseless, two_rdm_noiseless,
            one_rdm_noisy, two_rdm_noisy,
            one_rdm_thm,two_rdm_thm,two_rdm_noisy_Parallel,two_rdm_noisy_Parallel2,two_rdm_noisy_Parallel3,result_noiseless,result_noisy)

def calculate_and_display_rdm_table(rdm_list, rdm_type="RDM", labels=None):
    """
    计算并生成 RDM 的两两 Frobenius 范数对比表格，横纵坐标为 RDM 简写标签。

    参数:
    rdm_list : list
        包含所有要比较的 RDM 矩阵。
    rdm_type : str
        1-RDM 或 2-RDM 的类型，用于保存文件时命名。
    labels : list
        每个 RDM 的简写名称，用于表格显示。

    输出:
    Pandas DataFrame 显示并保存为 CSV 文件。
    """
    n = len(labels)
    
    # 创建空矩阵存储范数
    norm_matrix = np.zeros((n, n))

    # 两两组合计算 Frobenius 范数，并填充矩阵
    for (i, j) in combinations(range(n), 2):
        norm, _ = calculate_frobenius_norm_difference(rdm_list[i], rdm_list[j])
        norm_matrix[i, j] = norm
        norm_matrix[j, i] = norm  # 对称填充

    # 创建 Pandas DataFrame 生成表格
    df = pd.DataFrame(norm_matrix, index=labels, columns=labels)
    
    return df
def process_distance(d, molecule, prob_1, prob_2, shots_num):
    """
    处理单个距离d值的所有计算和数据保存任务。
    这是一个独立的“工作单元”，非常适合并行化。
    """
    print(f"--- [START] Processing for d = {d:.1f} ---")
    current_dir = os.getcwd() # 获取当前工作目录的字符串 (e.g., '/home/user/project')
    input_filename = f"{molecule}_{d}_sto-3g.pkl"
    full_input_path = os.path.join(current_dir, input_filename) # 安全地拼接路径
    
    # 3. 在 open() 中使用构建好的完整路径
    print(f"Attempting to load file: {full_input_path}")
    try:
        with open(full_input_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found. Skipping this distance.")
        print(f"Searched for file at: {full_input_path}")
        return f"Failed for d={d:.2f}"

    try:
        with open(input_filename, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found for d = {d:.2f}. Skipping this distance.")
        print(f"Missing file: {input_filename}")
        return f"Failed for d={d:.2f}"

    # 访问变量
    molecular_hamiltonian = data['molecular_hamiltonian']
    qubit_hamiltonian = data['qubit_hamiltonian']
    FCI_val = data['FCI_val']
    one_body_coefficients = data['one_body_coefficients']
    two_body_coefficients = data['two_body_coefficients']
    constant = data['constant']
    n_elec = data['n_elec']
    n_particles = data['n_particles']
    num_qubits = data['num_qubits']

    # 2. 执行核心计算
    
    (one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction, 
     one_rdm_given_circuit, two_rdm_given_circuit,
     one_rdm_noiseless, two_rdm_noiseless,
     one_rdm_noisy, two_rdm_noisy,
     one_rdm_thm, two_rdm_thm,
     two_rdm_noisy_Parallel, two_rdm_noisy_Parallel2, 
     two_rdm_noisy_Parallel3,result_noiseless,result_noisy) = VQE_RDM_thm_noisy_RG(
        molecule=molecule, d=d, num_qubits=num_qubits, n_particles=n_particles, 
        prob_1=prob_1, prob_2=prob_2, shots_num=shots_num
    )

    # 3. 后处理和数据整理
    one_rdm_list = [one_RDM_with_given_wavefunction, one_rdm_given_circuit, one_rdm_noiseless, one_rdm_noisy, one_rdm_thm]
    two_rdm_list = [two_RDM_with_given_wavefunction, two_rdm_given_circuit, two_rdm_noiseless, two_rdm_noisy, two_rdm_thm]
    labels = ['WGF', 'CIRC', 'NSL', 'NOI', 'THM']
    
    df_1_rdm = calculate_and_display_rdm_table(one_rdm_list, rdm_type="1-RDM", labels=labels)
    df_2_rdm = calculate_and_display_rdm_table(two_rdm_list, rdm_type="2-RDM", labels=labels)

    # 4. 使用包含d的动态文件名保存输出
    # 格式化d为两位小数，确保文件名统一美观
    d_str = f"{d:.1f}" 

    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_one_rdm_list.pkl'), 'wb') as f:
        pickle.dump(one_rdm_list, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_two_rdm_list.pkl'), 'wb') as f:
        pickle.dump(two_rdm_list, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_df_1_rdm.pkl'), 'wb') as f:
        pickle.dump(df_1_rdm, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_df_2_rdm.pkl'), 'wb') as f:
        pickle.dump(df_2_rdm, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_result_noiseless.pkl'), 'wb') as f:
        pickle.dump(result_noiseless, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_result_noisy.pkl'), 'wb') as f:
        pickle.dump(result_noisy, f)
    print(f"--- [DONE]  Processing for d = {d:.2f}. Data saved. ---")
    return f"Success for d={d:.2f}"


if __name__ == "__main__":
    
    molecule = "H4"
    
    d_list = [1.5,1.6,1.7,1.8,2.1,2.2,2.3,2.4,2.7,2.8,2.9,3.0,3.3,3.4,3.5]
    current_value = 0.4

    # while current_value <= 1.2:
    #     d_list.append(round(current_value, 1))
    #     current_value += 0.1
    #
    # while current_value <= 3.5:
    #     d_list.append(round(current_value, 1))
    #     current_value += 0.3

    print(d_list)
    
    # 噪声模型的参数（在循环外定义一次即可）
    prob_1 = 0.001   # 1-比特门的噪声概率
    prob_2 = 0.01    # 2-比特门的噪声概率
    shots_num = 10000 # 模拟的shots数量

    # 使用 joblib 并行执行
    # n_jobs=-1 表示使用所有可用的CPU核心
    # verbose=10 会打印详细的进度信息，非常有用！
    print("\nStarting parallel processing with joblib...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_distance)(d, molecule, prob_1, prob_2, shots_num) for d in d_list
    )
    
    print("\n--- All jobs completed! ---")
    print("Summary of results:", results)