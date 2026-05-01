import numpy as np

def compute_pauli_counts(input_file, output_file , N):
    """
    Reads a file where:
    - First `n` rows represent a Pauli string (each row = one element of the string).
    - Last row represents the probability distribution.
    
    Computes counts as `count = N * p` and saves to a file.

    :param input_file: Name of the input file containing Pauli strings and probabilities.
    :param output_file: Name of the output file to save counts.
    :param N: Number of samples.
    """
    # Load data from file
    data = np.loadtxt(input_file)

    # Extract the number of qubits (Pauli string length)
    n = data.shape[0] - 1  # Last row is the probability

    # Extract Pauli strings (n rows)
    pauli_strings = data[:n, :].T.astype(int)  # Transpose to align rows properly

    # Extract probabilities (last row)
    probabilities = data[-1, :]

    # Compute counts
    counts = (N * probabilities).astype(int)  # Convert to integer counts

    # Save output in the format: Pauli string followed by count
    with open(output_file, "w+") as f:
        for i in range(len(counts)):
            f.write(f"{counts[i]}" +" " + " ".join(map(str, pauli_strings[i])) +"\n")

    print(f"Counts saved to {output_file}")


