import argparse
import os
from typing import Iterable, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
import qiskit_nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.sparse.linalg import eigsh


PAULI_TO_OGM = {
    "I": "0",
    "X": "1",
    "Y": "2",
    "Z": "3",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_coeff(coeff: complex, atol: float = 1e-12) -> str:
    real = float(np.real(coeff))
    imag = float(np.imag(coeff))

    if abs(imag) <= atol:
        return f"({real:.16f}+0.0j)"
    sign = "+" if imag >= 0 else "-"
    return f"({real:.16f}{sign}{abs(imag):.16f}j)"


def simplify_real_if_close(coeff: complex, atol: float = 1e-12) -> complex:
    if abs(np.imag(coeff)) <= atol:
        return complex(float(np.real(coeff)), 0.0)
    return complex(coeff)


def to_sparse_pauli_op(qubit_op) -> SparsePauliOp:
    # qiskit-nature<=0.6 may return PauliSumOp, whose primitive is SparsePauliOp.
    if isinstance(qubit_op, SparsePauliOp):
        return qubit_op
    primitive = getattr(qubit_op, "primitive", None)
    if isinstance(primitive, SparsePauliOp):
        return primitive
    raise TypeError(f"Unsupported qubit operator type: {type(qubit_op)}")


def build_qubit_hamiltonian(
    atom: str,
    basis: str,
    charge: int,
    spin: int,
    add_nuclear_repulsion: bool,
    simplify_atol: float,
) -> SparsePauliOp:
    # Prefer SparsePauliOp path to avoid deprecated PauliSumOp APIs.
    try:
        qiskit_nature.settings.use_pauli_sum_op = False
    except Exception:
        pass

    driver = PySCFDriver(atom=atom, basis=basis, charge=charge, spin=spin)
    problem = driver.run()

    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = to_sparse_pauli_op(mapper.map(fermionic_op)).simplify(atol=simplify_atol)

    if add_nuclear_repulsion:
        e_nuc = problem.hamiltonian.nuclear_repulsion_energy
        if e_nuc is not None:
            identity = "I" * qubit_op.num_qubits
            qubit_op = (qubit_op + SparsePauliOp.from_list([(identity, complex(e_nuc))])).simplify(
                atol=simplify_atol
            )

    return qubit_op


def iter_terms_from_sparse_pauli_op(qubit_op: SparsePauliOp) -> Iterable[Tuple[str, complex]]:
    labels = qubit_op.paulis.to_labels()
    coeffs = qubit_op.coeffs
    for label, coeff in zip(labels, coeffs):
        yield label, simplify_real_if_close(complex(coeff))


def save_pauli_txt(terms: Iterable[Tuple[str, complex]], out_file: str, atol: float = 1e-12) -> None:
    with open(out_file, "w", encoding="utf-8") as f:
        first = True
        for pauli_str, coeff in terms:
            if not first:
                f.write("\n")
            f.write(f"{pauli_str}\n")
            f.write(f"{format_coeff(coeff, atol=atol)}\n")
            first = False


def save_ogm_txt(terms: Iterable[Tuple[str, complex]], out_file: str, atol: float = 1e-12) -> None:
    with open(out_file, "w", encoding="utf-8") as f:
        for pauli_str, coeff in terms:
            real = float(np.real(coeff))
            imag = float(np.imag(coeff))
            if abs(imag) > atol:
                raise ValueError(
                    f"Term {pauli_str} has non-negligible imaginary part ({imag}). "
                    "OGM file only supports real coefficients."
                )
            indices = " ".join(PAULI_TO_OGM[ch] for ch in pauli_str)
            f.write(f"{real:.16f} {indices}\n")


def save_hamiltonian_dense_npy(qubit_op: SparsePauliOp, out_file: str) -> None:
    h_dense = qubit_op.to_matrix(sparse=False)
    np.save(out_file, h_dense)


def solve_ground_state_vector(qubit_op: SparsePauliOp, k: int = 1):
    h_sparse = qubit_op.to_matrix(sparse=True)
    evals, evecs = eigsh(h_sparse, k=k, which="SA")
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals[0], evecs[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BeH2 STO-3G qubit Hamiltonian and ground-state vector."
    )
    parser.add_argument(
        "--atom",
        type=str,
        default="Be 0.000000 0.000000 0.000000; H 0.000000 0.000000 1.326400; H 0.000000 0.000000 -1.326400",
        help="Molecular geometry in PySCF format.",
    )
    parser.add_argument("--basis", type=str, default="sto3g", help="Basis set, e.g. sto3g.")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge.")
    parser.add_argument("--spin", type=int, default=0, help="2S value (0 for singlet).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="haozhaowu/BeH2/hamil_class",
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="BeH2_sto3g",
        help="Tag used in output file names.",
    )
    parser.add_argument(
        "--expected-qubits",
        type=int,
        default=14,
        help="If set, raise if qubit count differs.",
    )
    parser.add_argument(
        "--skip-dense-hamiltonian",
        action="store_true",
        help="Skip dense Hamiltonian .npy export (dense is ~4GB for 14 qubits).",
    )
    parser.add_argument(
        "--no-nuclear-repulsion",
        action="store_true",
        help="Do not add nuclear repulsion energy to identity term.",
    )
    parser.add_argument(
        "--simplify-atol",
        type=float,
        default=1e-12,
        help="Tolerance used when simplifying mapped Pauli operator.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ogm_dir = os.path.join(args.output_dir, "ogm_inputs")
    ensure_dir(ogm_dir)

    qubit_op = build_qubit_hamiltonian(
        atom=args.atom,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        add_nuclear_repulsion=not args.no_nuclear_repulsion,
        simplify_atol=args.simplify_atol,
    )

    if args.expected_qubits is not None and qubit_op.num_qubits != int(args.expected_qubits):
        raise ValueError(
            f"Expected {args.expected_qubits} qubits, got {qubit_op.num_qubits}. "
            "Check geometry/basis/charge/spin."
        )

    terms = list(iter_terms_from_sparse_pauli_op(qubit_op))

    pauli_txt_path = os.path.join(args.output_dir, f"hamiltonian_{args.tag}_{qubit_op.num_qubits}_pauli.txt")
    ogm_txt_path = os.path.join(ogm_dir, f"ogm_{args.tag}_{qubit_op.num_qubits}.txt")
    h_npy_path = os.path.join(args.output_dir, f"hamiltonian_{args.tag}_{qubit_op.num_qubits}.npy")
    state_npy_path = os.path.join(args.output_dir, f"state_{args.tag}_{qubit_op.num_qubits}_vector.npy")

    save_pauli_txt(terms, pauli_txt_path, atol=args.simplify_atol)
    save_ogm_txt(terms, ogm_txt_path, atol=args.simplify_atol)

    if not args.skip_dense_hamiltonian:
        save_hamiltonian_dense_npy(qubit_op, h_npy_path)

    e0, psi0 = solve_ground_state_vector(qubit_op)
    np.save(state_npy_path, psi0.astype(np.complex128))

    print(f"Qubit count: {qubit_op.num_qubits}")
    print(f"Pauli terms: {len(terms)}")
    print(f"Ground-state energy (with current Hamiltonian): {e0:.12f}")
    print(f"Saved Pauli txt: {pauli_txt_path}")
    print(f"Saved OGM txt: {ogm_txt_path}")
    if not args.skip_dense_hamiltonian:
        print(f"Saved Hamiltonian npy: {h_npy_path}")
    else:
        print("Skipped dense Hamiltonian npy export.")
    print(f"Saved state vector npy: {state_npy_path}")


if __name__ == "__main__":
    main()

