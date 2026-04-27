import argparse
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import qiskit_nature
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper, ParityMapper
from scipy.sparse.linalg import eigsh


PAULI_TO_OGM = {
	"I": "0",
	"X": "1",
	"Y": "2",
	"Z": "3",
}


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def simplify_real_if_close(coeff: complex, atol: float = 1e-12) -> complex:
	if abs(np.imag(coeff)) <= atol:
		return complex(float(np.real(coeff)), 0.0)
	return complex(coeff)


def format_coeff(coeff: complex, atol: float = 1e-12) -> str:
	real = float(np.real(coeff))
	imag = float(np.imag(coeff))

	if abs(imag) <= atol:
		return f"({real:.16f}+0.0j)"
	sign = "+" if imag >= 0 else "-"
	return f"({real:.16f}{sign}{abs(imag):.16f}j)"


def to_sparse_pauli_op(qubit_op) -> SparsePauliOp:
	if isinstance(qubit_op, SparsePauliOp):
		return qubit_op
	primitive = getattr(qubit_op, "primitive", None)
	if isinstance(primitive, SparsePauliOp):
		return primitive
	raise TypeError(f"Unsupported qubit operator type: {type(qubit_op)}")


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


def solve_ground_state_vector(qubit_op: SparsePauliOp):
	h_sparse = qubit_op.to_matrix(sparse=True)
	evals, evecs = eigsh(h_sparse, k=1, which="SA")
	return float(evals[0]), evecs[:, 0].astype(np.complex128)


def build_problem(atom: str, basis: str, charge: int, spin: int):
	try:
		qiskit_nature.settings.use_pauli_sum_op = False
	except Exception:
		pass

	driver = PySCFDriver(atom=atom, basis=basis, charge=charge, spin=spin)
	return driver.run()


def get_mappers() -> Dict[str, object]:
	return {
		"jw": JordanWignerMapper(),
		"parity": ParityMapper(),
		"bk": BravyiKitaevMapper(),
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate LiH STO-3G Hamiltonians (jw/parity/bk) and quantum states."
	)
	parser.add_argument(
		"--atom",
		type=str,
		default="Li 0.0 0.0 0.0; H 0.0 0.0 1.6",
		help="Molecular geometry in PySCF format.",
	)
	parser.add_argument("--basis", type=str, default="sto3g", help="Basis set, e.g. sto3g.")
	parser.add_argument("--charge", type=int, default=0, help="Molecular charge.")
	parser.add_argument("--spin", type=int, default=0, help="2S value (0 for singlet).")
	parser.add_argument(
		"--output-dir",
		type=str,
		default="haozhaowu/LiH12/hamil_class",
		help="Directory for generated files.",
	)
	parser.add_argument(
		"--tag-prefix",
		type=str,
		default="LiH_sto3g_12",
		help="Prefix used in output file names. Mapper suffix will be appended.",
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

	problem = build_problem(args.atom, args.basis, args.charge, args.spin)
	fermionic_op = problem.hamiltonian.second_q_op()
	e_nuc = problem.hamiltonian.nuclear_repulsion_energy

	for mapper_name, mapper in get_mappers().items():
		qubit_op = to_sparse_pauli_op(mapper.map(fermionic_op)).simplify(atol=args.simplify_atol)

		if e_nuc is not None:
			identity = "I" * qubit_op.num_qubits
			qubit_op = (qubit_op + SparsePauliOp.from_list([(identity, complex(e_nuc))])).simplify(
				atol=args.simplify_atol
			)

		tag = f"{args.tag_prefix}{mapper_name}"
		terms = list(iter_terms_from_sparse_pauli_op(qubit_op))

		pauli_txt_path = os.path.join(args.output_dir, f"hamiltonian_{tag}_pauli.txt")
		h_npy_path = os.path.join(args.output_dir, f"hamiltonian_{tag}.npy")
		ogm_txt_path = os.path.join(ogm_dir, f"ogm_hamiltonian_{tag}.txt")
		state_vec_path = os.path.join(args.output_dir, f"state_{tag}_vector.npy")
		state_rho_path = os.path.join(args.output_dir, f"state_{tag}_rho.npy")

		save_pauli_txt(terms, pauli_txt_path, atol=args.simplify_atol)
		save_ogm_txt(terms, ogm_txt_path, atol=args.simplify_atol)
		save_hamiltonian_dense_npy(qubit_op, h_npy_path)

		e0, psi0 = solve_ground_state_vector(qubit_op)
		rho0 = np.outer(psi0, np.conjugate(psi0)).astype(np.complex128)
		np.save(state_vec_path, psi0)
		np.save(state_rho_path, rho0)

		print(f"[{mapper_name}] qubits={qubit_op.num_qubits}, terms={len(terms)}, E0={e0:.12f}")
		print(f"  Pauli txt: {pauli_txt_path}")
		print(f"  OGM input: {ogm_txt_path}")
		print(f"  Hamiltonian npy: {h_npy_path}")
		print(f"  State vector npy: {state_vec_path}")
		print(f"  Density matrix npy: {state_rho_path}")


if __name__ == "__main__":
	main()
