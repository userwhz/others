"""
BeH2 分子：PySCF + Jordan–Wigner 映射后的 Pauli 哈密顿量，仅导出 txt 与 json（与 h2o.py 一致）。
"""
import argparse
import json
import os
from typing import Dict, Iterable, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
import qiskit_nature
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


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


def save_pauli_json(terms: Iterable[Tuple[str, complex]], out_file: str, atol: float = 1e-12) -> None:
	"""与 haozhaowu/H2O/h2o.py 一致：{"IIIX...": 实系数, ...}，sort_keys。"""
	H_dict: Dict[str, float] = {}
	for pauli_str, coeff in terms:
		if abs(coeff) <= atol:
			continue
		H_dict[pauli_str] = float(np.real(coeff))
	with open(out_file, "w", encoding="utf-8") as f:
		json.dump(H_dict, f, indent=0, sort_keys=True)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate BeH2 qubit Hamiltonian (Pauli txt + json only)."
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
		help="Tag used in output file names (qubit count is appended).",
	)
	parser.add_argument(
		"--expected-qubits",
		type=int,
		default=14,
		help="If set, raise if qubit count differs.",
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

	qubit_op = build_qubit_hamiltonian(
		atom=args.atom,
		basis=args.basis,
		charge=args.charge,
		spin=args.spin,
		add_nuclear_repulsion=not args.no_nuclear_repulsion,
		simplify_atol=args.simplify_atol,
	)

	nq = qubit_op.num_qubits
	if args.expected_qubits is not None and nq != int(args.expected_qubits):
		raise ValueError(
			f"Expected {args.expected_qubits} qubits, got {nq}. "
			"Check geometry/basis/charge/spin."
		)

	terms = list(iter_terms_from_sparse_pauli_op(qubit_op))

	pauli_txt_path = os.path.join(args.output_dir, f"hamiltonian_{args.tag}_{nq}_pauli.txt")
	pauli_json_path = os.path.join(args.output_dir, f"hamiltonian_{args.tag}_{nq}.json")

	save_pauli_txt(terms, pauli_txt_path, atol=args.simplify_atol)
	save_pauli_json(terms, pauli_json_path, atol=args.simplify_atol)

	print(f"Qubit count: {nq}")
	print(f"Pauli terms: {len(terms)}")
	print(f"Saved Pauli txt: {pauli_txt_path}")
	print(f"Saved Pauli json: {pauli_json_path}")


if __name__ == "__main__":
	main()
