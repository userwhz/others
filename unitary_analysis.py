"""
Analyze unitary matrices from build_fc_group_plan for BeH2.

Key question: For a pure state |ψ⟩, what happens when we apply
the unitary U from FC joint measurement?

Theory:
  ρ = |ψ⟩⟨ψ⟩  has rank 1 (pure state).
  U|ψ⟩ is also pure (unitary evolution preserves purity).
  After measurement: outcome is sampled from |⟨b|U|ψ⟩|².
  The eigenvalues matrix maps bitstring b → ±1 for each observable.
"""

import sys
import os
import numpy as np

if os.path.isdir("shadowgrouping"):
    sys.path.insert(0, ".")

from shadowgrouping.hamiltonian import load_pauli_list6
from shadowgrouping.measurement_schemes import (
    build_fc_group_plan, build_commuting_groups
)


def main():
    print("=" * 60)
    print("BeH2: STATE ANALYSIS")
    print("=" * 60)

    observables, w, offset, E_GS, state = load_pauli_list6(
        "haozhaowu/", "BeH2", "sto3g", "JW"
    )
    
    n_qubits = int(np.log2(state.shape[0]))
    norm_sq = np.real(np.vdot(state, state))
    print(f"Qubits: {n_qubits}, dim: {state.shape[0]}")
    print(f"⟨ψ|ψ⟩ = {norm_sq:.6f}")
    print(f"State type: {'PURE (state vector)' if state.ndim == 1 else 'DENSITY MATRIX'}")
    
    # For pure state: ρ = |ψ⟩⟨ψ|, rank(ρ) = 1 always
    print(f"\nρ = |ψ⟩⟨ψ|  =>  rank(ρ) = 1  (pure state by definition)")
    print(f"This is INDEPENDENT of measurement, grouping, or basis choice.")

    # Build FC groups (with qubit limit to avoid memory issues)
    MAX_QUBITS = 6
    groups = build_commuting_groups(observables, commutation_mode="fc",
                                     max_support_qubits=MAX_QUBITS)
    print(f"\n{'='*60}")
    print(f"BeH2: FC GROUP ANALYSIS (max_support_qubits={MAX_QUBITS})")
    print(f"{'='*60}")
    print(f"Total FC groups: {len(groups)}")

    # Find a group with qubits in [1, MAX_QUBITS] range
    for gid, group in enumerate(groups):
        qubits = np.where(np.any(observables[group] != 0, axis=0))[0]
        if 1 <= len(qubits) <= MAX_QUBITS:
            print(f"\nAnalyzing group {gid}: {len(group)} observables, "
                  f"{len(qubits)} qubits (indices {qubits.tolist()})")

            plan = build_fc_group_plan(observables, group, group_id=gid,
                                        max_support_qubits=MAX_QUBITS)
            U = plan["unitary"]
            eigvals = plan["eigenvalues"]

            dim = U.shape[0]
            print(f"  Unitary shape: {U.shape} ({dim}×{dim})")
            print(f"  Unitary dtype:  {U.dtype}")
            print(f"  Rank(U):        {np.linalg.matrix_rank(U)} (full rank expected)")
            print(f"  U·U† ≈ I?       {np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-5)}")
            print(f"  Eigenvalues:    {eigvals.shape[0]} obs × {eigvals.shape[1]} patterns")
            
            unique_patterns = len(np.unique(eigvals, axis=1))
            print(f"  Unique patterns: {unique_patterns} (max = 2^{len(qubits)} = {2**len(qubits)})")

            # Show first few eigenvalue columns as example
            print(f"\n  Example: first 4 eigenvalue columns (rows = observables)")
            for obs_i in range(min(4, eigvals.shape[0])):
                print(f"    obs[{obs_i}]: {eigvals[obs_i, :8].tolist()}...")

            # Now simulate: what happens when we apply U to a random pure state
            # and measure?
            print(f"\n  Simulating measurement of U|ψ⟩:")
            # Generate a random pure state on k qubits
            k = len(qubits)
            psi_test = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi_test /= np.linalg.norm(psi_test)

            # Apply unitary
            psi_rot = U @ psi_test
            probs = np.abs(psi_rot)**2  # Born probabilities

            # Entropy of measurement outcome distribution
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            print(f"  Measurement entropy: {entropy:.2f} bits "
                  f"(max = {k:.0f} for uniform)")

            print(f"\n  What happens during FC measurement:")
            print(f"  1. Apply U to |ψ⟩ → rotates to joint eigenbasis")
            print(f"  2. Measure in computational basis → get bitstring b")
            print(f"  3. Look up eigenvalues[:, b] → ±1 for each observable")
            print(f"  4. This is a PROJECTIVE MEASUREMENT")
            print(f"  5. Post-measurement state: |b⟩ (eigenstate of U† diag(λ) U)")
            break
    else:
        print("No suitable FC group found.")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print("=" * 60)
    print("""
1. |ψ⟩ is a PURE STATE → density matrix has rank 1 always.
2. The unitary U is a basis change to the joint eigenbasis of
   all Pauli observables in the FC group.
3. U|ψ⟩ is still pure (unitary preserves purity).
4. Measurement of U|ψ⟩ gives a bitstring b with probability |⟨b|U|ψ⟩|².
5. Each column of eigenvalues[] maps bitstring b → ±1 for each observable.
6. This is mathematically equivalent to simultaneous diagonalization
   of commuting observables — nothing quantum-magical here.
7. The per-group cost is ONE unitary diagonalization of 2^k × 2^k matrix.
""")

if __name__ == "__main__":
    main()
