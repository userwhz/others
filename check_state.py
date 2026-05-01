import numpy as np
import sys

path = "/home/biankaiming/shadowgrouping_beh2/haozhaowu/BeH2/hamil_class/state_BeH2_sto3g_14_vector.npy"
if len(sys.argv) > 1:
    path = sys.argv[1]

state = np.load(path)
print(f"file: {path}")
print(f"shape: {state.shape}")
print(f"dtype: {state.dtype}")
print(f"ndim:  {state.ndim}")

if state.ndim == 1:
    n = state.shape[0]
    nq = int(np.log2(n))
    norm_sq = np.real(np.vdot(state, state))
    print(f"\nqubits: {nq}  (dim={n})")
    print(f"type: state vector  ->  always PURE")
    print(f"⟨ψ|ψ⟩ = {norm_sq:.6f}")

elif state.ndim == 2:
    n = state.shape[0]
    nq = int(np.log2(n))
    # Purity = Tr(ρ²) without constructing a large matrix:
    # Tr(ρ²) = sum_i λ_i²  (eigenvalues squared)
    # For small matrices use trace, for large use eigh on a sub-block
    purity = np.real(np.trace(state @ state))
    print(f"\nqubits: {nq}  (dim={n})")
    print(f"type: density matrix")
    print(f"Tr(ρ²) = {purity:.6f}")
    if abs(purity - 1.0) < 1e-5:
        print("=> PURE state")
    else:
        print("=> MIXED state")
