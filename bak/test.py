import numpy as np

suiji = 17
state = np.load(f"AlgeState/PauliAlgebraDensityState{suiji}.npy")
init_obs = np.load(f"AlgeH/PauliAlgebraDensityH{suiji}.npy")

print(np.trace(init_obs @ state))