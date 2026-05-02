# CLAUDE.md - ShadowGrouping Project

## Project Overview

This project implements the **ShadowGrouping** measurement scheme from [arXiv:2301.03385](https://arxiv.org/abs/2301.03385) — a method to efficiently estimate molecular energies on quantum computers by grouping Pauli observables into jointly-measurable sets, reducing the number of measurement rounds needed.

## Old Codebase (bak/ directory)

The `bak/` directory contains the original "legacy" codebase that benchmarked ShadowGrouping against several competing methods. It is essentially the reference implementation accompanying the paper, later extended with additional molecules and OGM (Overlapped Grouping Measurements) experiments.

### Architecture of bak/

The old code follows a layered architecture:

1. **Hamiltonian Layer** (`hamiltonian.py`): Loads molecular Hamiltonians as Pauli decompositions. Uses Qiskit Nature for fermion-to-qubit mappings (JW, BK, Parity). Ground states are loaded from precomputed `.npy` density matrices (traced back from classical diagonalization).

2. **Measurement Schemes** (`measurement_schemes.py`): A class hierarchy rooted at `Measurement_scheme`. Each subclass implements a strategy to select which Pauli measurement basis to use next. The key methods benchmarked were:
   - **ShadowGrouping** — the paper's primary contribution. A greedy algorithm that sorts observables by a weight function (typically Bernstein bound derived) and builds measurement settings by sequentially "hitting" as many high-weight observables as possible in one QWC/FC setting.
   - **Derandomization** — Huang et al.'s derandomization approach. Builds measurement settings qubit-by-qubit, choosing the Pauli assignment (X/Y/Z) that minimizes an inconfidence bound. Has an ε-greedy variant (RandomPaulis when δ=1).
   - **OverlappedGrouping (OGM / SettingSampler)** — Loads pre-optimized probability distributions over measurement settings (optimized offline in MATLAB via `OGM_optimization/`). In FC mode, groups are mutually commuting sets with optimized sampling probabilities.
   - **L1_sampler** — Naive L1-norm-weighted sampling of individual Pauli observables.
   - **AdaptiveShadows** — Hadfield et al.'s adaptive classical shadows, biasing the Pauli basis distribution per qubit based on weighted squared coefficients.
   - **GraphColoringGrouping** — Static greedy graph coloring on the non-commutation graph.
   - **AEQuO** — Bayesian-adaptive greedy bucket filling (arXiv:2110.15339).

3. **Energy Estimation** (`energy_estimator.py`): `Energy_estimator` wraps a measurement scheme + state sampler. It tracks running averages per observable and estimates energy as Σ w_i ⟨O_i⟩. Supports two measurement modes:
   - **QWC (qubit-wise commuting)**: Standard basis rotation + computational basis measurement.
   - **FC (fully commuting / joint measurement)**: Builds a unitary that simultaneously diagonalizes all observables in a group, then measures in the computational basis. Limited by `max_support_qubits` (default 8) since the unitary is 2^k × 2^k.

4. **Benchmark** (`benchmark.py`): `track_method_epsilon()` runs N_reps independent trials, each with geometrically-spaced shot budgets, and tracks provable and empirical energy estimation errors.

5. **OGM Optimization** (`OGM_optimization/` in bak): MATLAB code (by Bujiao Wu) that implements the OGM algorithm — builds overlapped commuting groups from Pauli Hamiltonian data and solves for an optimal probability distribution over groups. The Python `ogm_fc.py` reimplements this using SLSQP.

6. **Data Pipeline** (`bak/haozhaowu/`): Contains precomputed Hamiltonians, ground states, and OGM-optimized group probabilities for molecules (H2, LiH, H2O, BeH2) and random/spin Hamiltonians at various parameters. Helper scripts (`trans.py`, `trans_rho.py`, `convert_to_ogm.py`) handle format conversions.

### Key Design Patterns
- Measurement schemes track `N_hits` (how many times each observable has been "hit" by chosen settings). This is used by weight functions and truncation criteria.
- Truncation: Observables hit more than N_delta(δ) times are considered "saturated" and removed, introducing a systematic error bounded by their total |weight|.
- The commutation mode parameter (`"qwc"` vs `"fc"`) propagates throughout — ShadowGrouping, Derandomization, and SettingSampler all support both modes.

## Current src/ Structure

The `src/` directory has been cleaned to retain only the ShadowGrouping method:

- `measurement_schemes.py` — Helper functions + `Measurement_scheme` base class + `Shadow_Grouping`
- `energy_estimator.py` — `StateSampler` + `Energy_estimator`
- `weight_functions.py` — `Inconfidence_bound` and `Bernstein_bound` weight functions
- `hamiltonian.py` — Core Hamiltonian loading utilities
- `molecules.py` — Molecular geometry definitions
- `benchmark.py` — Benchmark utilities

## Environment Management

使用 **uv** 进行 Python 环境管理。`pyproject.toml` 定义了所有依赖。

```bash
# 初始化项目环境（创建 .venv 并安装所有依赖）
uv sync
```

## Testing

使用 **pytest** 作为测试框架，所有测试放在 `tests/` 目录。

```bash
# 运行全部测试
uv run pytest

# 显示详细输出
uv run pytest -v

# 运行单个测试文件
uv run pytest tests/test_hamiltonian.py

# 运行单个测试类或函数
uv run pytest tests/test_measurement_schemes.py::TestHitBy
```

### 测试结构

```
tests/
├── __init__.py
├── test_hamiltonian.py           # Hamiltonian, random_hamiltonian, char/int 映射
├── test_measurement_schemes.py   # hit_by, qwc_compatible, N_delta, Shadow_Grouping
├── test_weight_functions.py      # Bernstein_bound
└── test_shadowgrouping.py        # 端到端集成测试
```

### 测试规范

- 所有测试函数必须有类型提示（参数 + 返回值 `-> None`）
- 使用 `pytest.fixture` 管理共享的测试对象
- 使用 `pytest.mark.parametrize` 覆盖边界情况
- 测试类按被测模块组织（`TestHitBy`, `TestShadowGrouping` 等）

## Running the Code

- Python >= 3.9，Qiskit < 1.0
- Core dependencies: numpy, qiskit (qiskit-terra)
