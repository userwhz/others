"""Count the number of measurement settings (groups) ShadowGrouping produces.

Usage:
    uv run python scripts/print_setting.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shadowgrouping import Shadow_Grouping, Bernstein_bound, char_to_int

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

HAM_FILES = [
    "hamiltonian_H2O_sto3g_14.json",
    "hamiltonian_BeH2_sto3g_14.json",
]

SHOTS = 25848


def load_hamiltonian(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def count_settings(ham: dict, label: str, mode: str) -> int:
    """Return num_unique_settings for the given mode."""
    pauli_strings = list(ham.keys())
    weights = np.array(list(ham.values()), dtype=float)
    obs = np.array([[char_to_int[c] for c in p] for p in pauli_strings], dtype=int)

    wf = Bernstein_bound(alpha=1)
    scheme = Shadow_Grouping(
        obs, weights, epsilon=0.1, weight_function=wf(),
        commutation_mode=mode,
    )

    # Run the greedy setting selection SHOTS times to count unique settings
    settings = []
    for _ in range(SHOTS):
        setting, _info = scheme.find_setting()
        settings.append(tuple(setting))

    unique_settings = len(set(settings))
    return unique_settings


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = Path(__file__).stem
    out_path = OUTPUT_DIR / f"{timestamp}.log"

    lines = [f"# ShadowGrouping setting count (shots={SHOTS})", ""]

    for name in HAM_FILES:
        path = DATA_DIR / name
        if not path.exists():
            lines.append(f"[SKIP] {name}: file not found")
            continue

        ham = load_hamiltonian(path)
        nqubit = len(next(iter(ham)))
        nterms = len(ham)
        lines.append(f"## {name}")
        lines.append(f"  qubits={nqubit}  terms={nterms}")

        for mode in ("qwc", "fc"):
            n_settings = count_settings(ham, name, mode)
            lines.append(f"  mode={mode}")
            lines.append(f"    unique_settings={n_settings}")
        lines.append("")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)

    print(text)


if __name__ == "__main__":
    main()
