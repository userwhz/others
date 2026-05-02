"""Generate an n-qubit random pure state and save it to data/.

Usage:
    uv run python scripts/gen_random_state.py 4              # 4-qubit state
    uv run python scripts/gen_random_state.py 8 --seed 42   # reproducible
    uv run python scripts/gen_random_state.py 6 -o my_state.npy  # custom filename
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def random_state_vector(nqubit: int) -> np.ndarray:
    """Generate a Haar-random pure state of nqubit qubits.

    Samples from the standard complex normal distribution and normalizes.
    """
    dim = 2 ** nqubit
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    return psi / np.linalg.norm(psi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an n-qubit random pure state vector."
    )
    parser.add_argument(
        "nqubit", type=int,
        help="Number of qubits (state dimension = 2^n)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output filename (default: state_{nqubit}q.npy)."
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    psi = random_state_vector(args.nqubit)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    fname = args.output or f"state_{args.nqubit}q.npy"
    out_path = data_dir / fname

    np.save(out_path, psi)
    print(f"Saved {args.nqubit}-qubit state vector ({len(psi)} dim) to {out_path}")


if __name__ == "__main__":
    main()
