"""Data construction and I/O for the [[48,3,3]] CSS quasi-transversal CCZ code."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np


def _parse_row(s: str) -> np.ndarray:
    """Convert a space-separated binary string to a numpy array."""
    return np.array([int(c) for c in s.replace(" ", "")], dtype=int)


def build_48_code() -> dict:
    """Construct the [[48,3,3]] code matrices from hardcoded rows.

    Returns a dict with keys: G, K, S, n, k, s.
    """
    K = np.array([
        _parse_row("11111010 10011100 10110111 00111111 10101010 10001000"),
        _parse_row("00110101 00110101 11000110 01110001 01100011 10001110"),
        _parse_row("10110111 11010001 01010100 10100111 10110011 01001111"),
    ])
    S = np.array([
        _parse_row("00000000 00000000 11111111 11111111 11111111 11111111"),
        _parse_row("00000000 11111111 00000000 00000000 11111111 11111111"),
        _parse_row("00001111 00001111 00000000 11111111 00000000 11111111"),
        _parse_row("00110011 00110011 00001111 00001111 00001111 00001111"),
        _parse_row("11111111 00000000 00110011 00110011 00110011 00110011"),
        _parse_row("01010101 01010101 01010101 01010101 01010101 01010101"),
    ])
    G = np.vstack([K, S])
    return dict(G=G, K=K, S=S, n=48, k=3, s=6)


def enumerate_codewords(G: np.ndarray, k: int) -> dict:
    """Enumerate C1, C2, coset labels, and fiber order.

    Returns dict with keys: C1, C2, coset_lbl, fiber_order.
    """
    m = G.shape[0]
    C1_coeffs = list(product([0, 1], repeat=m))
    C1 = np.array([(np.array(c) @ G) % 2 for c in C1_coeffs])
    coset_lbl = np.array([list(c[:k]) for c in C1_coeffs])
    C2 = C1[np.all(coset_lbl == 0, axis=1)]
    fiber_order = list(product([0, 1], repeat=k))
    return dict(C1=C1, C2=C2, coset_lbl=coset_lbl, fiber_order=fiber_order)


def save_48_code(path: str | Path, **extra_arrays) -> None:
    """Save code arrays to an npz file.

    Parameters
    ----------
    path : file path for the npz.
    **extra_arrays : additional arrays to include (e.g. C1, coset_lbl).
    """
    code = build_48_code()
    np.savez(
        path,
        G=code["G"],
        K=code["K"],
        S=code["S"],
        **extra_arrays,
    )


def load_48_code(path: str | Path) -> dict:
    """Load code arrays from an npz file.

    Returns dict with at least G, K, S, n, k, s.
    """
    data = dict(np.load(path, allow_pickle=False))
    G = data["G"]
    data["n"] = G.shape[1]
    data["k"] = 3
    data["s"] = G.shape[0] - 3
    return data
