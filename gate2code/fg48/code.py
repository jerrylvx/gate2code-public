"""Code-level wrappers for the FG48 projective two-weight point set."""
from __future__ import annotations

import numpy as np

from gate2code.ccz import f2_rank
from gate2code.fg48.geometry import fg48_points
from gate2code.ks_fiber import projective_summary, span_weight_distribution_small


def int_to_bits(x: int, m: int) -> np.ndarray:
    """Convert integer vector to a length-m binary vector, LSB/e0 first."""
    if not (0 <= x < (1 << m)):
        raise ValueError("x is outside F_2^m")
    return np.array([(int(x) >> i) & 1 for i in range(m)], dtype=np.uint8)


def matrix_from_columns(cols: list[int], m: int) -> np.ndarray:
    """Build an m x n binary matrix from integer-encoded columns."""
    return np.asarray([int_to_bits(c, m) for c in cols], dtype=np.uint8).T


def generator_fg48() -> np.ndarray:
    """Return the 6 x 48 generator whose columns are the FG48 points."""
    return matrix_from_columns(fg48_points(), 6)


def code_weight_distribution(G: np.ndarray) -> dict[int, int]:
    """Enumerate the row-span weight distribution for a small binary generator."""
    we = span_weight_distribution_small(np.asarray(G, dtype=np.uint8) % 2, max_dim=20)
    if we is None:
        raise ValueError("generator has too many rows for exact enumeration")
    return we


def is_projective_generator(G: np.ndarray) -> bool:
    """Check that all columns are nonzero and distinct."""
    return bool(projective_summary(np.asarray(G, dtype=np.uint8) % 2)["projective"])


def row_space_equal(A: np.ndarray, B: np.ndarray) -> bool:
    """Check equality of row spaces over F_2 by rank comparison."""
    A = np.asarray(A, dtype=np.uint8) % 2
    B = np.asarray(B, dtype=np.uint8) % 2
    if A.shape[1] != B.shape[1]:
        return False
    return f2_rank(A.astype(np.int64)) == f2_rank(B.astype(np.int64)) == f2_rank(np.vstack([A, B]).astype(np.int64))
