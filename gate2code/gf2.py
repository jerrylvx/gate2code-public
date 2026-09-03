"""GF(2) matrix utilities wrapping the galois library.

Provides convenient constructors and linear-algebra helpers for binary
matrices over the field GF(2).
"""

from __future__ import annotations

from typing import Union

import galois
import numpy as np

# The Galois field with two elements
GF2 = galois.GF(2)

# Type alias for a GF(2) matrix
GF2Array = galois.FieldArray


def to_gf2(matrix: Union[np.ndarray, list]) -> GF2Array:
    """Convert a list / numpy array of 0s and 1s to a GF(2) FieldArray."""
    return GF2(np.asarray(matrix, dtype=int) % 2)


def rand_sym(n: int, *, rng: np.random.Generator | None = None) -> GF2Array:
    """Generate a random symmetric n×n binary matrix over GF(2).

    Reproduces the ``RandSym`` function from *Synthillation2.nb*:
    upper-triangular random bits, symmetrised, with an optional random
    diagonal added.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Upper-triangular random binary values (strict upper triangle)
    raw = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            raw[i, j] = rng.integers(0, 2)

    # Symmetrise: OUT = (raw + raw^T) mod 2
    out = (raw + raw.T) % 2

    # Optionally add a random diagonal
    if rng.integers(0, 2) == 1:
        diag = np.diag([rng.integers(0, 2) for _ in range(n)])
        out = (out + diag) % 2

    return to_gf2(out)


def matrix_rank(M: GF2Array) -> int:
    """Rank of a GF(2) matrix."""
    return int(np.linalg.matrix_rank(M))


def null_space(M: GF2Array) -> GF2Array:
    """Null space of a GF(2) matrix (rows are basis vectors)."""
    return M.null_space()


def is_full_rank(M: GF2Array) -> bool:
    """Check whether *M* has full row-rank over GF(2)."""
    return matrix_rank(M) == M.shape[0]


def smith_transform_mod2(Q: GF2Array) -> GF2Array:
    """Row-reduce *Q* over GF(2) and return the row-transformation matrix *T*.

    Computes *T* such that ``T @ Q`` is in row-echelon form over GF(2) and
    *T* is invertible mod 2.  This replaces the Mathematica
    ``SmithDecomposition`` used in ``LempelTransform``.

    For a field (GF(2) *is* a field) the Smith normal form is simply the
    reduced row-echelon form, so we perform Gaussian elimination and track
    the transformation.
    """
    n = Q.shape[0]
    A = to_gf2(np.array(Q, dtype=int))  # work copy
    T = to_gf2(np.eye(n, dtype=int))   # accumulates row ops

    pivot_row = 0
    for col in range(Q.shape[1]):
        # Find pivot
        found = None
        for row in range(pivot_row, n):
            if int(A[row, col]) == 1:
                found = row
                break
        if found is None:
            continue
        # Swap
        if found != pivot_row:
            A[[pivot_row, found]] = A[[found, pivot_row]]
            T[[pivot_row, found]] = T[[found, pivot_row]]
        # Eliminate below (and above for reduced form)
        for row in range(n):
            if row != pivot_row and int(A[row, col]) == 1:
                A[row] = A[row] + A[pivot_row]       # GF(2) addition
                T[row] = T[row] + T[pivot_row]
        pivot_row += 1

    return T


def inverse_mod2(M: GF2Array) -> GF2Array:
    """Matrix inverse over GF(2).  Raises ``ValueError`` if singular."""
    return np.linalg.inv(M)
