"""Lempel factorisation over GF(2).

Given a symmetric binary matrix Q, find the smallest binary matrix B
satisfying  Q = B · Bᵀ  (mod 2).

Based on:
    Lempel, A. (1975), SIAM Journal on Computing 4(2), 175.

The implementation is a direct translation of the Mathematica code in
*Synthillation2.nb* (Campbell & Howard).
"""

from __future__ import annotations

import numpy as np

from gate2code.gf2 import (
    GF2,
    GF2Array,
    inverse_mod2,
    is_full_rank,
    matrix_rank,
    null_space,
    smith_transform_mod2,
    to_gf2,
)


# ---------------------------------------------------------------------------
# Initial guesses
# ---------------------------------------------------------------------------


def lempel_guess(Q: GF2Array) -> GF2Array:
    """Lempel's original initial guess for the factorisation.

    Constructs a (possibly non-minimal) B such that Q = B·Bᵀ (mod 2).
    Worst-case column count is O(n²).
    """
    Q = np.asarray(Q, dtype=int)
    sz = Q.shape[0]
    assert np.array_equal(Q, Q.T), "Input not symmetric"

    columns: list[np.ndarray] = []

    # Diagonal part: if row-sum is odd, add the unit vector e_k
    for k in range(sz):
        if Q[k].sum() % 2 == 1:
            e = np.zeros(sz, dtype=int)
            e[k] = 1
            columns.append(e)

    # Off-diagonal part: for each (k, k2) with k2 < k, if Q[k, k2] == 1
    # add the vector e_k + e_k2
    for k in range(sz):
        for k2 in range(k):
            if Q[k, k2] == 1:
                v = np.zeros(sz, dtype=int)
                v[k] = 1
                v[k2] = 1
                columns.append(v)

    OUT = np.column_stack(columns) if columns else np.zeros((sz, 0), dtype=int)
    OUT_gf = to_gf2(OUT)

    # Verify
    check = np.asarray(OUT_gf @ OUT_gf.T, dtype=int)
    assert np.array_equal(check, Q), "LempelGuess: verification failed"

    return OUT_gf


def improved_guess(Q: GF2Array) -> GF2Array:
    """Improved initial guess (O(n) columns) from *Synthillation2.nb*.

    Uses a sequential construction that typically produces far fewer
    columns than :func:`lempel_guess`.
    """
    Q = np.asarray(Q, dtype=int)
    sz = Q.shape[0]
    assert np.array_equal(Q, Q.T), "Input not symmetric"

    # OUT is built as a list of rows (will be transposed later)
    # First row = first column of Q, with Q[0,0] forced to 1
    rows: list[np.ndarray] = []
    first_row = Q[:, 0].copy()
    first_row[0] = 1
    rows.append(first_row)

    for k in range(1, sz):
        new_row = np.zeros(sz, dtype=int)
        for j in range(sz):
            if j >= k:
                cross = sum(int(rows[s][j]) * int(rows[s][k]) for s in range(k))
                new_row[j] = (Q[j, k] + cross) % 2
        new_row[k] = 1  # set diagonal to 1
        rows.append(new_row)

    row_mat = np.array(rows, dtype=int)  # shape (sz, sz)

    # fixLinear: diagonal matrix of column-wise parity of row_mat
    col_parity = row_mat.sum(axis=0) % 2
    fix = np.diag(col_parity)

    # OUT = Transpose[row_mat] | fixLinear  (horizontal concat)
    OUT = np.hstack([row_mat.T, fix])
    OUT_gf = to_gf2(OUT)

    # Verify
    check = np.asarray(OUT_gf @ OUT_gf.T, dtype=int)
    assert np.array_equal(check, Q), "ImprovedGuess: verification failed"

    return OUT_gf


# ---------------------------------------------------------------------------
# Core Lempel algorithm
# ---------------------------------------------------------------------------


def lempel_reduce(Q: GF2Array, B_init: GF2Array) -> GF2Array:
    """Reduce B by stripping columns using null-space vectors.

    Direct translation of ``LempelReduce`` from Synthillation2.nb.
    """
    Q_int = np.asarray(Q, dtype=int)
    n_rows = Q_int.shape[0]
    B = np.asarray(B_init, dtype=int).copy()  # shape (n_rows, cols)
    zero_col = np.zeros(n_rows, dtype=int)
    MAX_ITER = 100_000

    def _right_null_space(M: np.ndarray) -> np.ndarray:
        """Right null space of M over GF(2): vectors u with M @ u = 0.

        Returns array of shape (dim_nullspace, ncols).
        """
        M_gf = to_gf2(M)
        # galois .null_space() returns the RIGHT null space:
        # rows u of length ncols(M) with M @ u^T = 0.
        ns = np.asarray(null_space(M_gf), dtype=int)
        if ns.ndim == 1:
            ns = (
                ns.reshape(1, -1)
                if ns.size > 0
                else np.zeros((0, M.shape[1]), dtype=int)
            )
        return ns

    def _filter_all_ones(ns: np.ndarray, length: int) -> np.ndarray:
        """Remove the all-ones vector from the null space basis."""
        if ns.shape[0] == 0:
            return ns
        all_ones = np.ones(length, dtype=int)
        mask = ~np.all(ns == all_ones, axis=1)
        return ns[mask]

    ns = _right_null_space(B)
    ns = _filter_all_ones(ns, B.shape[1])

    counter = 0
    while ns.shape[0] > 0 and counter < MAX_ITER:
        counter += 1
        u = ns[0].copy()

        # If weight of u is odd, pad B with a zero column and u with a 1
        if u.sum() % 2 == 1:
            u = np.append(u, 1)
            B = np.hstack([B, zero_col.reshape(-1, 1)])

        # Find first position with 0 and first position with 1
        pos0 = int(np.where(u == 0)[0][0])
        pos1 = int(np.where(u == 1)[0][0])

        # x = B[:, pos0] + B[:, pos1]  (mod 2)
        x = (B[:, pos0] + B[:, pos1]) % 2

        # B = (x ⊗ u + B) mod 2, where x ⊗ u is outer product
        B = (np.outer(x, u) + B) % 2

        # Remove the two columns at positions pos0 and pos1
        # (remove higher index first to keep lower index valid)
        cols_to_del = sorted([pos0, pos1], reverse=True)
        B = np.delete(B, cols_to_del, axis=1)

        # Recompute null space
        ns = _right_null_space(B)
        ns = _filter_all_ones(ns, B.shape[1])

    B_gf = to_gf2(B)

    # Verify
    check = np.asarray(B_gf @ B_gf.T, dtype=int)
    assert np.array_equal(
        check, Q_int
    ), f"LempelReduce: verification failed\nQ=\n{Q_int}\nB·Bᵀ=\n{check}"

    return B_gf


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def lempel(Q: GF2Array, method: int = 0) -> GF2Array:
    """Full Lempel factorisation: find smallest B with Q = B·Bᵀ (mod 2).

    Parameters
    ----------
    Q : GF2Array
        Symmetric binary matrix.
    method : int
        0 → use :func:`improved_guess` (default, usually faster);
        1 → use :func:`lempel_guess` (Lempel's original).

    Returns
    -------
    B : GF2Array
        Binary matrix satisfying Q = B · Bᵀ (mod 2), with (near-)minimal
        number of columns.
    """
    Q_int = np.asarray(Q, dtype=int)
    Q_gf = to_gf2(Q_int)
    r = matrix_rank(Q_gf)
    d = Q_gf.shape[0]

    # Edge case: zero matrix has trivial factorisation
    if r == 0:
        return to_gf2(np.zeros((d, 0), dtype=int))

    if r == d:
        # Full rank case
        guess = lempel_guess(Q_gf) if method == 1 else improved_guess(Q_gf)
        B = lempel_reduce(Q_gf, guess)
    else:
        # Rank-deficient: transform to full-rank sub-problem
        T = smith_transform_mod2(Q_gf)
        T_int = np.asarray(T, dtype=int)
        Q_transformed = to_gf2((T_int @ Q_int @ T_int.T) % 2)
        L = Q_transformed[:r, :r]

        guess = lempel_guess(L) if method == 1 else improved_guess(L)
        H = lempel_reduce(L, guess)

        # Reconstruct full B
        H_int = np.asarray(H, dtype=int)
        h_cols = H_int.shape[1]
        padded = np.vstack([H_int, np.zeros((d - r, h_cols), dtype=int)])
        T_inv = np.asarray(inverse_mod2(T), dtype=int)
        B = to_gf2((T_inv @ padded) % 2)

    # Final verification
    B_int = np.asarray(B, dtype=int)
    check = (B_int @ B_int.T) % 2
    assert np.array_equal(check, Q_int), "Lempel: final verification failed"

    # Optimality check
    delta = 0 if np.trace(Q_int) > 0 else 1
    if B.shape[1] > d + delta:
        import warnings

        warnings.warn(
            f"Lempel: output has {B.shape[1]} columns, "
            f"expected ≤ {d + delta} (rank={r}, δ={delta})"
        )

    return B


def bmat(A: GF2Array) -> GF2Array:
    """Compute B from A such that A·Aᵀ = B·Bᵀ (mod 2).

    This is the main convenience function: given a gate-synthesis matrix A,
    compute its associated Lempel factor B.
    """
    A_int = np.asarray(A, dtype=int)
    Q = to_gf2((A_int @ A_int.T) % 2)
    return lempel(Q, method=0)
