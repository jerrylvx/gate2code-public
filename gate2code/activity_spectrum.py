"""Activity-spectrum machinery for the beta-fiber CCZ search.

The pure-S CH conditions (a),(d),(f) say exactly that the fiber-activity
indicator f : F_2^s -> F_2 (with f(0)=0, the origin fiber being forbidden by
the weight-1 obstruction) is orthogonal to every monomial of degree 1..3 in
the s stabilizer variables.  The span of those monomials is a codimension-1
subspace of RM(3,s) (missing the constant), whose dual is

    RM(s-4, s) + <delta_0>,

where delta_0 is the indicator of the origin (delta_0 has zero moments of all
degrees >= 1).  Consequently the admissible activity sets at level s are

    branch E ("even"):   f = c        with c in RM(s-4,s), c(0)=0,  n = wt(c)
    branch P ("puncture"): f = c + delta_0 with c in RM(s-4,s), c(0)=1, n = wt(c)-1

This quantizes the possible code lengths n to the weight spectrum of
RM(s-4,s) (branch E) union that spectrum shifted by -1 (branch P).  The
archived e1 [[48,3,3]] sits in branch E (activity = complement of a *linear*
solid in F_2^6); the Jacinto [[47,3,3]] sits in branch P (complement of an
*affine* solid, punctured at the origin).

Validated against: e1, Jacinto D3 (zero moment violations each), and the
2026-06 cluster window scan (solver effort concentrates exactly on the
allowed n).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def monomials_deg_1_to_3(s: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for d in range(1, 4):
        out.extend(combinations(range(s), d))
    return out


def moment_matrix(s: int) -> np.ndarray:
    """Rows: monomials of degree 1..3 in s vars, evaluated on all 2^s points.

    Point x uses MSB-first bit convention: bit j of x is (x >> (s-1-j)) & 1,
    matching the column convention of the G-matrices in this repo.
    """
    pts = np.array([[(x >> (s - 1 - j)) & 1 for j in range(s)]
                    for x in range(1 << s)], dtype=np.uint8)
    rows = []
    for idx in monomials_deg_1_to_3(s):
        mono = np.ones(1 << s, dtype=np.uint8)
        for j in idx:
            mono &= pts[:, j]
        rows.append(mono)
    return np.array(rows, dtype=np.uint8)


def gf2_nullspace_basis(M: np.ndarray) -> np.ndarray:
    """Basis (rows) of ker(M) over F_2."""
    M = (np.asarray(M, dtype=np.uint8) % 2).copy()
    m, n = M.shape
    pivots: list[int] = []
    row = 0
    for col in range(n):
        sel = None
        for r in range(row, m):
            if M[r, col]:
                sel = r
                break
        if sel is None:
            continue
        M[[row, sel]] = M[[sel, row]]
        for r in range(m):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        pivots.append(col)
        row += 1
    free = [c for c in range(n) if c not in pivots]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, fc in enumerate(free):
        basis[i, fc] = 1
        for prow, pcol in enumerate(pivots):
            basis[i, pcol] = M[prow, fc]
    return basis


def activity_space_basis(s: int) -> np.ndarray:
    """Basis of the admissible-activity ambient space RM(s-4,s) + <delta_0>."""
    return gf2_nullspace_basis(moment_matrix(s))


def is_valid_activity(f: np.ndarray, s: int) -> bool:
    """Zero degree-1..3 moments and f(0)=0."""
    f = np.asarray(f, dtype=np.uint8) % 2
    if f.shape != (1 << s,) or f[0]:
        return False
    return not (moment_matrix(s) @ f % 2).any()


def activity_of_S(S: np.ndarray) -> np.ndarray:
    """Activity indicator (length 2^s) of a stabilizer block S (s x n)."""
    S = np.asarray(S, dtype=np.uint8) % 2
    s = S.shape[0]
    f = np.zeros(1 << s, dtype=np.uint8)
    for col in S.T:
        b = 0
        for bit in col:
            b = (b << 1) | int(bit)
        f[b] = 1
    return f


def enumerate_weight_table(s: int, max_dim: int = 24) -> dict[str, dict[int, int]]:
    """Exact branch-resolved weight table by enumerating the activity space.

    Feasible for s <= 6 (space dimension = dim RM(s-4,s) + 1 <= 23).
    Returns {"E": {n: count}, "P": {n: count}} where n is the code length and
    count the number of admissible activity sets realizing it.
    """
    basis = activity_space_basis(s)
    dim = basis.shape[0]
    if dim > max_dim:
        raise ValueError(f"activity space dim {dim} too large to enumerate")
    n_pts = 1 << s
    tabE: dict[int, int] = {}
    tabP: dict[int, int] = {}
    cur = np.zeros(n_pts, dtype=np.uint8)
    prev_gray = 0
    for idx in range(1, 1 << dim):
        gray = idx ^ (idx >> 1)
        bit = (gray ^ prev_gray).bit_length() - 1
        cur ^= basis[bit]
        prev_gray = gray
        w = int(cur.sum())
        if cur[0] == 0:
            tabE[w] = tabE.get(w, 0) + 1
        else:
            # f = cur + delta_0 is the admissible activity, n = w - 1
            tabP[w - 1] = tabP.get(w - 1, 0) + 1
    return {"E": dict(sorted(tabE.items())), "P": dict(sorted(tabP.items()))}


def allowed_n(s: int) -> list[int]:
    """Sorted list of admissible code lengths at level s (n >= 1)."""
    tab = enumerate_weight_table(s)
    ns = {n for branch in tab.values() for n in branch if n >= 1}
    return sorted(ns)


def dx_feasible(n: int, s: int, k: int = 3) -> bool:
    """Z-check column-locking bound (necessary for d_X >= 3 when the
    X-stabilizer code has minimum weight >= 3, e.g. triply-even spines).

    d_X >= 3 forces the Z-check matrix (generator of C1-perp, dimension
    n-k-s) to have n distinct nonzero columns: a zero column j yields the
    weight-1 X-logical e_j, equal columns i,j yield the weight-2 X-logical
    e_i + e_j.  Hence n <= 2^(n-k-s) - 1.  Dual to the beta-fiber collapse
    (d_Z >= 3 => n <= 2^s - 1): both CSS distances at level 3 are
    column-locking conditions, one per check matrix.
    """
    r = n - k - s
    return r >= 0 and n <= (1 << r) - 1
