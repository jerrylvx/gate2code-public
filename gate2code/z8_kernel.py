"""Koh et al. Z₈-kernel method for logical diagonal gates (Koh et al. 2601.20927, App. H).

Given a k=3 CSS code described by a G-matrix (first k rows = X-logical operators,
remaining rows = X-stabilizer generators), this module determines whether there
exists *any* Γ ∈ Z₈ⁿ that simultaneously:

  1. Preserves the codespace (each stabilizer coset picks up the same phase):
       M · Γ ≡ 0  (mod 8)

  2. Implements the logical CCZ gate on the three encoded qubits:
       f_T(x₁,x₂,x₃) = 4·x₁·x₂·x₃  (mod 8)

The matrix M (Koh et al. App. H.2) arises from expanding the phase polynomial via
the Boolean XOR→product identity (Eq. D7 of Campbell–Howard 2017):

  Rows of M over Z₈:
    Type 1  –  g^j               for j ∈ {k+1,...,m}   (stabiliser rows)
    Type 2  –  2·(g^j ∧ g^l)     for j<l, NOT both in {1,...,k}
    Type 3  –  4·(g^j ∧ g^l ∧ g^r)  for j<l<r, NOT all three in {1,...,k}

  Γ ∈ ker(M mod 8) ⟺ Γ T^⊗n implements a valid logical diagonal gate.

The CCZ output condition is the additional linear system:
    g^j · Γ  ≡ 0 (mod 8)            for j = 0,1,2         (no linear logicals)
    (g^j ∧ g^l) · Γ ≡ 0 (mod 4)     for j<l, both in {0,1,2}  (no CZ logicals)
    (g^0 ∧ g^1 ∧ g^2) · Γ ≡ 1 (mod 2)                         (CCZ cubic = 4)

Solvability of the combined system is checked via Smith Normal Form over Z
(see `_snf_transforms`).  No enumeration of the (potentially huge) kernel is
needed — the method is purely algebraic and runs in polynomial time.

References
----------
Koh et al. 2025, arXiv:2601.20927, Appendix H ("Logical Clifford and magic diagonal
gates"), specifically §H.2 "Logical diagonal gates via products of single-qubit
Z-basis rotations".
"""

from __future__ import annotations

from itertools import combinations
from math import gcd

import numpy as np

from gate2code.gf2 import to_gf2
from gate2code.gmatrix import GInfo


# ---------------------------------------------------------------------------
# Smith Normal Form (integer, exact arithmetic)
# ---------------------------------------------------------------------------

def _snf_transforms(M_int: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Smith Normal Form: L · M_int · R = D (diagonal).

    Uses exact integer arithmetic (dtype=object) so there is no overflow.

    Parameters
    ----------
    M_int : ndarray, shape (r, n), integer dtype
        Input matrix.

    Returns
    -------
    D : ndarray, shape (r, n), dtype=object
        Diagonal matrix; D[i,i] are the invariant factors (≥0), sorted so
        D[0,0] | D[1,1] | ... | D[t-1,t-1], t = min(r,n).
    L : ndarray, shape (r, r), dtype=object
        Unimodular left transform; integer inverse of the left multiplier.
    R : ndarray, shape (n, n), dtype=object
        Unimodular right transform; integer inverse of the right multiplier.

    The relationship L @ M_int @ R == D holds exactly over Z.
    """
    r, c = M_int.shape
    D = np.array(M_int, dtype=object)
    L = np.eye(r, dtype=object)
    R = np.eye(c, dtype=object)

    def _swap_rows(A, i, j):
        tmp = A[i].copy(); A[i] = A[j]; A[j] = tmp

    def _swap_cols(A, i, j):
        tmp = A[:, i].copy(); A[:, i] = A[:, j]; A[:, j] = tmp

    for k in range(min(r, c)):
        while True:
            # --- find smallest nonzero entry in submatrix [k:, k:] ---
            bv, bi, bj = None, -1, -1
            for i in range(k, r):
                for j in range(k, c):
                    v = abs(int(D[i, j]))
                    if v > 0 and (bv is None or v < bv):
                        bv, bi, bj = v, i, j
            if bv is None:
                break  # remaining submatrix is zero

            # move pivot to (k, k)
            if bi != k:
                _swap_rows(D, k, bi); _swap_rows(L, k, bi)
            if bj != k:
                _swap_cols(D, k, bj); _swap_cols(R, k, bj)

            # make pivot positive
            if D[k, k] < 0:
                D[k, :] = -D[k, :]; L[k, :] = -L[k, :]

            pivot = int(D[k, k])

            # --- eliminate column below pivot ---
            col_dirty = False
            for i in range(k + 1, r):
                entry = int(D[i, k])
                if entry != 0:
                    q = entry // pivot
                    D[i, :] -= q * D[k, :]
                    L[i, :] -= q * L[k, :]
                    if int(D[i, k]) != 0:
                        _swap_rows(D, k, i); _swap_rows(L, k, i)
                        if D[k, k] < 0:
                            D[k, :] = -D[k, :]; L[k, :] = -L[k, :]
                        col_dirty = True
                        break
            if col_dirty:
                continue

            # --- eliminate row to the right of pivot ---
            pivot = int(D[k, k])
            row_dirty = False
            for j in range(k + 1, c):
                entry = int(D[k, j])
                if entry != 0:
                    q = entry // pivot
                    D[:, j] -= q * D[:, k]
                    R[:, j] -= q * R[:, k]
                    if int(D[k, j]) != 0:
                        _swap_cols(D, k, j); _swap_cols(R, k, j)
                        if D[k, k] < 0:
                            D[k, :] = -D[k, :]; L[k, :] = -L[k, :]
                        row_dirty = True
                        break
            if row_dirty:
                continue

            # --- ensure divisibility condition for SNF ---
            pivot = int(D[k, k])
            div_ok = True
            if pivot != 0:
                for i in range(k + 1, r):
                    if not div_ok:
                        break
                    for j in range(k + 1, c):
                        if int(D[i, j]) % pivot != 0:
                            # add row i to row k, then restart
                            D[k, :] += D[i, :]
                            L[k, :] += L[i, :]
                            div_ok = False
                            break
            if div_ok:
                break

    return D, L, R


# ---------------------------------------------------------------------------
# Build the Koh M-matrix over Z
# ---------------------------------------------------------------------------

def _build_M_matrix(G: np.ndarray, k: int) -> np.ndarray:
    """Build the integer constraint matrix M (Koh App. H.2).

    Parameters
    ----------
    G : ndarray, shape (m, n), binary (0/1) integer entries
        G-matrix: rows 0..k-1 are logical operators, rows k..m-1 are
        stabilizer generators.
    k : int
        Number of logical qubits.

    Returns
    -------
    M : ndarray, shape (num_rows, n), integer
        The constraint matrix whose Z₈-kernel characterises all valid Γ.
    """
    G = np.asarray(G, dtype=int)
    m, n = G.shape
    rows: list[np.ndarray] = []

    # --- Type 1: stabilizer rows (coefficient 1) ---
    for j in range(k, m):
        rows.append(G[j].copy())

    # --- Type 2: 2*(g^j ∧ g^l), not both indices in {0..k-1} ---
    for j in range(m):
        for l in range(j + 1, m):
            if j < k and l < k:
                continue  # both logical → skip (goes into f_T)
            rows.append(2 * (G[j] * G[l]))

    # --- Type 3: 4*(g^j ∧ g^l ∧ g^r), not all three in {0..k-1} ---
    for j in range(m):
        for l in range(j + 1, m):
            for r in range(l + 1, m):
                if j < k and l < k and r < k:
                    continue  # all logical → skip (goes into f_T)
                rows.append(4 * (G[j] * G[l] * G[r]))

    if not rows:
        return np.zeros((0, n), dtype=int)
    return np.array(rows, dtype=int)


# ---------------------------------------------------------------------------
# Linear-system solvability over Z/8Z
# ---------------------------------------------------------------------------

def _z8_solvable(M_int: np.ndarray, b: np.ndarray, modulus: int = 8) -> bool:
    """Return True iff M_int · Γ ≡ b (mod *modulus*) has a solution.

    Uses Smith Normal Form: L·M·R = D.  The system is solvable iff for each
    row *i* of D:  gcd(D[i,i], modulus)  divides  (L·b)[i] mod modulus.
    For rows *i* beyond the rank (D[i,i]=0): requires (L·b)[i] ≡ 0 (mod modulus).

    Parameters
    ----------
    M_int : ndarray, shape (r, n), integer
    b : ndarray, shape (r,), integer
        Right-hand side vector.
    modulus : int
        Usually 8 for the T-gate hierarchy.

    Returns
    -------
    bool
    """
    if M_int.shape[0] == 0:
        return True  # empty system is always solvable

    D, L, _ = _snf_transforms(M_int)
    r, n = M_int.shape

    Lb = np.array(
        [(L @ b.reshape(-1, 1)).flatten()[i] % modulus for i in range(r)],
        dtype=object,
    )

    t = min(r, n)
    for i in range(t):
        d = int(D[i, i]) % modulus
        c = int(Lb[i]) % modulus
        g = gcd(d, modulus) if d != 0 else modulus
        if c % g != 0:
            return False

    # Remaining rows (r > n): diagonal is 0 → need Lb[i] ≡ 0 (mod modulus)
    for i in range(t, r):
        if int(Lb[i]) % modulus != 0:
            return False

    return True


# ---------------------------------------------------------------------------
# CCZ solvability check (the main public function)
# ---------------------------------------------------------------------------

def supports_ccz_kernel(ginfo: GInfo) -> bool:
    """Return True iff there exists any Γ ∈ Z₈ⁿ implementing logical CCZ.

    Unlike ``supports_ccz()``, which only checks the uniform assignment
    Γ = **1**, this function checks whether *any* non-negative integer vector
    Γ ∈ {0,…,7}ⁿ (a product of T-gate powers) implements logical CCZ after
    Clifford correction.

    Specifically, it solves the combined Z₈ linear system:

        [M          ] · Γ ≡ [0]   (mod 8)
        [L_ccz_extra]       [b_ccz]

    where M enforces codespace preservation and L_ccz_extra + b_ccz enforce
    the CCZ logical output.  Requires k = 3.

    Parameters
    ----------
    ginfo : GInfo
        Must have k = 3.

    Returns
    -------
    bool
        True  → CCZ is achievable with *some* T-power assignment Γ ∈ Z₈ⁿ.
        False → no such Γ exists; the code cannot implement CCZ via any
                product of physical T-gate powers (even with Clifford correction).
    """
    G = np.asarray(ginfo.G, dtype=int)
    k = ginfo.k
    if k != 3:
        return False

    m, n = G.shape

    # Build codespace-preservation constraint matrix M
    M = _build_M_matrix(G, k)

    # Build CCZ output constraints:
    #   For k=3 canonical output f_T = 4·x₁x₂x₃:
    #     deg-1 coefficients = 0 mod 8  ⇒  g^j · Γ ≡ 0  (j=0,1,2)
    #     deg-2 coefficients = 0 mod 8  ⇒  2(g^j ∧ g^l) · Γ ≡ 0  for j<l in {0,1,2}
    #     deg-3 coefficient  = 4 mod 8  ⇒  4(g^0 ∧ g^1 ∧ g^2) · Γ ≡ 4
    extra_rows: list[np.ndarray] = []
    extra_rhs: list[int] = []

    # Degree-1: each logical row → 0
    for j in range(k):
        extra_rows.append(G[j].copy())
        extra_rhs.append(0)

    # Degree-2: each logical-pair → 0
    for j, l in combinations(range(k), 2):
        extra_rows.append(2 * (G[j] * G[l]))
        extra_rhs.append(0)

    # Degree-3: the unique logical triple → 4 (CCZ)
    triple = G[0] * G[1] * G[2]
    extra_rows.append(4 * triple)
    extra_rhs.append(4)

    E = np.array(extra_rows, dtype=int)
    b_E = np.array(extra_rhs, dtype=int)

    # Combine into one system
    if M.shape[0] > 0:
        M_total = np.vstack([M, E])
        b_total = np.concatenate([np.zeros(M.shape[0], dtype=int), b_E])
    else:
        M_total = E
        b_total = b_E

    return _z8_solvable(M_total, b_total, modulus=8)


# ---------------------------------------------------------------------------
# Enumerate all logical gate polynomials via kernel generators (small codes)
# ---------------------------------------------------------------------------

def enumerate_logical_polynomials(ginfo: GInfo) -> list[dict]:
    """Enumerate all logical diagonal gates for a k=3 code (brute-force, small n).

    WARNING: only feasible for n ≲ 20.  For each Γ ∈ Z₈ⁿ satisfying
    M·Γ ≡ 0 (mod 8), computes the logical polynomial f_T and returns a
    deduplicated list of achievable logical gates.

    This is an exponential-time function used for verification on small codes.
    Use ``supports_ccz_kernel`` for large codes.

    Returns
    -------
    list of dict, each with keys:
        'Gamma'   : np.ndarray  — the Γ vector
        'f_linear': list[int]   — [wt_Γ(g^j) mod 8 for j in 0..k-1]
        'f_quad'  : list[int]   — [2·wt_Γ(g^j∧g^l) mod 8 for j<l in 0..k-1]
        'f_cubic' : int         — 4·wt_Γ(g^0∧g^1∧g^2) mod 8
        'is_ccz'  : bool        — f_cubic==4 and all f_linear==f_quad==0
    """
    G = np.asarray(ginfo.G, dtype=int)
    k = ginfo.k
    if k != 3:
        raise ValueError("enumerate_logical_polynomials requires k=3")

    m, n = G.shape

    if n > 20:
        raise ValueError(f"n={n} too large for brute-force enumeration; use supports_ccz_kernel")

    M = _build_M_matrix(G, k)

    results: list[dict] = []
    seen_poly: set[tuple] = set()

    for bits in range(8**n):
        Gamma = np.array([(bits >> (3 * i)) & 7 for i in range(n)], dtype=int)
        # Check M · Gamma ≡ 0 (mod 8)
        if M.shape[0] > 0:
            if not np.all((M @ Gamma) % 8 == 0):
                continue
        # Compute f_T coefficients
        f_lin = [int((G[j] @ Gamma) % 8) for j in range(k)]
        f_quad = [
            int((2 * (G[j] * G[l]) @ Gamma) % 8)
            for j, l in combinations(range(k), 2)
        ]
        f_cub = int((4 * (G[0] * G[1] * G[2]) @ Gamma) % 8)

        poly_key = tuple(f_lin + f_quad + [f_cub])
        if poly_key in seen_poly:
            continue
        seen_poly.add(poly_key)

        results.append({
            "Gamma": Gamma.copy(),
            "f_linear": f_lin,
            "f_quad": f_quad,
            "f_cubic": f_cub,
            "is_ccz": (f_cub == 4 and all(x == 0 for x in f_lin) and all(x == 0 for x in f_quad)),
        })

    return results
