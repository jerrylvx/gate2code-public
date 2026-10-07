"""Single-qubit transversal T/T-dagger implementations of logical CCZ.

For a full-rank CSS generator G = [K; S] with three logical rows, Definition 3.4
requires physical signs Gamma in {+1,-1}^n such that

    sw_Gamma(K^T x + S^T y) = 4 x1 x2 x3 (mod 8) for every x,y.

The paper's binary flip vector gamma specifies Gamma_j = (-1)^gamma_j.
The implementation calls that binary vector delta. For each row G_i and each
pair of distinct rows G_i,G_j, the signed-weight condition requires

    delta . G_i           = |G_i| / 2           (mod 4),
    delta . (G_i & G_j)   = |G_i & G_j| / 2     (mod 2),

along with the triple-overlap parity conditions. The nine Campbell-Howard (CH)
conditions alone allow a quasi-transversal implementation with a diagonal
Clifford correction and do not imply a single-qubit transversal implementation.

Existing API identifiers are retained. NoCorrResult.delta is the binary flip
vector gamma of the paper. NoCorrResult.gamma stores exponents 1 (T) and
7 (T-dagger), which equal the physical signs Gamma modulo 8.

Public API:
- no_correction(G, k=3): solve for a single-qubit transversal pattern.
- verify_no_correction(G, gamma): check the signed-weight equation on all codewords.
- accepts(G, level, k=3): test the selected implementation constraints.
- deformation_cost(G, k=3): compute diagonal-Clifford deformation data.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np

# Constraint levels (shared toggle across search drivers)
CH_CANONICAL = "ch_canonical"     # quasi-transversal CCZ with a diagonal Clifford correction
NO_CORRECTION = "no_correction"   # single-qubit transversal T/T-dagger implementation
Z8_KERNEL = "z8_kernel"           # CCZ via any Γ∈Z₈ (with-correction, permissive)


# ---------------------------------------------------------------------------
# GF(2) solver + mod-4 lift (verified core, moved from _search_gamma_identity_fast)
# ---------------------------------------------------------------------------
def gf2_rref(A: np.ndarray, b: np.ndarray):
    """Gauss-Jordan over GF(2) on [A | b]. Returns (rank, particular, basis):
    particular is one solution (None if infeasible); basis spans ker A."""
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    pivot_col = [-1] * m
    row = 0
    for col in range(n):
        if row >= m:
            break
        pr = None
        for r in range(row, m):
            if A[r, col] == 1:
                pr = r
                break
        if pr is None:
            continue
        if pr != row:
            A[[row, pr]] = A[[pr, row]]
            b[[row, pr]] = b[[pr, row]]
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] ^= A[row]
                b[r] ^= b[row]
        pivot_col[row] = col
        row += 1
    rank = row
    for r in range(rank, m):
        if b[r] == 1:
            return rank, None, None
    x = np.zeros(n, dtype=np.int8)
    for r in range(rank):
        x[pivot_col[r]] = b[r]
    free_cols = [c for c in range(n) if c not in pivot_col[:rank]]
    basis = []
    for fc in free_cols:
        h = np.zeros(n, dtype=np.int8)
        h[fc] = 1
        for r in range(rank):
            if A[r, fc] == 1:
                h[pivot_col[r]] = 1
        basis.append(h)
    basis = np.array(basis, dtype=np.int8) if basis else np.zeros((0, n), dtype=np.int8)
    return rank, x, basis


def check_mod4_lift(delta, lift_supports, lift_targets) -> bool:
    """Check the L_i mod-4 lift: e_2(δ|_supp) ≡ target mod 2 for each L_i."""
    for sup, tgt in zip(lift_supports, lift_targets):
        s = delta[sup].astype(int)
        kk = int(s.sum())
        e2 = (kk * (kk - 1) // 2) % 2
        if e2 != tgt:
            return False
    return True


# ---------------------------------------------------------------------------
# Mod-4 lift enumeration over the kernel — Numba-accelerated with a guarded
# pure-python fallback.  Gray-code traversal updates δ by a single basis-row XOR
# per step (O(n) instead of O(dim·n)) and avoids per-iteration allocation.
# The arithmetic is identical to check_mod4_lift, so results are bit-for-bit
# the same as the python loop (asserted in test_no_correction.py).
# ---------------------------------------------------------------------------
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*a, **k):
        def _wrap(f):
            return f
        return _wrap if (a and callable(a[0])) is False else a[0]


@njit(cache=True)
def _lift_enumerate_kernel(x0, basis, G, targets):  # pragma: no cover (jit)
    """Return (found, delta). Enumerates all 2^dim kernel offsets via Gray code,
    returns the first δ = x0 ⊕ (subset of basis rows) passing the mod-4 lift
    e_2(δ|_supp(G_i)) ≡ targets[i] for all rows i."""
    dim = basis.shape[0]
    n = x0.shape[0]
    r = G.shape[0]
    delta = x0.copy()
    prev_gray = 0
    total = 1 << dim
    for idx in range(total):
        gray = idx ^ (idx >> 1)
        if idx > 0:
            diff = gray ^ prev_gray  # exactly one bit set
            bit = 0
            d = diff
            while d > 1:
                d >>= 1
                bit += 1
            for j in range(n):
                delta[j] ^= basis[bit, j]
        prev_gray = gray
        ok = True
        for i in range(r):
            pc = 0
            for j in range(n):
                pc += delta[j] & G[i, j]
            if (((pc * (pc - 1) // 2) & 1)) != targets[i]:
                ok = False
                break
        if ok:
            return True, delta.copy()
    return False, delta


def _lift_enumerate(x0, basis, G, targets):
    """Find a kernel offset δ passing the mod-4 lift, or None. Uses Numba if
    available; the pure-python fallback is identical in semantics."""
    x0 = np.ascontiguousarray(x0, dtype=np.int64)
    basis = np.ascontiguousarray(basis, dtype=np.int64)
    G = np.ascontiguousarray(G, dtype=np.int64)
    targets = np.ascontiguousarray(targets, dtype=np.int64)
    if _HAS_NUMBA:
        found, delta = _lift_enumerate_kernel(x0, basis, G, targets)
        return delta.astype(np.int8) if found else None
    # pure-python fallback (Gray code)
    dim = basis.shape[0]
    n = x0.shape[0]
    delta = x0.copy()
    prev_gray = 0
    for idx in range(1 << dim):
        gray = idx ^ (idx >> 1)
        if idx > 0:
            diff = gray ^ prev_gray
            bit = (diff).bit_length() - 1
            delta ^= basis[bit]
        prev_gray = gray
        pc = G @ delta
        if np.all(((pc * (pc - 1) // 2) & 1) == targets):
            return delta.astype(np.int8)
    return None


def search_identity_fast(G: np.ndarray):
    """Search for a single-qubit transversal pattern; return (delta, gamma) if found,
    else a status string in {cubic_fail, parity_fail_L, parity_fail_Q, gf2_unsat,
    lift_unsat, lift_unsat_sampled}. Here delta is the binary flip vector and
    gamma contains exponents in {1,7} (1=T, 7=T-dagger). The mod-4 search is
    exhaustive for kernel dimension at most 28. Above that dimension, a bounded
    random sample is used; lift_unsat_sampled leaves existence unresolved.
    """
    r, n = G.shape

    # cond3 (cubic, Γ-free): K-triple → 1, every other triple → 0
    KIDX = (0, 1, 2)
    for i, j, k in combinations(range(r), 3):
        ov = int((G[i] & G[j] & G[k]).sum()) % 2
        req = 1 if (i, j, k) == KIDX else 0
        if ov != req:
            return 'cubic_fail'

    # cond4 (L_i mod-2 + mod-4 lift) + cond5 (Q_ij mod-2): GF(2) system on δ
    A_rows, b_rows = [], []
    L_lift_supports, L_lift_targets = [], []
    for i in range(r):
        v = G[i]
        wt = int(v.sum())
        if wt % 2 != 0:
            return 'parity_fail_L'
        A_rows.append(v.astype(np.int8))
        b_rows.append((wt // 2) % 2)
        L_lift_supports.append(np.where(v == 1)[0])
        L_lift_targets.append(((wt // 2) // 2) % 2)
    for i, j in combinations(range(r), 2):
        v_ij = (G[i] & G[j])
        wt = int(v_ij.sum())
        if wt % 2 != 0:
            return 'parity_fail_Q'
        A_rows.append(v_ij.astype(np.int8))
        b_rows.append((wt // 2) % 2)

    A = np.array(A_rows, dtype=np.int8)
    b = np.array(b_rows, dtype=np.int8)
    rank, x0, basis = gf2_rref(A, b)
    if x0 is None:
        return 'gf2_unsat'

    dim_ker = len(basis)
    targets = np.array(L_lift_targets, dtype=np.int64)
    if dim_ker > 28:
        # Too large to enumerate — bounded random sample (cap 2^20).
        for _ in range(1 << 20):
            coefs = np.random.randint(0, 2, size=dim_ker, dtype=np.int8)
            delta = (x0 + (coefs @ basis)) % 2
            if check_mod4_lift(delta, L_lift_supports, L_lift_targets):
                return delta, np.where(delta == 1, 7, 1)
        return 'lift_unsat_sampled'
    # Exhaustive mod-4 lift over the 2^dim_ker kernel (Numba Gray-code kernel).
    delta = _lift_enumerate(x0, basis, G.astype(np.int64), targets)
    if delta is not None:
        return delta, np.where(delta == 1, 7, 1)
    return 'lift_unsat'


# ---------------------------------------------------------------------------
# Public decision API
# ---------------------------------------------------------------------------
@dataclass
class NoCorrResult:
    """Result of the single-qubit transversal implementation check."""
    found: bool
    status: str                       # 'ok' or the failing-stage tag
    gamma: Optional[np.ndarray]       # shape (n,), entries in {1,7} (1=T, 7=T†)
    delta: Optional[np.ndarray]       # shape (n,), binary; gamma = 1 + 6*delta (0->1, 1->7)
    n_T: int
    n_Tdag: int


def no_correction(G: np.ndarray, k: int = 3) -> NoCorrResult:
    """Decide whether G admits a single-qubit transversal T/T-dagger implementation.

    The search checks triple-overlap parities, solves a binary linear system,
    and tests the mod-4 conditions. A gf2_unsat or lift_unsat status excludes a
    pattern for the supplied generator matrix. Kernel dimensions above 28 use
    bounded sampling: lift_unsat_sampled leaves existence unresolved.
    Returned gamma entries are
    exponents in {1,7}; returned delta entries are the paper's binary flip vector.
    """
    if k != 3:
        raise ValueError("C=I condition is defined for k=3 (CCZ).")
    G = np.asarray(G, dtype=int) % 2
    res = search_identity_fast(G)
    if isinstance(res, tuple):
        delta, gamma = res
        return NoCorrResult(True, 'ok', gamma, delta.astype(int),
                            int((gamma == 1).sum()), int((gamma == 7).sum()))
    return NoCorrResult(False, res, None, None, 0, 0)


def verify_no_correction(G: np.ndarray, gamma: np.ndarray, k: int = 3) -> bool:
    """Check the signed-weight equation on all 2^{k+s} codewords.

    The argument gamma contains exponents 1 (T) and 7 (T-dagger).
    The paper's physical signs are Gamma = +1 and -1, respectively.
    """
    G = np.asarray(G, dtype=int) % 2
    m, n = G.shape
    sign = np.where(np.asarray(gamma) == 7, -1, 1).astype(int)
    s = m - k
    for x_int in range(1 << k):
        x_bits = np.array([(x_int >> i) & 1 for i in range(k)], dtype=int)
        target = (4 * x_bits[0] * x_bits[1] * x_bits[2]) % 8
        for y_int in range(1 << s):
            y_bits = np.array([(y_int >> i) & 1 for i in range(s)], dtype=int)
            cw = (np.concatenate([x_bits, y_bits]) @ G) % 2
            if int(np.sum(sign * cw)) % 8 != target:
                return False
    return True


def accepts(G: np.ndarray, level: str = CH_CANONICAL, k: int = 3) -> bool:
    """Check the selected implementation constraints.

    CH_CANONICAL: the nine CH conditions for quasi-transversal CCZ.
    NO_CORRECTION: the CH conditions and a single-qubit transversal pattern.
    Z8_KERNEL: a CCZ implementation with physical exponents in Z_8.
    """
    from gate2code.ccz import verify_ch_conditions
    G = np.asarray(G, dtype=int) % 2
    if level == CH_CANONICAL:
        return bool(verify_ch_conditions(G, k=k)[0])
    if level == NO_CORRECTION:
        return bool(verify_ch_conditions(G, k=k)[0]) and no_correction(G, k=k).found
    if level == Z8_KERNEL:
        from gate2code.gmatrix import GInfo
        from gate2code.z8_kernel import supports_ccz_kernel
        return bool(supports_ccz_kernel(GInfo(G=G.astype(np.uint8), k=k)))
    raise ValueError(f"unknown constraint level: {level!r}")


# ---------------------------------------------------------------------------
# Deformation reachability (the "honest figure of merit")
# ---------------------------------------------------------------------------
@dataclass
class DeformationCost:
    """Diagonal-Clifford deformation data and single-qubit transversal status."""
    native_CI: bool          # single-qubit transversal implementation exists?
    n_S: int                 # S-gates in the diagonal Clifford D
    n_CZ: int                # CZ-gates in D
    err_count: int           # find_correction residual (0 = valid diagonal Clifford)
    added_Z_weight: int      # total Z-decoration weight on X-checks after deforming
    n_checks_decorated: int  # how many X-checks gain a Z-part
    css_survives: bool       # deformed code still CSS?  (decoration weight 0)


def deformation_cost(G: np.ndarray, k: int = 3) -> DeformationCost:
    """Report single-qubit transversal status and diagonal-Clifford deformation data.

    The correction calculation extracts a diagonal Clifford D (S and CZ gates),
    expands D to physical gates, and conjugates the X-stabilizers. The returned
    css_survives flag tests whether any X-stabilizer acquires a Z component.
    """
    from gate2code.correction import (find_gamma, find_correction,
                                       dual_basis, physical_diagonal_clifford)
    G = np.asarray(G, dtype=int) % 2
    m, n = G.shape
    nc = no_correction(G, k=k)
    native = nc.found

    # Prefer the no-correction Γ when it exists: then the deformation is trivial
    # (0 S, 0 CZ, stays CSS), the deformation cost for a single-qubit transversal implementation.
    if native:
        sigma = np.where(nc.gamma == 7, -1, 1).astype(int)
    else:
        sigma = find_gamma(G, k).gamma
    cr = find_correction(G, k, sigma)
    H = dual_basis(G).H
    D = physical_diagonal_clifford(cr.c_lin, cr.c_quad, H)

    A = np.zeros((n, n), dtype=int)
    for (j, power) in D.S_gates:
        A[j, j] ^= (power % 2)           # only odd S-power adds a Z-decoration
    for (j1, j2) in D.CZ_gates:
        A[j1, j2] ^= 1
        A[j2, j1] ^= 1
    X_stab = G[k:] % 2
    Zdec = (X_stab @ A) % 2
    n_dec = int(Zdec.any(axis=1).sum())
    added_w = int(Zdec.sum())
    return DeformationCost(
        native_CI=native,
        n_S=len(D.S_gates),
        n_CZ=len(D.CZ_gates),
        err_count=int(cr.err_count),
        added_Z_weight=added_w,
        n_checks_decorated=n_dec,
        css_survives=(added_w == 0),
    )
