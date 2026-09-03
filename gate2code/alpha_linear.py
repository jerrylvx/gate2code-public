"""Linearized exhaustive decision of the CH alpha-layer on a fixed activity.

Pivot P2 ("last-label linearization"), generalized from the s=5 simplex
script.  For a fixed activity A (the active fibers), the nine CH conditions
split by degree in the labels alpha_i : A -> F_2:

  deg 0  (a),(d),(f)   hold by admissibility of A;
  deg 1  QT1,(b),(e)   force each alpha_i into the linear space
                       V = {x : <x, mu|_A> = 0, deg mu <= 2};
  deg 2  QT2,(c)       are linear in alpha_2 once alpha_1 is fixed;
  deg 3  QT3           and the pair conditions are affine-linear in alpha_3
                       once (alpha_1, alpha_2) are fixed.

So the full decision is: enumerate alpha_1 over V, alpha_2 over the
per-alpha_1 linear solution space, and check consistency of the affine
system for alpha_3 --- polynomial work per step, no CDCL, certificate-grade
either way.  Feasible when dim V <~ 20.

Batched over alpha_2 with numpy for throughput.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from gate2code.activity_spectrum import gf2_nullspace_basis

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                  # pragma: no cover
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=False)
    def _consistent_numba(M, rhs, d):
        """Batched GF(2) consistency, single-threaded compiled. Each augmented
        row [coeffs(d)|rhs] is packed into one uint64 (d+1<=64) and eliminated
        with XOR; a system is inconsistent iff some reduced row has zero coeffs
        but rhs=1. Bit-identical to _batched_consistent_unpacked. No prange:
        one thread per (1-CPU) task -- avoids BLAS-style oversubscription."""
        B = M.shape[0]
        R = M.shape[1]
        out = np.empty(B, np.bool_)
        rows = np.empty(R, np.uint64)
        one = np.uint64(1)
        dd = np.uint64(d)
        coeff_mask = (one << dd) - one
        rhs_bit = one << dd
        for b in range(B):
            for r in range(R):
                w = np.uint64(0)
                for c in range(d):
                    if M[b, r, c]:
                        w |= (one << np.uint64(c))
                if rhs[b, r]:
                    w |= rhs_bit
                rows[r] = w
            piv = 0
            for c in range(d):
                bitc = one << np.uint64(c)
                sel = -1
                for r in range(piv, R):
                    if rows[r] & bitc:
                        sel = r
                        break
                if sel < 0:
                    continue
                tmp = rows[piv]
                rows[piv] = rows[sel]
                rows[sel] = tmp
                pr = rows[piv]
                for r in range(R):
                    if r != piv and (rows[r] & bitc):
                        rows[r] ^= pr
                piv += 1
            bad = False
            for r in range(R):
                if (rows[r] & coeff_mask) == np.uint64(0) and \
                        (rows[r] & rhs_bit) != np.uint64(0):
                    bad = True
                    break
            out[b] = not bad
        return out

    @njit(cache=True, fastmath=False)
    def _decide_fused_numba(vmj, va1, R1pk, a2p, d, ns):
        """Fully bit-packed alpha_2 batch decision -- fuses R2, q3 and the
        consistency solve into one kernel, eliminating the numpy einsum / sum /
        concatenate. Per alpha_2 (packed over n cols in a2p[b]):
          R2[j,r] = parity(vmj[j,r] & a2),  q3[r] = parity(va1[r] & a2),
        and R1pk[j] are the precomputed packed-d R1 rows. The augmented system
        (R1 | R2 | q3=1) is eliminated over the d basis columns. Single-thread.
        Bit-identical to the einsum + _batched_consistent path."""
        B = a2p.shape[0]
        R = 2 * ns + 1
        out = np.empty(B, np.bool_)
        rows = np.empty(R, np.uint64)
        one = np.uint64(1)
        dd = np.uint64(d)
        coeff_mask = (one << dd) - one
        rhs_bit = one << dd
        for b in range(B):
            a2 = a2p[b]
            for j in range(ns):
                rows[j] = R1pk[j]
            for j in range(ns):
                w = np.uint64(0)
                for r in range(d):
                    x = vmj[j, r] & a2
                    x ^= x >> np.uint64(32)
                    x ^= x >> np.uint64(16)
                    x ^= x >> np.uint64(8)
                    x ^= x >> np.uint64(4)
                    x ^= x >> np.uint64(2)
                    x ^= x >> np.uint64(1)
                    if x & one:
                        w |= (one << np.uint64(r))
                rows[ns + j] = w
            w = np.uint64(0)
            for r in range(d):
                x = va1[r] & a2
                x ^= x >> np.uint64(32)
                x ^= x >> np.uint64(16)
                x ^= x >> np.uint64(8)
                x ^= x >> np.uint64(4)
                x ^= x >> np.uint64(2)
                x ^= x >> np.uint64(1)
                if x & one:
                    w |= (one << np.uint64(r))
            rows[2 * ns] = w | rhs_bit
            piv = 0
            for c in range(d):
                bitc = one << np.uint64(c)
                sel = -1
                for r in range(piv, R):
                    if rows[r] & bitc:
                        sel = r
                        break
                if sel < 0:
                    continue
                tmp = rows[piv]
                rows[piv] = rows[sel]
                rows[sel] = tmp
                pr = rows[piv]
                for r in range(R):
                    if r != piv and (rows[r] & bitc):
                        rows[r] ^= pr
                piv += 1
            bad = False
            for r in range(R):
                if (rows[r] & coeff_mask) == np.uint64(0) and \
                        (rows[r] & rhs_bit) != np.uint64(0):
                    bad = True
                    break
            out[b] = not bad
        return out


def _pack_rows(a: np.ndarray) -> np.ndarray:
    """Pack the last axis (<=64 bits) of a uint8 array into uint64 words."""
    a = np.ascontiguousarray(a, dtype=np.uint64)
    w = (np.uint64(1) << np.arange(a.shape[-1], dtype=np.uint64))
    return (a * w).sum(axis=-1).astype(np.uint64)


def _monomial_masks(active: list[int], s: int, max_deg: int) -> np.ndarray:
    """Rows: monomials of degree 0..max_deg evaluated on the active fibers."""
    bits = np.array([[(b >> (s - 1 - j)) & 1 for j in range(s)] for b in active],
                    dtype=np.uint8)
    rows = [np.ones(len(active), dtype=np.uint8)]
    for d in range(1, max_deg + 1):
        for idx in combinations(range(s), d):
            m = np.ones(len(active), dtype=np.uint8)
            for j in idx:
                m &= bits[:, j]
            rows.append(m)
    return np.array(rows, dtype=np.uint8)


def label_space(active: list[int], s: int, relaxed: bool = False) -> np.ndarray:
    """Basis of V (rows, over positions of `active`).

    relaxed=True drops QT1 (the degree-0 moment), modeling general
    (non-diagonal) Clifford corrections / the distillation-protocol class.
    """
    M = _monomial_masks(active, s, 2)
    if relaxed:
        M = M[1:]
    return gf2_nullspace_basis(M)


def _pair_rows(V: np.ndarray, g: np.ndarray, mus: np.ndarray) -> np.ndarray:
    """rows[j,k] = parity( V[k] & g & mus[j] ) -- the pair conditions vs g."""
    return ((V[None, :, :] & g[None, None, :] & mus[:, None, :]).sum(axis=2) % 2
            ).astype(np.uint8)


def _batched_consistent(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """For a batch of GF(2) systems (B,R,d) with rhs (B,R): solvable mask.

    Bit-packs each augmented row [coeffs(d)|rhs] into one uint64 (when d+1<=64,
    always true here since d=dim V_A <= |A| <= 48), so the per-column eliminate
    is a single uint64 XOR per row instead of d+1 uint8 ops -- ~2.8x faster,
    verified bit-identical to the unpacked form on 3000 random batteries.
    Falls back to the unpacked form if d+1>64."""
    B, R, d = M.shape
    if d + 1 > 64:
        return _batched_consistent_unpacked(M, rhs)
    if _HAVE_NUMBA:
        return _consistent_numba(np.ascontiguousarray(M, dtype=np.uint8),
                                 np.ascontiguousarray(rhs, dtype=np.uint8), int(d))
    aug = np.concatenate([M, rhs[:, :, None]], axis=2).astype(np.uint64)
    w = np.uint64(1) << np.arange(d + 1, dtype=np.uint64)
    P = (aug * w).sum(axis=2).astype(np.uint64)        # (B,R) packed rows
    row = np.zeros(B, dtype=np.int64)
    idx = np.arange(R)[None, :]
    for col in range(d):
        bit = np.uint64(1) << np.uint64(col)
        has = (P & bit) != 0
        cand = np.where(has & (idx >= row[:, None]), idx, R)
        piv = cand.min(axis=1)
        act = piv < R
        if not act.any():
            continue
        bsel = np.flatnonzero(act)
        pr = piv[bsel]; rr = row[bsel]
        tmp = P[bsel, pr].copy()
        P[bsel, pr] = P[bsel, rr]
        P[bsel, rr] = tmp
        sub = P[bsel]
        colvals = ((sub & bit) != 0)
        colvals[np.arange(len(bsel)), rr] = False      # don't eliminate the pivot
        pivot_rows = sub[np.arange(len(bsel)), rr]
        sub ^= colvals.astype(np.uint64) * pivot_rows[:, None]
        P[bsel] = sub
        row[bsel] += 1
    coeff_mask = np.uint64((1 << d) - 1)
    rhs_bit = np.uint64(1) << np.uint64(d)
    bad = (((P & coeff_mask) == 0) & ((P & rhs_bit) != 0)).any(axis=1)
    return ~bad


def _batched_consistent_unpacked(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Unpacked fallback (d+1>64). Original implementation."""
    B, R, d = M.shape
    aug = np.concatenate([M, rhs[:, :, None]], axis=2).astype(np.uint8)
    row = np.zeros(B, dtype=np.int64)
    for col in range(d):
        has = aug[:, :, col] == 1
        idx = np.arange(R)[None, :]
        cand = np.where(has & (idx >= row[:, None]), idx, R)
        piv = cand.min(axis=1)
        act = piv < R
        if not act.any():
            continue
        bsel = np.flatnonzero(act)
        pr = piv[bsel]
        rr = row[bsel]
        tmp = aug[bsel, pr, :].copy()
        aug[bsel, pr, :] = aug[bsel, rr, :]
        aug[bsel, rr, :] = tmp
        sub = aug[bsel]
        colvals = sub[:, :, col].copy()
        colvals[np.arange(len(bsel)), rr] = 0
        sub ^= colvals[:, :, None] & sub[np.arange(len(bsel)), rr, :][:, None, :]
        aug[bsel] = sub
        row[bsel] += 1
    coeff_zero = (aug[:, :, :d] == 0).all(axis=2)
    bad = (coeff_zero & (aug[:, :, d] == 1)).any(axis=1)
    return ~bad


def _beta_linear_span(active: list[int], s: int) -> np.ndarray:
    """Restrictions of beta-linear functionals (and 1) to the activity:
    the span of C2-rows + constant on A.  Used for the K-independence test:
    K rows are independent mod C2 iff no nonzero combination of the labels
    lies in this span."""
    bits = np.array([[(b >> (s - 1 - j)) & 1 for j in range(s)] for b in active],
                    dtype=np.uint8)
    return np.vstack([np.ones((1, len(active)), dtype=np.uint8), bits.T])


def _independent_labels(a1, a2, a3, span) -> bool:
    from gate2code.activity_spectrum import gf2_nullspace_basis  # local import
    base = span.copy()
    r0 = _rank(base)
    stacked = np.vstack([base, a1[None, :], a2[None, :], a3[None, :]])
    return _rank(stacked) == r0 + 3


def _rank(M) -> int:
    M = (np.asarray(M, dtype=np.uint8) % 2).copy()
    r = 0
    for c in range(M.shape[1]):
        p = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def _t_tensor_even(V: np.ndarray) -> bool:
    """True iff |b_j & b_k & b_l| is even for ALL basis triples of V
    (indices with repetition).  AND is bilinear over F_2, so this is
    equivalent to: every label triple from span(V) has even triple
    overlap -- QT3 is then unsatisfiable and the activity is unsat.
    See gate2code/trilinear_prefilter.py for derivation and gates."""
    rows = [int("".join("1" if b else "0" for b in row), 2) if row.any() else 0
            for row in V]
    r = len(rows)
    for j in range(r):
        for k_ in range(j, r):
            bjk = rows[j] & rows[k_]
            if bjk == 0:
                continue
            for l in range(k_, r):
                if bin(bjk & rows[l]).count("1") & 1:
                    return False
    return True


def decide_activity(active: list[int], s: int, k: int = 3,
                    a1_start: int = 1, a1_stop: int | None = None,
                    batch: int = 4096, relaxed: bool = False,
                    require_independent: bool = False,
                    prefilter: bool = True) -> dict:
    """Exhaustive linearized CH alpha-layer decision on the activity.

    Returns {"result": "sat"|"unsat", "dimV": ., "pairs_checked": .,
             "witness": (a1,a2,a3 truth tables) on sat}.
    Optional [a1_start, a1_stop) shards the alpha_1 Gray enumeration.
    """
    assert k == 3, "linearization implemented for k=3"
    n = len(active)
    V = label_space(active, s, relaxed=relaxed)
    d = V.shape[0]
    if d == 0:
        return {"result": "unsat", "dimV": 0, "pairs_checked": 0}
    if prefilter and not relaxed and _t_tensor_even(V):
        return {"result": "unsat", "dimV": d, "pairs_checked": 0,
                "prefilter": "t_tensor_even"}
    mus = _monomial_masks(active, s, 1)            # {1, y_1..y_s}: 1+s rows
    if relaxed:
        mus = mus[1:]                              # drop QT2 (constant row)
    ns = mus.shape[0]
    if _HAVE_NUMBA:
        # bit-packed (uint64 over the n cols) constants for the fused kernel;
        # vmj depends only on (V, mus), so it is built once for the whole decide.
        vr = _pack_rows(V)                         # (d,)  V[r] packed
        muj = _pack_rows(mus)                      # (ns,) mus[j] packed
        vmj = (vr[None, :] & muj[:, None]).astype(np.uint64)   # (ns, d)
    else:
        # loop-invariant for the numpy fallback path (feeds the R2 einsum)
        T = (V[None, :, :] & mus[:, None, :])      # (1+s, d, n)
    stop = a1_stop if a1_stop is not None else (1 << d)

    # Gray-walk alpha_1 over V
    cur = np.zeros(n, dtype=np.uint8)
    # advance to a1_start
    for i in range(1, a1_start):
        g = i ^ (i >> 1)
        b = (g ^ ((i - 1) ^ ((i - 1) >> 1))).bit_length() - 1
        cur ^= V[b]
    pairs = 0
    for i in range(max(a1_start, 1), stop):
        g = i ^ (i >> 1)
        b = (g ^ ((i - 1) ^ ((i - 1) >> 1))).bit_length() - 1
        cur ^= V[b]
        a1 = cur
        if not a1.any():
            continue
        R1 = _pair_rows(V, a1, mus)                # (1+s) x d
        sol2 = gf2_nullspace_basis(R1)
        d2 = sol2.shape[0]
        if d2 == 0:
            continue
        if _HAVE_NUMBA:
            a1p = _pack_rows(a1)
            va1 = (vr & a1p).astype(np.uint64)     # (d,)  V[r] & a1
            R1pk = _pack_rows(R1)                   # (ns,) packed-d R1 rows
        # all 2^d2 coordinate vectors in Gray-code order (vectorized; replaces
        # a Python loop that called np.eye(d2)[bb] millions of times)
        gj = np.arange(1 << d2, dtype=np.int64)
        gray = gj ^ (gj >> 1)
        coords = ((gray[:, None] >> np.arange(d2)[None, :]) & 1).astype(np.uint8)
        a2_all = (coords @ sol2 @ V) % 2           # (2^d2) x n
        a2_all = a2_all[a2_all.any(axis=1)]
        for lo in range(0, len(a2_all), batch):
            blk = a2_all[lo:lo + batch]            # B x n
            B = len(blk)
            if B == 0:
                continue
            pairs += B
            if _HAVE_NUMBA:
                ok = _decide_fused_numba(vmj, va1, R1pk, _pack_rows(blk),
                                         int(d), int(ns))
            else:
                # numpy fallback: R2 (B,1+s,d), q3 (B,d), batched consistency
                R2 = np.einsum("jdn,bn->bjd", T, blk) % 2
                q3 = ((V[None, :, :] & (a1[None, None, :] & blk[:, None, :])
                       ).sum(axis=2) % 2)          # (B, d)
                M = np.concatenate([np.broadcast_to(R1, (B,) + R1.shape),
                                    R2, q3[:, None, :]], axis=1).astype(np.uint8)
                rhs = np.zeros((B, M.shape[1]), dtype=np.uint8)
                rhs[:, -1] = 1
                ok = _batched_consistent(M, rhs)
            if ok.any():
                span = _beta_linear_span(active, s) if require_independent else None
                for bi in np.flatnonzero(ok):
                    a2 = blk[int(bi)]
                    if not require_independent:
                        return {"result": "sat", "dimV": int(d),
                                "pairs_checked": int(pairs),
                                "witness_a1": a1.tolist(),
                                "witness_a2": a2.tolist()}
                    # solve the alpha_3 system fully and look for an
                    # independent solution
                    R2 = _pair_rows(V, a2, mus)
                    q3v = ((V & (a1 & a2)[None, :]).sum(axis=1) % 2
                           ).astype(np.uint8)
                    Mat = np.vstack([R1, R2, q3v[None, :]])
                    rhsv = np.zeros(Mat.shape[0], dtype=np.uint8)
                    rhsv[-1] = 1
                    sol = _affine_solutions(Mat, rhsv)
                    if sol is None:
                        continue
                    part, null = sol
                    for ci in range(1 << null.shape[0]):
                        coef = np.array([(ci >> j) & 1
                                         for j in range(null.shape[0])],
                                        dtype=np.uint8)
                        a3 = ((part + (coef @ null if null.shape[0] else 0))
                              % 2 @ V) % 2
                        if _independent_labels(a1, a2, a3, span):
                            return {"result": "sat", "dimV": int(d),
                                    "pairs_checked": int(pairs),
                                    "witness_a1": a1.tolist(),
                                    "witness_a2": a2.tolist(),
                                    "witness_a3": a3.tolist(),
                                    "independent": True}
    return {"result": "unsat", "dimV": int(d), "pairs_checked": int(pairs)}


def _affine_solutions(A, b):
    """Particular solution + nullspace basis of Ax=b over F_2 (in coords)."""
    A = (np.asarray(A, dtype=np.uint8) % 2).copy()
    b = (np.asarray(b, dtype=np.uint8) % 2).copy()
    m, n = A.shape
    aug = np.hstack([A, b[:, None]])
    piv = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, m) if aug[i, c]), None)
        if p is None:
            continue
        aug[[r, p]] = aug[[p, r]]
        for i in range(m):
            if i != r and aug[i, c]:
                aug[i] ^= aug[r]
        piv.append(c)
        r += 1
    if any(aug[i, :n].sum() == 0 and aug[i, n] for i in range(m)):
        return None
    part = np.zeros(n, dtype=np.uint8)
    for i, c in enumerate(piv):
        part[c] = aug[i, n]
    free = [c for c in range(n) if c not in piv]
    null = np.zeros((len(free), n), dtype=np.uint8)
    for i, fc in enumerate(free):
        null[i, fc] = 1
        for j, c in enumerate(piv):
            null[i, c] = aug[j, fc]
    return part, null
