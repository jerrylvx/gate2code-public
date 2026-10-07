"""Native T/T-dagger pattern analysis for one generator matrix G = [K; S] (rows as ints, n <= 64).

delta (bit j = 1 means T-dagger on column j) is native iff, for every row G_i and pair G_i&G_k,
  |delta & G_i|      = |G_i|/2      (mod 4)
  |delta & G_i & G_k| = |G_i & G_k|/2 (mod 2)
(together with the CH triple conditions, which do not involve delta). This is equivalent to
|c| - 2|delta & c| = 4 x1 x2 x3 (mod 8) for every codeword c = K^T x + S^T y (unique Z_8
multilinear expansion). All solutions are enumerated exactly: x0 + span(kernel), filtered by the
mod-4 conditions.
"""
from itertools import combinations, product
import numpy as np
from sc_lib import rref_int, reduce_int, popcount, nullspace_of_dot


def ch_ok(rows):
    r = len(rows)
    for t in (1, 2, 3):
        for sub in combinations(range(r), t):
            ov = -1
            for i in sub:
                ov &= rows[i]
            exp = 1 if sub == (0, 1, 2) else 0
            if popcount(ov) & 1 != exp:
                return False
    return True


def native_solutions(rows, n, max_dim=26):
    """Return (kernel_dim, array of all native deltas as uint64) or (kernel_dim, None) if too big;
    returns (-1, empty) if the mod-2 system is inconsistent."""
    r = len(rows)
    A = list(rows) + [rows[i] & rows[k] for i, k in combinations(range(r), 2)]
    b = [(popcount(v) // 2) & 1 for v in A]
    # solve A delta = b over F_2: augment with target bit at position n
    aug = [(v << 1) | bi for v, bi in zip(A, b)]   # rhs in bit 0, never a pivot unless v = 0
    ech = rref_int(aug)
    full = (1 << n) - 1
    x0 = 0
    for pv, e in ech:
        if pv == 0:
            return -1, np.zeros(0, dtype=np.uint64)
    # particular solution: pivot variable = rhs, free variables 0 (reduced echelon form)
    for pv, e in ech:
        if e & 1:
            x0 |= 1 << (pv - 1)
    ker = nullspace_of_dot([v & full for v in A], n)
    kd = len(ker)
    if kd > max_dim:
        return kd, None
    # enumerate all 2^kd
    ker_arr = np.array(ker, dtype=np.uint64)
    N = 1 << kd
    idx = np.arange(N, dtype=np.uint64)
    sols = np.full(N, np.uint64(x0))
    for j in range(kd):
        sel = ((idx >> np.uint64(j)) & np.uint64(1)).astype(bool)
        sols[sel] ^= ker_arr[j]
    ok = np.ones(N, dtype=bool)
    for v in rows:
        tgt = (popcount(v) // 2) % 4
        c = np.bitwise_count(sols & np.uint64(v)).astype(np.int64)
        ok &= (c % 4) == tgt
    # sanity: mod-2 constraints
    return kd, sols[ok]


def verify_native_bruteforce(rows, n, delta):
    """|c| - 2|delta & c| = 4 x1x2x3 mod 8 for all codewords (k = 3)."""
    K, S = rows[:3], rows[3:]
    s = len(S)
    Sspan = [0]
    for v in S:
        Sspan += [x ^ v for x in Sspan]
    for x in product((0, 1), repeat=3):
        base = 0
        for i in range(3):
            if x[i]:
                base ^= K[i]
        for y in Sspan:
            c = base ^ y
            if (popcount(c) - 2 * popcount(c & delta) - 4 * x[0] * x[1] * x[2]) % 8:
                return False
    return True


def distances(rows, n, points_cols):
    """d_X = min |K^T x + S^T y| (x != 0); d_Z = min |v| with S v = 0, K v != 0 (BFS)."""
    K, S = rows[:3], rows[3:]
    Sspan = [0]
    for v in S:
        Sspan += [x ^ v for x in Sspan]
    Sarr = np.array(Sspan, dtype=np.uint64)
    dx = n + 1
    for x in range(1, 8):
        base = 0
        for i in range(3):
            if (x >> i) & 1:
                base ^= K[i]
        dx = min(dx, int(np.bitwise_count(Sarr ^ np.uint64(base)).min()))
    labels = [sum(((K[i] >> j) & 1) << i for i in range(3)) for j in range(n)]
    cols = points_cols
    seen = {(0, 0)}
    frontier = [(0, 0)]
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for syn, lab in frontier:
            for j in range(n):
                st = (syn ^ cols[j], lab ^ labels[j])
                if st in seen:
                    continue
                if st[0] == 0 and st[1] != 0:
                    return dx, depth
                seen.add(st)
                nxt.append(st)
        frontier = nxt
    raise RuntimeError


def weight_enumerator(rows, n):
    span = [0]
    for v in rows:
        span += [x ^ v for x in span]
    w = np.bitwise_count(np.array(span, dtype=np.uint64))
    vals, cnts = np.unique(w, return_counts=True)
    return tuple((int(a), int(b)) for a, b in zip(vals, cnts))


# ---------------------------------------------------------------- large kernels (numba, n <= 64)
from numba import njit


@njit(cache=True)
def _popc(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def _gray_native(x0, basis, rows, targets, n, stop_first):
    """Histogram of |delta| over native deltas in x0 + span(basis); if stop_first, stop at first."""
    hist = np.zeros(n + 1, dtype=np.int64)
    kd = basis.shape[0]
    delta = x0
    best = np.uint64(0)
    bestw = n + 1
    total = np.int64(1) << kd
    for idx in range(total):
        if idx > 0:
            t = idx
            bit = 0
            while (t & 1) == 0:
                t >>= 1
                bit += 1
            delta ^= basis[bit]
        ok = True
        for i in range(rows.shape[0]):
            if (_popc(delta & rows[i]) & 3) != targets[i]:
                ok = False
                break
        if ok:
            w = _popc(delta)
            hist[w] += 1
            if w < bestw:
                bestw = w
                best = delta
            if stop_first:
                break
    return hist, best


def native_histogram(rows, n, stop_first=False, max_dim=34):
    """Exact: returns (kd, hist array or None, best delta). kd = -1 if the mod-2 system fails."""
    r = len(rows)
    A = list(rows) + [rows[i] & rows[k] for i, k in combinations(range(r), 2)]
    b = [(popcount(v) // 2) & 1 for v in A]
    ech = rref_int([(v << 1) | bi for v, bi in zip(A, b)])
    if any(pv == 0 for pv, _ in ech):
        return -1, None, None
    x0 = 0
    for pv, e in ech:
        if e & 1:
            x0 |= 1 << (pv - 1)
    ker = nullspace_of_dot(list(A), n)
    kd = len(ker)
    if kd > max_dim:
        return kd, None, None
    hist, best = _gray_native(np.uint64(x0), np.array(ker, dtype=np.uint64),
                              np.array(rows, dtype=np.uint64),
                              np.array([(popcount(v) // 2) % 4 for v in rows], dtype=np.int64),
                              n, stop_first)
    return kd, hist, int(best)


# ---------------------------------------------------------------- any n (multiword numba)
def _words(v, nw):
    return [(v >> (64 * k)) & 0xFFFFFFFFFFFFFFFF for k in range(nw)]


@njit(cache=True)
def _gray_native_mw(x0, basis, rows, targets, n, stop_first):
    nw = x0.shape[0]
    hist = np.zeros(n + 1, dtype=np.int64)
    kd = basis.shape[0]
    delta = x0.copy()
    best = x0.copy()
    bestw = n + 1
    total = np.int64(1) << kd
    for idx in range(total):
        if idx > 0:
            t = idx
            bit = 0
            while (t & 1) == 0:
                t >>= 1
                bit += 1
            for k in range(nw):
                delta[k] ^= basis[bit, k]
        ok = True
        for i in range(rows.shape[0]):
            c = 0
            for k in range(nw):
                c += _popc(delta[k] & rows[i, k])
            if (c & 3) != targets[i]:
                ok = False
                break
        if ok:
            w = 0
            for k in range(nw):
                w += _popc(delta[k])
            hist[w] += 1
            if w < bestw:
                bestw = w
                best[:] = delta
            if stop_first:
                break
    return hist, best


def native_histogram_any(rows, n, stop_first=False, max_dim=34):
    """As native_histogram, for any n (rows are Python ints)."""
    r = len(rows)
    A = list(rows) + [rows[i] & rows[k] for i, k in combinations(range(r), 2)]
    b = [(popcount(v) // 2) & 1 for v in A]
    ech = rref_int([(v << 1) | bi for v, bi in zip(A, b)])
    if any(pv == 0 for pv, _ in ech):
        return -1, None, None
    x0 = 0
    for pv, e in ech:
        if e & 1:
            x0 |= 1 << (pv - 1)
    ker = nullspace_of_dot(list(A), n)
    kd = len(ker)
    if kd > max_dim:
        return kd, None, None
    nw = (n + 63) // 64
    hist, best = _gray_native_mw(np.array(_words(x0, nw), dtype=np.uint64),
                                 np.array([_words(v, nw) for v in ker], dtype=np.uint64).reshape(kd, nw),
                                 np.array([_words(v, nw) for v in rows], dtype=np.uint64),
                                 np.array([(popcount(v) // 2) % 4 for v in rows], dtype=np.int64),
                                 n, stop_first)
    bi = sum(int(best[k]) << (64 * k) for k in range(nw))
    return kd, hist, bi


def distances_any(rows, n, cols):
    """d_X (all codewords, Python ints) and d_Z (BFS), any n."""
    K, S = rows[:3], rows[3:]
    Sspan = [0]
    for v in S:
        Sspan += [x ^ v for x in Sspan]
    dx = n + 1
    for x in range(1, 8):
        base = 0
        for i in range(3):
            if (x >> i) & 1:
                base ^= K[i]
        dx = min(dx, min(popcount(base ^ y) for y in Sspan))
    labels = [sum(((K[i] >> j) & 1) << i for i in range(3)) for j in range(n)]
    seen = {(0, 0)}
    frontier = [(0, 0)]
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for syn, lab in frontier:
            for j in range(n):
                st = (syn ^ cols[j], lab ^ labels[j])
                if st in seen:
                    continue
                if st[0] == 0 and st[1] != 0:
                    return dx, depth
                seen.add(st)
                nxt.append(st)
        frontier = nxt
    raise RuntimeError


# ---------------------------------------------------------------- very large kernels: constructive search
def native_isotropic_search(rows, n, tries=200, seed=0):
    """Constructive (one-sided) search for native deltas when the kernel is too large to enumerate.
    delta = y0 + t with y0 random in x0 + span(ker) and t in a random subspace I of span(ker) that is
    totally isotropic for every form B_i(e,f) = |e & f & G_i| mod 2; on y0 + I the mod-4 conditions are
    affine-linear, so they are solved exactly. Returns (kd, list of distinct native deltas found)
    (kd = -1 if the mod-2 system is inconsistent). Finding none is NOT a proof of non-existence."""
    import random as _r
    rng = _r.Random(seed)
    r = len(rows)
    A = list(rows) + [rows[i] & rows[k] for i, k in combinations(range(r), 2)]
    b = [(popcount(v) // 2) & 1 for v in A]
    ech = rref_int([(v << 1) | bi for v, bi in zip(A, b)])
    if any(pv == 0 for pv, _ in ech):
        return -1, []
    x0 = 0
    for pv, e in ech:
        if e & 1:
            x0 |= 1 << (pv - 1)
    ker = nullspace_of_dot(list(A), n)
    kd = len(ker)
    tg = [(popcount(v) // 2) % 4 for v in rows]
    found = set()

    def comb(basis):
        v = 0
        for x in basis:
            if rng.random() < 0.5:
                v ^= x
        return v
    for _ in range(tries):
        y0 = x0 ^ comb(ker)
        # greedy isotropic subspace inside span(ker)
        I = []
        cons = []
        while True:
            # subspace of span(ker) orthogonal to all cons vectors
            if cons:
                rws = []
                for c in cons:
                    rr = 0
                    for j, kv in enumerate(ker):
                        if popcount(kv & c) & 1:
                            rr |= 1 << j
                    rws.append(rr)
                loc = nullspace_of_dot(rws, kd)
                cand = []
                for v in loc:
                    w = 0
                    j = 0
                    while v:
                        if v & 1:
                            w ^= ker[j]
                        v >>= 1
                        j += 1
                    cand.append(w)
            else:
                cand = list(ker)
            ech_I = rref_int(I)
            cand = [c for c in cand if reduce_int(c, ech_I)]
            if not cand:
                break
            e = comb(cand)
            if reduce_int(e, ech_I) == 0:
                e = cand[rng.randrange(len(cand))]
            I.append(e)
            cons += [e & g for g in rows]
        # affine-linear system on y0 + span(I): L_i(t) = |t G_i|/2 + |y0 t G_i|  (mod 2)
        def Q(d):
            return [((popcount(d & g) - t) // 2) & 1 for g, t in zip(rows, tg)]
        rhs = Q(y0)
        mat = []
        for i, g in enumerate(rows):
            rr = 0
            for j, e in enumerate(I):
                val = ((popcount(e & g) // 2) + popcount(y0 & e & g)) & 1
                rr |= val << j
            mat.append((rr << 1) | rhs[i])
        ech2 = rref_int(mat)
        if any(pv == 0 for pv, _ in ech2):
            continue
        # particular + random homogeneous solution
        free_ech = [(pv - 1, e >> 1) for pv, e in ech2]
        piv = {pv: e for pv, e in free_ech}
        sol = 0
        # random free variables
        for j in range(len(I)):
            if j not in piv and rng.random() < 0.5:
                sol |= 1 << j
        for pv, e in ech2:
            j = pv - 1
            val = e & 1
            row = e >> 1
            val ^= popcount(row & sol & ~(1 << j)) & 1
            sol = (sol & ~(1 << j)) | (val << j)
        t = 0
        for j, e in enumerate(I):
            if (sol >> j) & 1:
                t ^= e
        d = y0 ^ t
        if all(((popcount(d & g) - tt) % 4) == 0 for g, tt in zip(rows, tg)):
            found.add(d)
    return kd, sorted(found, key=popcount)
