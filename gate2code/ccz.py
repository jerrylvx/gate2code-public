"""
CCZ coset classification — shared utilities.

Utilities for the generator-column parity indicator and the CH conditions:
  1. Seed f0 construction
  2. RM(r-4, r) basis computation
  3. Coset element → G-matrix conversion
  4. CH condition verification
  5. Distance computation (dx, dz)
  6. Canonical form via pynauty graph isomorphism

Convention:
  - r variables; generator rows need not include the all-ones row
  - k=3 logical rows, s=r-3 stabiliser rows
  - Indicator f̄ : F₂ʳ → F₂,  f̄(a) = |{j : G_{·,j} = a}| mod 2
  - CCZ coset: f̄ ∈ f₀ + RM(3,r)⊥ = f₀ + RM(r−4, r)

Dependencies: numpy, galois (already in gate2code.gf2), pynauty (optional).
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from math import comb
from typing import Optional, Tuple, List, Dict, Any


# ─── F₂ linear algebra (fast, numpy-only) ────────────────────────────

def f2_rref(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Row-reduce M over F₂. Returns (rref_matrix, pivot_columns)."""
    A = M.copy().astype(np.int64)
    m, n = A.shape
    pivots: List[int] = []
    row = 0
    for col in range(n):
        found = -1
        for r in range(row, m):
            if A[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        A[[row, found]] = A[[found, row]]
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        pivots.append(col)
        row += 1
    return A[:row] % 2, pivots


def f2_solve(M: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solve Mx = b over F₂. Returns particular solution or None."""
    m, n = M.shape
    A = np.hstack([M.astype(np.int64), b.reshape(-1, 1).astype(np.int64)])
    pivots: List[int] = []
    row = 0
    for col in range(n):
        found = -1
        for r in range(row, m):
            if A[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        A[[row, found]] = A[[found, row]]
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        pivots.append(col)
        row += 1
    for r in range(row, m):
        if A[r, n] == 1:
            return None
    x = np.zeros(n, dtype=np.int64)
    for i, pc in enumerate(pivots):
        x[pc] = A[i, n]
    return x % 2


def f2_nullspace(M: np.ndarray) -> np.ndarray:
    """Null space of M over F₂. Returns (dim_ker × n) matrix of basis vectors."""
    m, n = M.shape
    A = M.copy().astype(np.int64)
    pivots: List[int] = []
    row = 0
    for col in range(n):
        found = -1
        for r in range(row, m):
            if A[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        A[[row, found]] = A[[found, row]]
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        pivots.append(col)
        row += 1
    pivot_set = set(pivots)
    free_vars = [c for c in range(n) if c not in pivot_set]
    basis = []
    for fv in free_vars:
        vec = np.zeros(n, dtype=np.int64)
        vec[fv] = 1
        for i, pc in enumerate(pivots):
            if A[i, fv] == 1:
                vec[pc] = 1
        basis.append(vec % 2)
    return np.array(basis, dtype=np.int64) if basis else np.zeros((0, n), dtype=np.int64)


def f2_inv(M: np.ndarray) -> Optional[np.ndarray]:
    """Invert a square F₂ matrix. Returns None if singular."""
    d = M.shape[0]
    aug = np.hstack([M.copy().astype(np.int64), np.eye(d, dtype=np.int64)])
    for col in range(d):
        found = -1
        for row in range(col, d):
            if aug[row, col] == 1:
                found = row
                break
        if found == -1:
            return None
        aug[[col, found]] = aug[[found, col]]
        for row in range(d):
            if row != col and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[col]) % 2
    return aug[:, d:] % 2


def f2_rank(M: np.ndarray) -> int:
    """Rank of M over F₂."""
    _, pivots = f2_rref(M)
    return len(pivots)


def f2_nullspace_dim(M: np.ndarray) -> int:
    """Dimension of ker(M) over F₂."""
    return M.shape[1] - f2_rank(M)


# ─── Bit-packed helpers ──────────────────────────────────────────────

def popcount(x: int) -> int:
    """Population count (Hamming weight of integer)."""
    return bin(x).count('1')


def vec_to_int(v) -> int:
    """Binary vector → integer (MSB first)."""
    out = 0
    for b in v:
        out = (out << 1) | int(b)
    return out


def int_to_vec(x: int, n: int) -> np.ndarray:
    """Integer → binary vector of length n (MSB first)."""
    return np.array([(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.int64)


# ─── Reed-Muller construction ────────────────────────────────────────

def f2_points(r: int) -> np.ndarray:
    """All 2^r points of F₂ʳ, shape (2^r, r), MSB-first ordering."""
    N = 2**r
    return np.array([[int(b) for b in format(v, f'0{r}b')] for v in range(N)],
                    dtype=np.uint8)


def monomial_eval_matrix(r: int, max_deg: int = 3) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """
    Evaluation matrix of all monomials of degree ≤ max_deg over F₂ʳ.

    Returns:
        M: shape (num_monomials, 2^r), M[i,v] = monomial_i(v)
        monos: list of tuples, each a sorted set of variable indices
               e.g. (0,1,2) means x₁x₂x₃
    """
    pts = f2_points(r)
    N = 2**r
    monos: List[Tuple[int, ...]] = []
    rows: List[List[int]] = []
    for deg in range(max_deg + 1):
        for idx_set in combinations(range(r), deg):
            monos.append(idx_set)
            row = np.ones(N, dtype=np.uint8)
            for i in idx_set:
                row = row & pts[:, i]
            rows.append(row.tolist())
    return np.array(rows, dtype=np.uint8), monos


def rm_basis(r: int, max_deg: int) -> np.ndarray:
    """
    Basis for RM(max_deg, r) as rows of (dim × 2^r) matrix over F₂.
    """
    M, _ = monomial_eval_matrix(r, max_deg)
    rref, pivots = f2_rref(M.astype(np.int64))
    return rref[:len(pivots)]


def rm_dim(s: int, m: int) -> int:
    """Dimension of RM(s, m)."""
    if s < 0:
        return 0
    if s >= m:
        return 2**m
    return sum(comb(m, i) for i in range(s + 1))


# ─── Seed f₀ construction ────────────────────────────────────────────

def build_seed_f0(r: int) -> np.ndarray:
    """
    Build seed f₀ ∈ F₂^{2^r} satisfying:
      ⟨f₀, x₁x₂x₃⟩ = 1
      ⟨f₀, m⟩ = 0  for all other monomials of degree ≤ 3

    Returns: binary vector of length 2^r.
    """
    M, monos = monomial_eval_matrix(r, max_deg=3)
    target = (0, 1, 2)  # x₁x₂x₃
    target_idx = monos.index(target)
    b = np.zeros(len(monos), dtype=np.int64)
    b[target_idx] = 1
    f0 = f2_solve(M.astype(np.int64), b)
    assert f0 is not None, f"No solution for seed f0 at r={r}"
    return f0


def build_seed_f0_int(r: int) -> int:
    """Seed f₀ as a packed integer (bit i = f₀(point_i))."""
    f0_vec = build_seed_f0(r)
    return vec_to_int(f0_vec)


# ─── CH condition verification ────────────────────────────────────────

def verify_ch_conditions(G: np.ndarray, k: int = 3) -> Tuple[bool, Optional[str]]:
    """
    Check all 9 CH-canonical conditions on G-matrix.
    G: (r × n) binary matrix, first k rows are logical.

    Returns: (pass, failing_condition) where failing_condition is
             'QT1','QT2','QT3','a','b','c','d','e','f' or None.
    """
    r, n = G.shape
    s = r - k
    K = G[:k]
    S = G[k:]

    # QT1: |K_i| even
    for i in range(k):
        if np.sum(K[i]) % 2 != 0:
            return False, 'QT1'

    # QT2: |K_i ∧ K_j| even
    for i, j in combinations(range(k), 2):
        if np.sum(K[i] & K[j]) % 2 != 0:
            return False, 'QT2'

    # QT3: |K_1 ∧ K_2 ∧ K_3| odd
    if k >= 3 and np.sum(K[0] & K[1] & K[2]) % 2 != 1:
        return False, 'QT3'

    # (a): |s_l| even
    for l in range(s):
        if np.sum(S[l]) % 2 != 0:
            return False, 'a'

    # (b): |K_i ∧ s_l| even
    for i in range(k):
        for l in range(s):
            if np.sum(K[i] & S[l]) % 2 != 0:
                return False, 'b'

    # (c): |K_i ∧ K_j ∧ s_l| even
    for i, j in combinations(range(k), 2):
        for l in range(s):
            if np.sum(K[i] & K[j] & S[l]) % 2 != 0:
                return False, 'c'

    # (d): |s_l ∧ s_m| even
    for l, m in combinations(range(s), 2):
        if np.sum(S[l] & S[m]) % 2 != 0:
            return False, 'd'

    # (e): |K_i ∧ s_l ∧ s_m| even
    for i in range(k):
        for l, m in combinations(range(s), 2):
            if np.sum(K[i] & S[l] & S[m]) % 2 != 0:
                return False, 'e'

    # (f): |s_l ∧ s_m ∧ s_p| even
    for l, m, p in combinations(range(s), 3):
        if np.sum(S[l] & S[m] & S[p]) % 2 != 0:
            return False, 'f'

    return True, None


# ─── Indicator function ↔ G-matrix conversion ────────────────────────

def indicator_to_gmatrix(f_vec: np.ndarray, r: int) -> Optional[np.ndarray]:
    """
    Convert indicator function f̄ (length 2^r binary vector) to the parent
    G-matrix. Columns are the points a ∈ F₂ʳ where f̄(a)=1.

    Returns: (r × n) binary matrix, or None if support is empty or rank-deficient.
    """
    pts = f2_points(r)
    support_mask = f_vec.astype(bool)
    n = int(np.sum(support_mask))
    if n == 0:
        return None
    cols = pts[support_mask]  # (n, r)
    G = cols.T.copy()  # (r, n)
    if f2_rank(G.astype(np.int64)) < r:
        return None
    return G.astype(np.int64)


def gmatrix_to_indicator(G: np.ndarray, r: int) -> np.ndarray:
    """
    Convert G-matrix to indicator function f̄ (length 2^r binary vector).
    """
    N = 2**r
    f = np.zeros(N, dtype=np.int64)
    for j in range(G.shape[1]):
        idx = vec_to_int(G[:, j])
        f[idx] = (f[idx] + 1) % 2
    return f


# ─── Distance computation ────────────────────────────────────────────

def compute_distances(G: np.ndarray, k: int = 3) -> Tuple[int, int]:
    """
    Compute (d_X, d_Z) for a G-matrix.

    d_X = min weight over {K·x + S·y : x ≠ 0} (X-logicals).
    d_Z = min weight over {v ∈ ker(S) \\ ker(G)} (Z-logicals).

    Uses:
      - Gray-code enumeration for d_X (O(2^r) time).
      - DP over F₂ʳ syndromes for d_Z (O(n · 2^r) time),
        avoiding the exponential enumeration of ker(G).
    """
    r, n = G.shape
    s = r - k
    K = G[:k]
    S = G[k:]

    # Pack rows as integers for fast XOR + popcount
    K_ints = [vec_to_int(K[i]) for i in range(k)]
    S_ints = [vec_to_int(S[i]) for i in range(s)]

    # ── d_X: min weight over {K·x + S·y : x ≠ 0} ──
    dx = n + 1
    for xi in range(1, 2**k):
        x_int = 0
        for j in range(k):
            if (xi >> (k - 1 - j)) & 1:
                x_int ^= K_ints[j]
        # Enumerate span(S) via Gray code
        current = x_int
        best = popcount(current)
        if s > 0:
            for si in range(1, 2**s):
                flip_bit = (si ^ (si - 1)).bit_length() - 1
                current ^= S_ints[flip_bit]
                w = popcount(current)
                if w < best:
                    best = w
        if best < dx:
            dx = best
            if dx <= 1:
                break

    # ── d_Z via DP over F₂ʳ syndromes ──
    #
    # Z-logical = vector v ∈ F₂ⁿ with S·v = 0 and K·v ≠ 0.
    # Each column G_j ∈ F₂ʳ has syndrome G_j itself under the parity-check G.
    # Choosing a subset T of columns, ⊕_{j∈T} G_j is the syndrome of the
    # indicator vector 1_T.  We want min |T| with K-part of syndrome ≠ 0
    # and S-part = 0.
    #
    # DP: dp[x] = minimum |T| such that ⊕_{j∈T} G_j = x, over x ∈ F₂ʳ.
    # Then dz = min over non-zero s ∈ F₂ᵏ of dp[(s << s_bits) | 0].
    #
    # This is the 0-1 knapsack over XOR, O(n · 2^r).

    N_states = 2**r
    INF = n + 1
    dp = [INF] * N_states
    dp[0] = 0

    # Compute column syndromes (each column IS its own syndrome)
    col_syns = []
    for j in range(n):
        syn = 0
        for i in range(r):
            syn = (syn << 1) | int(G[i, j])
        col_syns.append(syn)

    # 0-1 knapsack DP: process each column once
    for syn in col_syns:
        # Process in reverse to avoid using the same column twice
        # (standard 0-1 knapsack technique: new choices based on old dp)
        new_dp = dp.copy()
        for x in range(N_states):
            if dp[x] < INF:
                y = x ^ syn
                cand = dp[x] + 1
                if cand < new_dp[y]:
                    new_dp[y] = cand
        dp = new_dp

    # Extract dz: min dp[(s, 0)] over non-zero s ∈ F₂ᵏ
    # Syndrome layout: top k bits = K-syndrome, bottom s bits = S-syndrome
    dz = INF
    for s_val in range(1, 2**k):
        target = s_val << s  # K-part = s_val, S-part = 0
        if dp[target] < dz:
            dz = dp[target]

    return dx, dz


# ─── GL(n, F₂) enumeration ──────────────────────────────────────────

def gl_matrices(dim: int) -> List[np.ndarray]:
    """Enumerate all invertible dim×dim F₂ matrices."""
    if dim == 0:
        return [np.zeros((0, 0), dtype=np.int64)]
    mats = []
    for bits in range(2**(dim * dim)):
        M = np.array([(bits >> i) & 1 for i in range(dim * dim)],
                     dtype=np.int64).reshape(dim, dim)
        if f2_rank(M) == dim:
            mats.append(M)
    return mats


# ─── Canonical form via pynauty ──────────────────────────────────────

_PYNAUTY_AVAILABLE = False
try:
    import pynauty
    _PYNAUTY_AVAILABLE = True
except ImportError:
    pass


def indicator_canonical_form(f_int: int, r: int, k: int = 3) -> bytes:
    """
    Compute a canonical form for an indicator function using a bipartite
    graph encoding solved by nauty.

    Encoding:
      - r "row" vertices (first k colored K-type, next s colored S-type)
      - n "column" vertices (one per support point of indicator)
      - Bipartite edges: row i — col j  iff  G[i,j] = 1

    The automorphism group of this colored bipartite graph captures
    (row-permutation within K, row-permutation within S, column-permutation)
    equivalence.  This is a REFINEMENT of the true GL-equivalence: it may
    over-count classes (separate GL-equivalent codes as distinct) but never
    merges non-equivalent codes.  Safe for enumeration purposes.

    Returns: nauty canonical certificate (bytes), or a fallback hash if
             pynauty is unavailable.
    """
    if not _PYNAUTY_AVAILABLE:
        return _fallback_canonical(f_int, r, k)

    N = 2**r
    s = r - k

    # Extract support (column indices where indicator = 1)
    support = []
    for v in range(N):
        if (f_int >> (N - 1 - v)) & 1:
            support.append(v)
    n = len(support)

    if n == 0:
        return b'\x00'

    # Build bipartite graph: r row-vertices + n column-vertices
    # Vertices 0..r-1 are rows; vertices r..r+n-1 are columns
    total_v = r + n
    g = pynauty.Graph(total_v)

    # Extract G-matrix columns (inline bit decomposition, no numpy)
    for col_idx, pt_int in enumerate(support):
        for row in range(r):
            if (pt_int >> (r - 1 - row)) & 1:
                g.connect_vertex(row, [r + col_idx])

    # Coloring: 3 colors
    #   - K-rows (vertices 0..k-1)
    #   - S-rows (vertices k..r-1)
    #   - Columns (vertices r..r+n-1)
    coloring = []
    k_set = set(range(k))
    s_set = set(range(k, r))
    col_set = set(range(r, r + n))
    if k_set:
        coloring.append(k_set)
    if s_set:
        coloring.append(s_set)
    if col_set:
        coloring.append(col_set)
    g.set_vertex_coloring(coloring)

    return pynauty.certificate(g)


def _fallback_canonical(f_int: int, r: int, k: int) -> bytes:
    """Fallback canonical form when pynauty is unavailable.
    Uses weight profile as a cheap invariant (not canonical, just fingerprint)."""
    N = 2**r
    s = r - k
    profile = []
    for v in range(N):
        if (f_int >> (N - 1 - v)) & 1:
            k_part = v >> s
            s_part = v & ((1 << s) - 1)
            profile.append((popcount(k_part), popcount(s_part)))
    profile.sort()
    return str(profile).encode()


# ─── Coset enumeration (complete pipeline) ────────────────────────────

class CCZCoset:
    """
    Manages the coset f₀ + RM(r−4, r) for classification.

    Precomputes:
      - Seed f₀
      - RM basis (in coefficient and evaluation forms)
      - Optional: group actions for orbit decomposition
    """

    def __init__(self, r: int, k: int = 3):
        self.r = r
        self.k = k
        self.s = r - k
        self.N = 2**r

        # Build seed
        self.f0_vec = build_seed_f0(r)
        self.f0_int = vec_to_int(self.f0_vec)

        # Build RM basis
        self.basis = rm_basis(r, r - 4)  # (dim, 2^r)
        self.dim = self.basis.shape[0]
        self.coset_size = 2**self.dim

        # Pivot positions for coefficient extraction
        _, self.pivots = f2_rref(self.basis.copy())

        # Pack basis rows as integers
        self.basis_ints = [vec_to_int(self.basis[i]) for i in range(self.dim)]

    def coeffs_to_indicator_int(self, c: np.ndarray) -> int:
        """22-bit coefficient vector → packed indicator integer."""
        g_int = 0
        for i in range(self.dim):
            if c[i]:
                g_int ^= self.basis_ints[i]
        return self.f0_int ^ g_int

    def int_to_coeffs(self, idx: int) -> np.ndarray:
        """Coset index (0..2^dim-1) → coefficient vector."""
        return int_to_vec(idx, self.dim)

    def indicator_int(self, idx: int) -> int:
        """Coset index → packed indicator integer (no numpy, pure bit ops)."""
        g_int = 0
        for i in range(self.dim):
            if (idx >> (self.dim - 1 - i)) & 1:
                g_int ^= self.basis_ints[i]
        return self.f0_int ^ g_int

    def indicator_vec(self, idx: int) -> np.ndarray:
        """Coset index → indicator vector (length 2^r)."""
        f_int = self.indicator_int(idx)
        return int_to_vec(f_int, self.N)

    def indicator_weight(self, idx: int) -> int:
        """Weight of coset element at index idx."""
        return popcount(self.indicator_int(idx))

    def to_gmatrix(self, idx: int) -> Optional[np.ndarray]:
        """Coset index → parent G-matrix, or None if rank-deficient."""
        f_vec = self.indicator_vec(idx)
        return indicator_to_gmatrix(f_vec, self.r)

    def verify_element(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Verify CH conditions for coset element at index idx."""
        G = self.to_gmatrix(idx)
        if G is None:
            return False, 'rank'
        return verify_ch_conditions(G, self.k)

    def canonical_form(self, idx: int) -> bytes:
        """Canonical form of coset element for equivalence classification."""
        f_int = self.indicator_int(idx)
        return indicator_canonical_form(f_int, self.r, self.k)

    def full_analysis(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Complete analysis of a coset element:
        G-matrix, CH check, distances, canonical form.
        """
        G = self.to_gmatrix(idx)
        if G is None:
            return None

        ch_ok, ch_fail = verify_ch_conditions(G, self.k)
        if not ch_ok:
            return {'idx': idx, 'ch_ok': False, 'ch_fail': ch_fail}

        n = G.shape[1]
        dx, dz = compute_distances(G, self.k)
        d = min(dx, dz)

        return {
            'idx': idx,
            'n': n,
            'k': self.k,
            'd': d,
            'dx': dx,
            'dz': dz,
            'ch_ok': True,
            'G': G,
        }

    def enumerate_all(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Enumerate all coset elements, classify by canonical form,
        and compute distances for one representative per class.
        """
        import time
        t0 = time.time()

        if verbose:
            print(f"CCZ coset: r={self.r}, dim={self.dim}, "
                  f"|coset|={self.coset_size}")

        # Phase 1: partition by canonical form
        canon_to_indices: Dict[bytes, List[int]] = {}
        rank_deficient = 0

        for idx in range(self.coset_size):
            f_int = self.indicator_int(idx)
            wt = popcount(f_int)

            # Quick rank check: need at least r non-zero columns
            if wt < self.r:
                rank_deficient += 1
                continue

            cert = indicator_canonical_form(f_int, self.r, self.k)
            canon_to_indices.setdefault(cert, []).append(idx)

            if verbose and (idx + 1) % 100000 == 0:
                elapsed = time.time() - t0
                print(f"  {idx+1:,}/{self.coset_size:,} "
                      f"({100*(idx+1)/self.coset_size:.0f}%) "
                      f"classes={len(canon_to_indices)} [{elapsed:.1f}s]")

        if verbose:
            elapsed = time.time() - t0
            print(f"  Phase 1 done: {len(canon_to_indices)} classes, "
                  f"{rank_deficient} rank-deficient [{elapsed:.1f}s]")

        # Phase 2: analyze one representative per class
        results = []
        for i, (cert, indices) in enumerate(canon_to_indices.items()):
            rep_idx = indices[0]
            info = self.full_analysis(rep_idx)
            if info is not None:
                info['class_size'] = len(indices)
                info['class_id'] = i
                results.append(info)

            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Analyzed {i+1}/{len(canon_to_indices)} reps "
                      f"[{elapsed:.1f}s]")

        if verbose:
            elapsed = time.time() - t0
            print(f"  Phase 2 done: {len(results)} valid codes [{elapsed:.1f}s]")

        return results
