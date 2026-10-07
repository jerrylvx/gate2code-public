"""Shared machinery for classifying CH label maps on the subspace-complement support.

Support P = F_2^s minus U, U = {w = 0}, point x = u + (w << p) with u in F_2^p, w in F_2^q.
Point order: w = 1 .. 2^q - 1 ascending, then u ascending (matches data/family_witnesses).
Functions on P are Python ints (bit j = value at point j).
Group H = Stab_{GL(s,2)}(U); an element g is a tuple of s column images g(e_i) (s-bit ints).
Action on functions: (g.f)(g x) = f(x).
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


def popcount(x: int) -> int:
    return bin(x).count("1")


# ---------------------------------------------------------------- GF(2) helpers on int rows
def rref_int(rows):
    """Return list of (pivot_bit, row) in reduced echelon form (rows are ints)."""
    basis = []  # list of [pivot, row]
    for r in rows:
        for pv, b in basis:
            if (r >> pv) & 1:
                r ^= b
        if r:
            pv = r.bit_length() - 1
            for e in basis:
                if (e[1] >> pv) & 1:
                    e[1] ^= r
            basis.append([pv, r])
    return basis


def reduce_int(r, basis):
    for pv, b in basis:
        if (r >> pv) & 1:
            r ^= b
    return r


def rank_int(rows):
    return len(rref_int(rows))


def nullspace_of_dot(constraint_rows, n):
    """Basis (ints) of {v in F_2^n : popcount(v & c) even for all c}."""
    basis = rref_int(constraint_rows)
    piv = {pv: b for pv, b in basis}
    free = [j for j in range(n) if j not in piv]
    out = []
    for fcol in free:
        v = 1 << fcol
        for pv, b in piv.items():
            if (b >> fcol) & 1:
                v |= 1 << pv
        out.append(v)
    return out


def span_complement(sub_basis, full_basis):
    """Vectors of full_basis extending sub_basis to a basis of span(full)."""
    ech = rref_int(sub_basis)
    comp = []
    for v in full_basis:
        r = reduce_int(v, ech)
        if r:
            comp.append(v)
            ech = rref_int([b for _, b in ech] + [r])
    return comp


class Support:
    def __init__(self, p, q):
        self.p, self.q, self.s = p, q, p + q
        s = self.s
        self.points = [u | (w << p) for w in range(1, 1 << q) for u in range(1 << p)]
        self.n = n = len(self.points)
        self.index = {x: j for j, x in enumerate(self.points)}
        self.full = (1 << n) - 1
        self.coord = [self.func(lambda x, i=i: (x >> i) & 1) for i in range(s)]
        self.ones = self.full
        # RM(2,s)|_P generators
        rm1 = [self.ones] + self.coord
        rm2 = rm1 + [a & b for a, b in combinations(self.coord, 2)]
        self.rm1, self.rm2 = rm1, rm2
        self.VP = nullspace_of_dot(rm2, n)            # allowed single-label space
        self.C2 = list(self.coord)
        self.C2_ech = rref_int(self.C2)
        self.comp = span_complement(self.C2, self.VP)  # basis of V_P / C2
        self.dq = len(self.comp)
        # coordinates on V_P/C2: echelon of C2 + comp, record which comp each pivot row belongs to
        self._build_proj()
        self.gens = self._stab_generators()

    def func(self, f):
        v = 0
        for j, x in enumerate(self.points):
            if f(x):
                v |= 1 << j
        return v

    def _build_proj(self):
        """proj(v) for v in V_P: coordinates (int, dq bits) of v mod C2 in the comp basis."""
        n = self.n
        rows = self.C2 + self.comp
        m = len(rows)
        # track combinations: augment each row with an identity tag in high bits
        aug = [r | (1 << (n + i)) for i, r in enumerate(rows)]
        ech = []
        for r in aug:
            for pv, b in ech:
                if (r >> pv) & 1:
                    r ^= b
            low = r & self.full
            assert low, "dependent rows"
            pv = low.bit_length() - 1
            for e in ech:
                if (e[1] >> pv) & 1:
                    e[1] ^= r
            ech.append([pv, r])
        self._proj_ech = ech
        self._m = m

    def proj(self, v):
        """Coordinates of v (in V_P) modulo C2 w.r.t. self.comp, as int with dq bits."""
        n, c2 = self.n, len(self.C2)
        tag = 0
        for pv, b in self._proj_ech:
            if (v >> pv) & 1:
                v ^= b & self.full
                tag ^= b >> n
        assert v == 0, "vector not in V_P"
        return tag >> c2

    def lift(self, c):
        v = 0
        i = 0
        while c:
            if c & 1:
                v ^= self.comp[i]
            c >>= 1
            i += 1
        return v

    def in_C2(self, v):
        return reduce_int(v, self.C2_ech) == 0

    # ------------------------------------------------------------ group
    def _stab_generators(self):
        p, q, s = self.p, self.q, self.s
        ident = [1 << i for i in range(s)]
        gens = []

        def perm_gen(offset, k, kind):
            g = list(ident)
            if kind == "swap":
                g[offset], g[offset + 1] = ident[offset + 1], ident[offset]
            elif kind == "cycle":
                for i in range(k):
                    g[offset + i] = ident[offset + (i + 1) % k]
            elif kind == "transv":
                g[offset + 1] = ident[offset + 1] ^ ident[offset]
            return tuple(g)

        for off, k in ((0, p), (p, q)):
            if k >= 2:
                gens.append(perm_gen(off, k, "swap"))
                if k >= 3:
                    gens.append(perm_gen(off, k, "cycle"))
                gens.append(perm_gen(off, k, "transv"))
        g = list(ident)
        g[p] = ident[p] ^ ident[0]  # shear w-direction by a U vector
        gens.append(tuple(g))
        return gens

    def group_order(self):
        def gl(k):
            o = 1
            for i in range(k):
                o *= (1 << k) - (1 << i)
            return o
        return gl(self.p) * gl(self.q) * (1 << (self.p * self.q))

    @staticmethod
    def gmul(g, h, s):
        """(g h)(x) = g(h(x))."""
        return tuple(Support.gapply(g, hi) for hi in h)

    @staticmethod
    def gapply(g, x):
        y = 0
        i = 0
        while x:
            if x & 1:
                y ^= g[i]
            x >>= 1
            i += 1
        return y

    def ginv(self, g):
        s = self.s
        # solve via brute force over images: build inverse table
        inv = [0] * s
        table = {self.gapply(g, x): x for x in range(1 << s)}
        for i in range(s):
            inv[i] = table[1 << i]
        return tuple(inv)

    def perm(self, g):
        """perm[j] = index of g(x_j)."""
        return [self.index[self.gapply(g, x)] for x in self.points]

    def act(self, g, f, perm=None):
        if perm is None:
            perm = self.perm(g)
        out = 0
        j = 0
        while f:
            if f & 1:
                out |= 1 << perm[j]
            f >>= 1
            j += 1
        return out

    def quot_matrix(self, g):
        """Columns: images (dq-bit ints) of comp basis vectors under g, in quotient coords."""
        perm = self.perm(g)
        return [self.proj(self.act(g, b, perm)) for b in self.comp]


def mat_apply(cols, c):
    y = 0
    i = 0
    while c:
        if c & 1:
            y ^= cols[i]
        c >>= 1
        i += 1
    return y


def sym_perm_group(sup, gens):
    from sympy.combinatorics import Permutation, PermutationGroup
    return PermutationGroup([Permutation(sup.perm(g)) for g in gens])
