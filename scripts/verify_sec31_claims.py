#!/usr/bin/env python3
"""Machine-check the Section 3.1 claims as shipped in the 2026-07-27 rewrite (Gates f5dd8aa).

Checks, on the canonical full generator G = [K; S_FG48] (block order of eq:exp-blockform):

  1. distinct-columns guard (def:exp-indicator): the stabilizer columns are distinct
     (projectivity), and that alone forces the 48 columns of G to be distinct.
  2. bridge (eq:exp-bridge): for ALL 129 monomials of degree <= 3 on F_2^9, the
     row-overlap parity computed on the rows of G equals the pairing <chi_G, m>,
     and the value is 0 for every condition instance except <chi_G, x1x2x3> = 1 (K3).
  3. chi_S (eq:exp-chiS): the column set of S_FG48 is exactly {(u,w) : u in F_2^4,
     w in F_2^2, w != 0}, and the ANF of chi_S has weight 48 and degree 2.
  4. |K_T| parities: even for |T| = 1, 2 and odd for T = {1,2,3} (lem:exp-lift display).
  5. lem:exp-topcoeff: the coefficient of v_1...v_m equals the support parity, for the
     eight summand functions chi_S * prod_{i in T} kappa_i of eq:exp-graphind-expand
     and for 20 random Boolean functions on F_2^6.
  6. lem:exp-coset / eq:exp-fullcoset at chi_{G,0} = y_1...y_6: the defining property
     (only nonzero degree-<=3 inner product is against x1x2x3), the delta_0 coefficient
     equals n mod 2 = 0, and chi_G - chi_{G,0} lies in RM(5,9) (ANF degree <= 5).
  7. deg chi_G = 6 (lem:exp-lift).
  8. eq:exp-graphind extension independence: two random extensions of kappa off P give
     the same chi_G, equal to the direct column-set indicator.

Exit 0 iff every check passes. Prints one line per check plus a summary count.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gate2code.decreasing_monomial import anf_of_truth_table  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def col_int(col: np.ndarray) -> int:
    x = 0
    for bit in col:
        x = (x << 1) | int(bit)
    return x


def truth_table(cols: list[int], nvars: int) -> np.ndarray:
    f = np.zeros(1 << nvars, dtype=np.uint8)
    for c in cols:
        f[c] ^= 1  # XOR so duplicated columns would be caught by weight
    return f


def anf_deg(anf: np.ndarray) -> int:
    idx = np.nonzero(anf)[0]
    return max((bin(int(i)).count("1") for i in idx), default=-1)


def main() -> int:
    data = np.load(ROOT / "code_48_3_3_fg48.npz")
    Kc = data["K"].astype(np.uint8) % 2
    Sc = data["S_fg48"].astype(np.uint8) % 2
    S_fg48 = Sc
    G = np.vstack([Kc, Sc])
    n = G.shape[1]

    # --- 1. distinct columns from projectivity alone -------------------------
    betas = [col_int(c) for c in Sc.T]
    full = [col_int(c) for c in G.T]
    check("1a stabilizer columns distinct (projective)", len(set(betas)) == n)
    check("1b distinct betas force distinct G columns",
          len(set(betas)) == n and len(set(full)) == n)

    # --- chi_G truth table (row 0 = MSB: x1 x2 x3 y1..y6) --------------------
    chiG = truth_table(full, 9)
    check("0 chi_G weight = n = 48 (no duplicate columns)", int(chiG.sum()) == 48)

    # --- 2. bridge over all 129 degree-<=3 monomials --------------------------
    ok_eq, ok_val = True, True
    n_mon = 0
    for d in (1, 2, 3):
        for T in combinations(range(9), d):
            n_mon += 1
            rows_overlap = np.ones(n, dtype=np.uint8)
            for i in T:
                rows_overlap &= G[i]
            lhs = int(rows_overlap.sum()) & 1
            mask = sum(1 << (8 - i) for i in T)
            vals = [v for v in range(512) if (v & mask) == mask]
            rhs = int(chiG[vals].sum()) & 1
            expect = 1 if T == (0, 1, 2) else 0
            ok_eq &= (lhs == rhs)
            ok_val &= (rhs == expect)
    check("2a bridge identity holds for all monomials", ok_eq, f"{n_mon} monomials")
    check("2b all pairings 0 except K3 = 1", ok_val, f"{n_mon} monomials")

    # --- 2c-2e. the geometric reading of the bridge (Gates 2026-07-29) --------
    # Section 3.1 now reaches the pairing from Section 2's eq:exp-weight-section rather than
    # from scratch. Three claims underwrite that prose, one per axis of the specialization.
    #
    #   2c  RING + DEGREE 1.  Pairing chi_G against a general degree-1 function l_a(z) = a.z
    #       reproduces the Section 2 hyperplane count mod 2, for EVERY nonzero a, not just the
    #       coordinate ones. This is the precise sense in which hyperplanes are the special
    #       case: the CH conditions use only coordinate hyperplanes because they are stated on
    #       rows of G, not because the formalism cannot express the others.
    #   2d  Kurz's weight-multiplicity relation (arXiv:2112.11763v4,
    #       thm_correspondence_codes_multisets) in the lifted ambient, over Z:
    #       wt(c_a) = n - |P_G cap H_a|.
    #   2e  DEGREE d.  The value-1 set of a degree-d monomial, i.e. the columns lying off the d
    #       coordinate hyperplanes it names, is a coset of a codimension-d subspace.
    ints = [int(v) for v in range(512)]
    ok_2c = ok_2d = True
    for a in range(1, 512):
        wt = sum(1 for z in full if bin(a & z).count("1") & 1)      # columns off H_a
        sect = len(full) - wt                                        # columns on H_a
        pair = 0
        for z in ints:
            if chiG[z] and (bin(a & z).count("1") & 1):
                pair ^= 1
        ok_2c &= (pair == (wt & 1))
        ok_2d &= (wt == n - sect)
    check("2c degree-1 pairing = wt(c_a) mod 2, every nonzero a in F_2^9", ok_2c,
          "511 hyperplanes")
    check("2d wt(c_a) = n - |P_G cap H_a| over Z (Kurz relation, lifted)", ok_2d,
          "511 hyperplanes")

    ok_2e = True
    n_flat = 0
    for d in (1, 2, 3):
        for T in combinations(range(9), d):
            n_flat += 1
            mask = sum(1 << (8 - i) for i in T)
            ok_2e &= (sum(1 for v in ints if (v & mask) == mask) == 1 << (9 - d))
    check("2e value-1 set of a degree-d monomial is a codim-d coset", ok_2e,
          f"{n_flat} monomials")

    # --- 3. chi_S column set and ANF ------------------------------------------
    scols = set(betas)
    expected = {(u << 2) | w for u in range(16) for w in (1, 2, 3)}
    check("3a S_FG48 columns = {(u,w): w != 0} (u high bits, w low)",
          scols == expected)
    chiS = truth_table(betas, 6)
    anfS = anf_of_truth_table(chiS)
    check("3b chi_S weight 48, degree 2",
          int(chiS.sum()) == 48 and anf_deg(anfS) == 2,
          f"deg={anf_deg(anfS)}")

    # --- 4. |K_T| parities -----------------------------------------------------
    par = {}
    for d in (1, 2, 3):
        for T in combinations(range(3), d):
            v = np.ones(n, dtype=np.uint8)
            for i in T:
                v &= Kc[i]
            par[T] = int(v.sum()) & 1
    check("4 |K_T| parities even/even/odd",
          all(p == 0 for T, p in par.items() if len(T) < 3) and par[(0, 1, 2)] == 1,
          str(par))

    # --- 5. top-coefficient lemma ----------------------------------------------
    # summand functions on F_2^6: chi_S(beta) * prod_{i in T} kappa_i(beta)
    ok5 = True
    kappa = {b: col_int(c) for b, c in zip(betas, Kc.T)}
    for d in range(4):
        for T in combinations(range(3), d):
            g = np.zeros(64, dtype=np.uint8)
            for b in betas:
                val = 1
                for i in T:
                    val &= (kappa[b] >> (2 - i)) & 1
                g[b] ^= val
            anfg = anf_of_truth_table(g)
            top = int(anfg[63])
            ok5 &= (top == (int(g.sum()) & 1))
    rng = np.random.default_rng(20260727)
    for _ in range(20):
        g = rng.integers(0, 2, size=64).astype(np.uint8)
        anfg = anf_of_truth_table(g)
        ok5 &= (int(anfg[63]) == (int(g.sum()) & 1))
    check("5 lem:exp-topcoeff on 8 summands + 20 random g", ok5)

    # --- 6. coset lemma at chi_{G,0} = y1...y6 ----------------------------------
    chiG0 = np.zeros(512, dtype=np.uint8)
    for a in range(8):
        chiG0[(a << 6) | 63] = 1
    ok_def = True
    for d in (0, 1, 2, 3):
        for T in combinations(range(9), d):
            mask = sum(1 << (8 - i) for i in T)
            vals = [v for v in range(512) if (v & mask) == mask]
            ip = int(chiG0[vals].sum()) & 1
            expect = 1 if T == (0, 1, 2) else 0
            ok_def &= (ip == expect)
    check("6a chi_G0 = y1..y6 has the defining property (8-point support)",
          ok_def and int(chiG0.sum()) == 8)
    gdiff = (chiG ^ chiG0).astype(np.uint8)
    const_ip = int(gdiff.sum()) & 1
    anfd = anf_of_truth_table(gdiff)
    check("6b delta_0 coefficient = n mod 2 = 0 and chi_G - chi_G0 in RM(5,9)",
          const_ip == 0 and anf_deg(anfd) <= 5, f"deg diff={anf_deg(anfd)}")

    # --- 7. deg chi_G = 6 --------------------------------------------------------
    anfG = anf_of_truth_table(chiG)
    check("7 deg chi_G = 6", anf_deg(anfG) == 6, f"deg={anf_deg(anfG)}")

    # --- 8. extension independence of eq:exp-graphind ----------------------------
    tables = []
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        ext = {b: (kappa[b] if b in kappa else int(r.integers(0, 8)))
               for b in range(64)}
        f = np.zeros(512, dtype=np.uint8)
        for a in range(8):
            for b in range(64):
                prod = 1
                for i in range(3):
                    ai = (a >> (2 - i)) & 1
                    ki = (ext[b] >> (2 - i)) & 1
                    prod &= (1 ^ ai ^ ki)
                f[(a << 6) | b] = chiS[b] & prod
        tables.append(f)
    check("8 two random kappa extensions give the same chi_G = column indicator",
          np.array_equal(tables[0], tables[1]) and np.array_equal(tables[0], chiG))

    width = max(len(nm) for nm, _, _ in RESULTS)
    fails = 0
    for nm, ok, detail in RESULTS:
        print(f"{nm:<{width}}  {'PASS' if ok else 'FAIL'}  {detail}")
        fails += (not ok)
    print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} checks pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
