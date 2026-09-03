#!/usr/bin/env python3
"""Machine-check eq:exp-graphind-expand (eq. 12, Lemma 3.4) and tabulate the
label-map and label-monomial coefficient polynomials of Q48.

Coordinates.  A point of F_2^9 is (alpha, u, w) with alpha in F_2^3 the logical
label, u in F_2^4 along the deleted subspace P, and w in F_2^2 the coset label.
Integer encoding, MSB first: alpha_1 alpha_2 alpha_3 u_1 u_2 u_3 u_4 w_1 w_2.
This is the row order of the canonical generator G = [K; S_FG48], so the truth
table of chi_G is read straight off the columns of G.

Checks:
  A  bit convention self-test (ANF index bit j is variable bit j).
  B  chi_S(w) = w_1 + w_2 + w_1 w_2, weight 48, degree 2 (eq:exp-chiS).
  C  the per-coset label map printed in eq:exp-labelmap agrees term by term
     with kappa read off the columns of K.
  D  eq. 12: the RHS sum over T of chi_S * prod_{i in T} kappa_i *
     prod_{i not in T} (1 + alpha_i) equals chi_G on all 512 points.
  E  deg chi_G = 6, and the alpha-free top monomial u1u2u3u4w1w2 has
     coefficient 1 (Lemma 3.4's conclusion).
  F  label-monomial coefficients: chi_G = sum_A alpha^A g_A(u,w) with
     g_A = chi_S * prod_{i not in A} (1 + kappa_i), checked against the ANF of
     chi_G regrouped by its alpha part.
  G  fiber indicators chi_a(u,w) for the eight labels a in F_2^3, with weights
     summing to 48 and |chi_0| odd (which is why the top monomial survives).

Exit 0 iff every check passes.
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

# variable names, MSB first, for the 9-variable index
VARS9 = ["a1", "a2", "a3", "u1", "u2", "u3", "u4", "w1", "w2"]
VARS6 = ["u1", "u2", "u3", "u4", "w1", "w2"]


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def col_int(col: np.ndarray) -> int:
    x = 0
    for bit in col:
        x = (x << 1) | int(bit)
    return x


def monomial(idx: int, names: list[str]) -> str:
    r = len(names)
    parts = [names[r - 1 - j] for j in range(r) if (idx >> j) & 1]
    parts.reverse()
    return "".join(parts) if parts else "1"


def poly_str(anf: np.ndarray, names: list[str]) -> str:
    terms = [monomial(int(i), names) for i in np.nonzero(anf)[0]]
    terms.sort(key=lambda t: (0 if t == "1" else len(t), t))
    return " + ".join(terms) if terms else "0"


def anf_deg(anf: np.ndarray) -> int:
    idx = np.nonzero(anf)[0]
    return max((bin(int(i)).count("1") for i in idx), default=-1)


def main() -> int:
    data = np.load(ROOT / "code_48_3_3_fg48.npz")
    Kc = data["K"].astype(np.uint8) % 2
    Sc = data["S_fg48"].astype(np.uint8) % 2
    G = np.vstack([Kc, Sc])
    n = G.shape[1]

    # ---- A. bit convention self-test ------------------------------------
    probe = np.array([(b >> 5) & 1 for b in range(64)], dtype=np.uint8)  # u1
    probe_anf = anf_of_truth_table(probe)
    check("A ANF index bit j is variable bit j",
          np.array_equal(np.nonzero(probe_anf)[0], np.array([32])),
          f"u1 -> monomial index {list(np.nonzero(probe_anf)[0])}")

    # ---- point data ------------------------------------------------------
    betas = [col_int(c) for c in Sc.T]          # (u,w) as 6-bit int, u high
    labels = [col_int(c) for c in Kc.T]         # alpha as 3-bit int
    assert len(set(betas)) == n

    chiS6 = np.zeros(64, dtype=np.uint8)
    for b in betas:
        chiS6[b] ^= 1
    anfS = anf_of_truth_table(chiS6)
    check("B chi_S = w1 + w2 + w1w2, weight 48, degree 2",
          poly_str(anfS, VARS6) == "w1 + w2 + w1w2"
          and int(chiS6.sum()) == 48 and anf_deg(anfS) == 2,
          f"chi_S = {poly_str(anfS, VARS6)}")

    # kappa_i as functions on F_2^6, extended by 0 off P (w = 0)
    kap = np.zeros((3, 64), dtype=np.uint8)
    for b, a in zip(betas, labels):
        for i in range(3):
            kap[i, b] = (a >> (2 - i)) & 1

    # ---- C. eq:exp-labelmap, coset by coset ------------------------------
    # paper text, transcribed from sections/expedited_criterion.tex:493-502
    paper = {
        (1, 0): ["u1 + u3",
                 "1 + u1 + u2 + u3 + u4",
                 "1 + u1 + u1u2 + u1u4 + u3u4"],
        (0, 1): ["1 + u3",
                 "1 + u1 + u3 + u1u3 + u1u4",
                 "u1 + u1u2 + u2u3 + u2u4 + u3u4"],
        (1, 1): ["u4",
                 "u3 + u4 + u1u3 + u1u4",
                 "u1 + u3 + u1u4 + u2u3 + u2u4"],
    }
    ok_c, detail_c = True, []
    for (w1, w2), rows in paper.items():
        wbits = (w1 << 1) | w2
        for i in range(3):
            tt = np.array([kap[i, (u << 2) | wbits] for u in range(16)],
                          dtype=np.uint8)
            got = poly_str(anf_of_truth_table(tt), ["u1", "u2", "u3", "u4"])
            want = " + ".join(sorted(rows[i].replace(" ", "").split("+"),
                                     key=lambda t: (0 if t == "1" else len(t), t)))
            ok_c &= (got == want)
            if got != want:
                detail_c.append(f"w=({w1},{w2}) kappa_{i+1}: got {got}, paper {want}")
    check("C eq:exp-labelmap matches kappa on all three cosets", ok_c,
          "; ".join(detail_c) if detail_c else "9 polynomials, 9 agree")

    # ---- lift everything to F_2^9 ----------------------------------------
    idx = np.arange(512)
    lo6 = idx & 63
    chiS9 = chiS6[lo6]
    kap9 = np.stack([kap[i][lo6] for i in range(3)])
    alpha9 = np.stack([(idx >> (8 - i)) & 1 for i in range(3)]).astype(np.uint8)

    chiG = np.zeros(512, dtype=np.uint8)
    for b, a in zip(betas, labels):
        chiG[(a << 6) | b] ^= 1
    check("D0 chi_G weight = n = 48", int(chiG.sum()) == 48)

    # ---- D. equation 12 ---------------------------------------------------
    rhs = np.zeros(512, dtype=np.uint8)
    for d in range(4):
        for T in combinations(range(3), d):
            term = chiS9.copy()
            for i in T:
                term = term & kap9[i]
            for i in range(3):
                if i not in T:
                    term = term & ((1 ^ alpha9[i]).astype(np.uint8))
            rhs ^= term
    check("D eq. 12 RHS equals chi_G at all 512 points",
          np.array_equal(rhs, chiG),
          f"disagreements: {int((rhs ^ chiG).sum())}")

    # product form eq:exp-graphind, for completeness
    prod = chiS9.copy()
    for i in range(3):
        prod = prod & ((1 ^ alpha9[i] ^ kap9[i]).astype(np.uint8))
    check("D' eq. 9 product form equals chi_G", np.array_equal(prod, chiG))

    # ---- E. degree and top monomial --------------------------------------
    anfG = anf_of_truth_table(chiG)
    top = 63  # u1u2u3u4w1w2, no alpha
    check("E deg chi_G = 6 with alpha-free top monomial u1u2u3u4w1w2",
          anf_deg(anfG) == 6 and int(anfG[top]) == 1,
          f"deg={anf_deg(anfG)}, coeff(u1u2u3u4w1w2)={int(anfG[top])}")

    # ---- F. label-monomial coefficients g_A ------------------------------
    ok_f = True
    g_from_anf, g_from_formula = {}, {}
    for d in range(4):
        for A in combinations(range(3), d):
            mask = sum(1 << (2 - i) for i in A)
            ga = np.zeros(64, dtype=np.uint8)
            for i in np.nonzero(anfG)[0]:
                if (int(i) >> 6) == mask:
                    ga[int(i) & 63] = 1
            g_from_anf[A] = ga
            f = chiS6.copy()
            for i in range(3):
                if i not in A:
                    f = f & ((1 ^ kap[i]).astype(np.uint8))
            fa = anf_of_truth_table(f)
            g_from_formula[A] = fa
            ok_f &= np.array_equal(ga, fa)
    check("F g_A = chi_S * prod_{i not in A}(1 + kappa_i) for all 8 label monomials",
          ok_f)

    # ---- G. fiber indicators ---------------------------------------------
    fibers = {}
    for a in range(8):
        f = np.zeros(64, dtype=np.uint8)
        for b, lab in zip(betas, labels):
            if lab == a:
                f[b] ^= 1
        fibers[a] = f
    tot = sum(int(f.sum()) for f in fibers.values())
    w0 = int(fibers[0].sum())
    check("G fiber weights sum to 48 and |kappa^{-1}(0)| is odd",
          tot == 48 and w0 % 2 == 1, f"total={tot}, |fiber 0|={w0}")

    # ---- report -----------------------------------------------------------
    print("=" * 78)
    print("Label map kappa by coset (eq:exp-labelmap), recomputed from K")
    print("=" * 78)
    for wbits, wname in ((2, "(1,0)"), (1, "(0,1)"), (3, "(1,1)")):
        for i in range(3):
            tt = np.array([kap[i, (u << 2) | wbits] for u in range(16)],
                          dtype=np.uint8)
            print(f"  w={wname}  kappa_{i+1} = "
                  f"{poly_str(anf_of_truth_table(tt), ['u1','u2','u3','u4'])}")

    print()
    print("=" * 78)
    print("Label-monomial coefficients: chi_G = sum_A alpha^A g_A(u,w)")
    print("=" * 78)
    for d in range(4):
        for A in combinations(range(3), d):
            name = "".join(f"a{i+1}" for i in A) or "1"
            fa = g_from_formula[A]
            print(f"  alpha^A = {name:6s} deg {anf_deg(fa)}  wt "
                  f"{int(anf_of_truth_table(fa).sum()):2d}  g_A = {poly_str(fa, VARS6)}")

    print()
    print("=" * 78)
    print("Fiber indicators chi_a(u,w) = 1 on the columns labelled a")
    print("=" * 78)
    for a in range(8):
        fa = anf_of_truth_table(fibers[a])
        bits = f"({(a>>2)&1},{(a>>1)&1},{a&1})"
        print(f"  a = {bits}  |fiber| {int(fibers[a].sum()):2d}  deg "
              f"{anf_deg(fa):2d}  chi_a = {poly_str(fa, VARS6)}")

    print()
    print("=" * 78)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    print(f"{npass}/{len(RESULTS)} checks passed")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
