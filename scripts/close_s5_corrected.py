#!/usr/bin/env python3
"""Corrected, exhaustive closure of the s=5 question via the activity spectrum.

The admissible activity sets at s=5 are the ~126 nonzero elements of
RM(1,5)+<delta_0> (both branches; the pre-correction proof covered only the
even branch).  For each admissible activity A we decide the alpha-layer:
labels alpha_b in F_2^3 per active fiber subject to QT1-3,(b),(c),(e)
(conditions (a),(d),(f) hold by construction of A).  UNSAT for every A proves:
no CSS code with k=3, d_Z>=3, s=5 supports CCZ via transversal T with any
diagonal Clifford correction -- sharpening the N>=8 no-go of
arXiv:2606.07734 to N>=9.

Writes reports/s5_corrected_closure.json with a per-activity verdict.
"""
import json
import sys
import time
from itertools import combinations

import numpy as np
import z3

sys.path.insert(0, ".")
from gate2code.activity_spectrum import activity_space_basis  # noqa: E402

S = 5
K = 3


def beta_bits(b):
    return [(b >> (S - 1 - j)) & 1 for j in range(S)]


def alpha_feasible(active, timeout_ms=60_000):
    """z3 decision of the alpha-layer on a fixed activity set."""
    a = {b: [z3.Bool(f"a_{b}_{i}") for i in range(K)] for b in active}
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    def parity(terms, odd):
        if not terms:
            return z3.BoolVal(not odd)  # empty XOR = 0
        acc = terms[0]
        for t in terms[1:]:
            acc = z3.Xor(acc, t)
        return acc == z3.BoolVal(bool(odd))

    # QT1: |K_i| even; QT2: |K_i ^ K_j| even; QT3: |K_1^K_2^K_3| odd
    for i in range(K):
        solver.add(parity([a[b][i] for b in active], 0))
    for i, j in combinations(range(K), 2):
        solver.add(parity([z3.And(a[b][i], a[b][j]) for b in active], 0))
    solver.add(parity([z3.And(a[b][0], a[b][1], a[b][2]) for b in active], 1))
    # (b): sum_{b: beta_l=1} alpha_i even
    for l in range(S):
        sel = [b for b in active if beta_bits(b)[l]]
        for i in range(K):
            solver.add(parity([a[b][i] for b in sel], 0))
    # (c): sum_{b: beta_l=1} alpha_i alpha_j even
    for l in range(S):
        sel = [b for b in active if beta_bits(b)[l]]
        for i, j in combinations(range(K), 2):
            solver.add(parity([z3.And(a[b][i], a[b][j]) for b in sel], 0))
    # (e): sum_{b: beta_l beta_m = 1} alpha_i even
    for l, m in combinations(range(S), 2):
        sel = [b for b in active if beta_bits(b)[l] and beta_bits(b)[m]]
        for i in range(K):
            solver.add(parity([a[b][i] for b in sel], 0))
    res = solver.check()
    return str(res)


def main():
    basis = activity_space_basis(S)
    dim = basis.shape[0]
    seen = set()
    verdicts = []
    t0 = time.time()
    cur = np.zeros(1 << S, dtype=np.uint8)
    prev_gray = 0
    for idx in range(1, 1 << dim):
        gray = idx ^ (idx >> 1)
        bit = (gray ^ prev_gray).bit_length() - 1
        cur ^= basis[bit]
        prev_gray = gray
        f = cur.copy()
        if f[0]:
            f[0] = 0  # puncture branch: activity = element + delta_0
            branch = "P"
        else:
            branch = "E"
        key = f.tobytes()
        if key in seen:
            continue
        seen.add(key)
        n = int(f.sum())
        if n == 0:
            continue
        active = [b for b in range(1, 1 << S) if f[b]]
        res = alpha_feasible(active)
        verdicts.append({"branch": branch, "n": n, "alpha": res})
        if res != "unsat":
            print(f"!! branch {branch} n={n}: {res}  active={active}")
    dt = round(time.time() - t0, 1)
    from collections import Counter
    summary = dict(Counter((v["n"], v["alpha"]) for v in verdicts))
    out = {
        "claim": "no k=3 d_Z>=3 CCZ code at s=5 (any n, any diagonal Clifford correction)",
        "activities_checked": len(verdicts),
        "verdict_summary": {f"n={k[0]}:{k[1]}": v for k, v in sorted(summary.items())},
        "all_unsat": all(v["alpha"] == "unsat" for v in verdicts),
        "seconds": dt,
    }
    print(json.dumps(out, indent=1))
    json.dump({"summary": out, "verdicts": verdicts},
              open("reports/s5_corrected_closure.json", "w"), indent=1)


if __name__ == "__main__":
    main()
