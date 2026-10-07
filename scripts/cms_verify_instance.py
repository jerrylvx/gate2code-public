#!/usr/bin/env python3
"""SAT search for generator matrices satisfying the nine CH conditions.

The encoding has one support bit chi_S(beta) and three label bits kappa_i(beta) for each nonzero
syndrome beta. Tseitin AND variables give the products chi_S(beta) prod_{i in T_K} kappa_i(beta),
CryptoMiniSat XOR clauses impose the CH parities of degree one to three, and a PySAT cardinality
encoding fixes the length sum_beta chi_S(beta) = n. One label per syndrome gives d_Z >= 3.

Task 's5' : stabilizer rank 5, any length.
Task <n>  : stabilizer rank --s with length exactly n.

Each run first checks the encoder on the [[47,3,3]] code of Jacinto et al. (arXiv:2606.07734),
pinned as a known solution, which must be satisfiable. The output records the first solution.
That solution satisfies the CH conditions; a single-qubit transversal sign pattern must then be
tested separately with the signed-weight condition of the paper.
"""
import argparse
import json
import time
from itertools import combinations

from pycryptosat import Solver
from pysat.card import CardEnc, EncType

K = 3

# Verified Jacinto D3 matrix (arXiv:2606.07734v1 App. D), K = rows 1-3.
D3_ROWS = [
    "01111101011011011111010011101001100001001011101",
    "10010011110000101100111101110100111010010110000",
    "00010011100101100000010001110111101111101010110",
    "10011101001010011101001010011101001010011101001",
    "01001011011001001011011001001011011001001011011",
    "00111000111000111000111000111000111000111000111",
    "00000111111000000111111000000111111000000111111",
    "00000000000111111111111000000000000111111111111",
    "00000000000000000000000111111111111111111111111",
]


def build(s, threads=1, card_eq=None, pin_d3=False):
    fibers = list(range(1, 1 << s))
    nv = 0

    def new():
        nonlocal nv
        nv += 1
        return nv

    g = {b: new() for b in fibers}
    a = {b: [new() for _ in range(K)] for b in fibers}
    solver = Solver(threads=threads)

    def tseitin_and(lits):
        t = new()
        for x in lits:
            solver.add_clause([-t, x])
        solver.add_clause([t] + [-x for x in lits])
        return t

    mono_var = {}
    for b in fibers:
        mono_var[b] = {(): g[b]}
        for d in range(1, K + 1):
            for k_idx in combinations(range(K), d):
                mono_var[b][k_idx] = tseitin_and([g[b]] + [a[b][i] for i in k_idx])

    def beta_bit(b, j):
        return (b >> (s - 1 - j)) & 1

    for deg in range(1, 4):
        for idx in combinations(range(K + s), deg):
            k_idx = tuple(i for i in idx if i < K)
            s_idx = tuple(i - K for i in idx if i >= K)
            target = idx == tuple(range(K))
            lits = [mono_var[b][k_idx] for b in fibers
                    if all(beta_bit(b, j) for j in s_idx)]
            solver.add_xor_clause(lits, target)

    if card_eq is not None:
        cnf = CardEnc.equals([g[b] for b in fibers], bound=card_eq,
                             top_id=nv, encoding=EncType.seqcounter)
        for cl in cnf.clauses:
            solver.add_clause(cl)

    if pin_d3:
        rows = [[int(c) for c in r] for r in D3_ROWS]
        act, lab = set(), {}
        for j in range(47):
            b = int("".join(str(rows[i][j]) for i in range(3, 9)), 2)
            act.add(b)
            lab[b] = [rows[i][j] for i in range(3)]
        for b in fibers:
            solver.add_clause([g[b]] if b in act else [-g[b]])
            if b in act:
                for i in range(3):
                    solver.add_clause([a[b][i]] if lab[b][i] else [-a[b][i]])

    return solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="'s5' or an integer n (exact cardinality at --s)")
    ap.add_argument("--s", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctrl = build(6, threads=1, pin_d3=True)
    sat, _ = ctrl.solve()
    assert sat, "encoder control FAILED: D3 should be SAT"
    print("control: D3 pinned sat OK", flush=True)

    t0 = time.time()
    if args.task == "s5":
        s_val = 5
        solver, meta = build_with_vars(s_val, threads=args.threads)
        label = "s5_free"
    else:
        n = int(args.task)
        s_val = args.s
        solver, meta = build_with_vars(s_val, threads=args.threads, card_eq=n)
        label = f"s{s_val}_n{n}"
    sat, sol = solver.solve()
    dt = round(time.time() - t0, 1)
    res = "sat" if sat else "unsat"
    rec = {"task": label, "result": res, "seconds": dt,
           "threads": args.threads, "control": "sat"}
    if sat:
        rec.update(extract_witness(sol, meta, s_val))
    print(f"{label}: {res} ({dt}s)" + (
        f"  rank={rec.get('rank_G')} dX={rec.get('dX')} dZ={rec.get('dZ')}"
        if sat else ""), flush=True)

    out = args.out or f"cms_{label}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print("wrote", out, flush=True)



def build_with_vars(s, threads=1, card_eq=None, pin_d3=False):
    """Like build() but also returns the (g, a) variable maps for witness
    extraction.  Kept separate to avoid touching validated call sites."""
    fibers = list(range(1, 1 << s))
    nv = 0

    def new():
        nonlocal nv
        nv += 1
        return nv

    g = {b: new() for b in fibers}
    a = {b: [new() for _ in range(K)] for b in fibers}
    solver = Solver(threads=threads)

    def tseitin_and(lits):
        t = new()
        for x in lits:
            solver.add_clause([-t, x])
        solver.add_clause([t] + [-x for x in lits])
        return t

    mono_var = {}
    for b in fibers:
        mono_var[b] = {(): g[b]}
        for d in range(1, K + 1):
            for k_idx in combinations(range(K), d):
                mono_var[b][k_idx] = tseitin_and([g[b]] + [a[b][i] for i in k_idx])

    def beta_bit(b, j):
        return (b >> (s - 1 - j)) & 1

    for deg in range(1, 4):
        for idx in combinations(range(K + s), deg):
            k_idx = tuple(i for i in idx if i < K)
            s_idx = tuple(i - K for i in idx if i >= K)
            target = idx == tuple(range(K))
            lits = [mono_var[b][k_idx] for b in fibers
                    if all(beta_bit(b, j) for j in s_idx)]
            solver.add_xor_clause(lits, target)

    if card_eq is not None:
        cnf = CardEnc.equals([g[b] for b in fibers], bound=card_eq,
                             top_id=nv, encoding=EncType.seqcounter)
        for cl in cnf.clauses:
            solver.add_clause(cl)

    return solver, (g, a)


def extract_witness(sol, meta, s):
    """Reconstruct G=[K;S] from a CMS model and compute exact invariants."""
    import numpy as np
    g, a = meta
    cols = []
    for b in sorted(g):
        if sol[g[b]]:
            alpha = [1 if sol[a[b][i]] else 0 for i in range(K)]
            beta = [(b >> (s - 1 - j)) & 1 for j in range(s)]
            cols.append(alpha + beta)
    G = np.array(cols, dtype=np.uint8).T
    out = {"n": int(G.shape[1]),
           "G_rows": ["".join(map(str, row.tolist())) for row in G]}
    try:
        import sys
        sys.path.insert(0, ".")
        from gate2code.ccz import compute_distances, f2_rank
        out["rank_G"] = int(f2_rank(G.astype(int)))
        dX, dZ = compute_distances(G.astype(int), k=K)
        out["dX"], out["dZ"] = int(dX), int(dZ)
    except Exception as exc:  # cluster venv may lack extras; record raw G anyway
        out["invariant_error"] = str(exc)
    return out


if __name__ == "__main__":
    main()
