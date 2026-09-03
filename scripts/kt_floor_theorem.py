#!/usr/bin/env python3
"""Global floor theorem via Kasami-Tokura normal forms.

For n < 2*16 = 32, every admissible support at ANY level s is, up to affine
equivalence, the support of a KT normal form [KT70, Thm 1]:
  type (i):  f = y_1...y_{nu-mu} (y_{nu-mu+1}...y_nu + y_{nu+1}...y_{nu+mu}),
             nu >= mu >= 3
  type (ii): f = y_1...y_{nu-2} (y_{nu-1}y_nu + ... + y_{nu+2mu-3}y_{nu+2mu-2}),
             mu >= 1
with weight w = 2^{m-nu+1} - 2^{m-nu+1-mu}.  Writing w = 2^{a+1}-2^{a+1-mu}
fixes a = m - nu, so for fixed support weight the class is determined by
(type, mu) and the effective dimension s_eff = dim span(support):
  type (i):  s_eff = nu + mu       (variables appearing)
  type (ii): s_eff = nu + 2mu - 2
A support spanning fewer than s dimensions is a lower-level instance, so
deciding each class AT ITS EFFECTIVE LEVEL decides all s simultaneously.
Both spectrum branches are covered: the plain support (n = w) and the
origin-punctured support (n = w - 1, positioning the form so f(0)=1).

Decides every class with weight w <= 32 via the linearized alpha-layer
(gate2code.alpha_linear).  All-unsat proves: no CH code (k=3, d_Z>=3) with
n <= 31 at any s, except possibly n=31 punctured-from-32 handled here too.
"""
import json
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, ".")
from gate2code.alpha_linear import decide_activity  # noqa: E402


def support_type_i(nu, mu):
    """Support of y_1..y_{nu-mu}(y_{nu-mu+1}..y_nu + y_{nu+1}..y_{nu+mu})
    in s_eff = nu+mu coordinate variables (y_i = x_i)."""
    s = nu + mu
    pts = []
    for x in range(1 << s):
        bits = [(x >> (s - 1 - i)) & 1 for i in range(s)]
        head = all(bits[i] for i in range(nu - mu))
        t1 = all(bits[i] for i in range(nu - mu, nu))
        t2 = all(bits[i] for i in range(nu, nu + mu))
        if head and (t1 ^ t2):
            pts.append(x)
    return s, pts


def support_type_ii(nu, mu):
    """Support of y_1..y_{nu-2}(y_{nu-1}y_nu + ... ) with mu quadratic terms,
    in s_eff = nu + 2mu - 2 variables."""
    s = nu + 2 * mu - 2
    pts = []
    for x in range(1 << s):
        bits = [(x >> (s - 1 - i)) & 1 for i in range(s)]
        head = all(bits[i] for i in range(nu - 2))
        quad = 0
        for j in range(mu):
            i1 = nu - 2 + 2 * j
            quad ^= bits[i1] & bits[i1 + 1]
        if head and quad:
            pts.append(x)
    return s, pts


def punctured_variants(s, pts):
    """Both spectrum branches: the support as-is (if 0 not in it) and the
    origin-punctured translate (shift so some support point moves to 0,
    then delete it).  Affine positioning freedom allows any support point
    at the origin; all translates are affinely equivalent, so one suffices."""
    out = []
    if 0 not in pts:
        out.append(("plain", pts))
    t = pts[0]
    shifted = sorted(p ^ t for p in pts)
    assert 0 in shifted
    out.append(("punctured", [p for p in shifted if p != 0]))
    return out


def main():
    rows = []
    t0 = time.time()
    classes = []
    for mu in range(3, 7):
        for nu in range(mu, 9):
            s, pts = support_type_i(nu, mu)
            if len(pts) <= 32 and s <= 12:
                classes.append(("i", nu, mu, s, pts))
    for mu in range(1, 6):
        for nu in range(2, 9):
            s, pts = support_type_ii(nu, mu)
            if len(pts) <= 32 and s <= 12:
                classes.append(("ii", nu, mu, s, pts))
    for typ, nu, mu, s, pts in sorted(classes, key=lambda c: len(c[4])):
        for branch, act in punctured_variants(s, pts):
            n = len(act)
            if n < 9:
                rows.append({"type": typ, "nu": nu, "mu": mu, "s_eff": s,
                             "branch": branch, "n": n,
                             "result": "skip (n<9 trivially short)"})
                continue
            r = decide_activity(act, s)
            rows.append({"type": typ, "nu": nu, "mu": mu, "s_eff": s,
                         "branch": branch, "n": n, "result": r["result"],
                         "dimV": r["dimV"], "pairs": r["pairs_checked"]})
            print(f"type {typ} nu={nu} mu={mu} s_eff={s} {branch} n={n}: "
                  f"{r['result']} (dimV={r['dimV']}, {time.time()-t0:.0f}s)",
                  flush=True)
            if r["result"] == "sat":
                print("!!! SAT — floor theorem fails here", flush=True)
    out = {"claim": "no CH code (k=3, dZ>=3) on any KT-classified support "
                    "(weight <= 32), uniformly in s",
           "classes_decided": len(rows),
           "all_unsat": all(x["result"].startswith(("unsat", "skip"))
                            for x in rows),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    json.dump({"summary": out, "classes": rows},
              open("reports/kt_floor_theorem.json", "w"), indent=1)


if __name__ == "__main__":
    main()
