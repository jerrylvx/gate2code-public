#!/usr/bin/env python3
"""Decide the weight-32 affine classes at m=7 (n in {31,32} at s=7).

By KTA76 Corollary 1(1) (p. 385, transcribed): every deg-3 weight-2^{m-2}
function is affinely one of three forms; deg <= 2 weight-32 at m=7 is the
single Dickson class 64 - 32 = rank-... the hyperbolic rank-2 quadratic
x1x2 has weight 32 at m=7. Four classes total, plain + all punctures.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from gate2code.alpha_linear import decide_activity
from scripts_lib_m7 import BITS, N, M

def truth(monos_or_fn):
    t = np.zeros(N, dtype=np.uint8)
    for term in monos_or_fn:
        p = np.ones(N, dtype=np.uint8)
        for v in term:
            if v > 0:
                p &= BITS[:, v - 1]
            else:
                p &= 1 ^ BITS[:, -v - 1]
        t ^= p
    return t

CLASSES = {
    # Corollary 1(1): x1x2x3 + (x1+1)x4x5
    "cor1_a": truth([(1, 2, 3), (-1, 4, 5)]),
    # x1x2x3 + x4(x2x5 + x3x6) = x1x2x3 + x2x4x5 + x3x4x6
    "cor1_b": truth([(1, 2, 3), (2, 4, 5), (3, 4, 6)]),
    # x1x2x3 + x4(x3x5 + x6x7) = x1x2x3 + x3x4x5 + x4x6x7
    "cor1_c": truth([(1, 2, 3), (3, 4, 5), (4, 6, 7)]),
    # Dickson: rank-2 quadratic
    "quad_h1": truth([(1, 2)]),
}

def main():
    t0 = time.time()
    rows = []
    for name, f in CLASSES.items():
        sup = [int(v) for v in np.flatnonzero(f)]
        assert len(sup) == 32, (name, len(sup))
        recs = {"name": name, "non_unsat": []}
        if 0 not in sup:
            r = decide_activity(sorted(sup), M)
            if r["result"] != "unsat":
                recs["non_unsat"].append("plain")
        for p in sup:
            act = sorted(v ^ p for v in sup if v != p)
            r = decide_activity(act, M)
            if r["result"] != "unsat":
                recs["non_unsat"].append(f"punct{p}")
        rows.append(recs)
        print(f"{name}: done, non-unsat {len(recs['non_unsat'])}, "
              f"{time.time()-t0:.0f}s", flush=True)
    out = {"claim": "all weight-32 affine classes at m=7 (KTA Cor 1 + Dickson), "
                    "plain + all punctures",
           "all_unsat": all(not r["non_unsat"] for r in rows),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    json.dump({"summary": out, "rows": rows},
              open("reports/m7_w32_classes.json", "w"), indent=1)

if __name__ == "__main__":
    main()
