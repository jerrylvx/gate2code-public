"""Run the exact classification for (p,q), then analyze every class.

Per class: CH check, all native deltas (exact enumeration), min / set of T-dagger counts,
all-T possible, d_X, d_Z, weight enumerator of C1, per-block quadratic-rank profile of the labels,
per-block degree of a minimal delta, stabilizer orders. Also locates the class of the stored
Q48 witness and runs random invariance tests of the canonical form.
Usage: python analyze_classes.py 4 2
"""
import sys, time, pickle, random, json
from collections import Counter
from pathlib import Path
import numpy as np
from classify import run
from sc_lib import popcount
from native import ch_ok, native_solutions, native_histogram, verify_native_bruteforce, distances, weight_enumerator

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]


def block_tables(sup, f):
    p = sup.p
    tabs = {}
    for j, x in enumerate(sup.points):
        u, w = x & ((1 << p) - 1), x >> p
        tabs.setdefault(w, [0] * (1 << p))[u] = (f >> j) & 1
    return tabs


def anf(tab, p):
    a = list(tab)
    for i in range(p):
        st = 1 << i
        for m in range(1 << p):
            if m & st:
                a[m] ^= a[m ^ st]
    return a


def anf_terms(a, p):
    return [m for m in range(1 << p) if a[m]]


def deg(a):
    return max((bin(m).count("1") for m in range(len(a)) if a[m]), default=-1)


def quad_rank(a, p):
    """Rank of the alternating form of the degree-2 part of ANF a."""
    M = [0] * p
    for m in range(1 << p):
        if a[m] and bin(m).count("1") == 2:
            i, j = [t for t in range(p) if (m >> t) & 1]
            M[i] ^= 1 << j
            M[j] ^= 1 << i
    from sc_lib import rank_int
    return rank_int(M)


def fmt_anf(a, p, names=None):
    names = names or [f"u{i+1}" for i in range(p)]
    ts = []
    for m in anf_terms(a, p):
        ts.append("1" if m == 0 else "".join(names[i] for i in range(p) if (m >> i) & 1))
    return "+".join(ts) if ts else "0"


def to_int(bits):
    return sum(int(b) << j for j, b in enumerate(bits))


def main(p, q, sup=None, tag=""):
    t0 = time.time()
    results, canon, sup = run(p, q, sup=sup, tag=tag)
    n, s = sup.n, sup.s
    cols = list(sup.points)
    classes = results["classes"]
    # ---------------- consistency / invariance tests
    total_members = sum(len(c["members"]) for c in classes)
    assert total_members == len(results["triples"]), (total_members, len(results["triples"]))
    key_to_class = {}
    for ci, c in enumerate(classes):
        for m in c["members"]:
            key_to_class[m] = ci
    rnd = random.Random(7)
    from itertools import permutations
    for trial in range(300):
        c = classes[rnd.randrange(len(classes))]
        trip = list(c["rep"])
        # random group element: random word in generators
        g = tuple(1 << i for i in range(s))
        for _ in range(30):
            g = sup.gmul(sup.gens[rnd.randrange(len(sup.gens))], g, s)
        M = sup.quot_matrix(g)
        from sc_lib import mat_apply
        img = [mat_apply(M, t) for t in trip]
        # add random C2 offsets are invisible in quotient coords; permute labels
        perm = list(permutations(range(3)))[rnd.randrange(6)]
        img = [img[i] for i in perm]
        ck = canon(*img)
        assert key_to_class[ck] == key_to_class[canon(*trip)]
    print(f"invariance tests passed ({time.time()-t0:.1f}s)", flush=True)
    # ---------------- per-class analysis
    rows_S = list(sup.coord)
    out = []
    for ci, c in enumerate(classes):
        K = [sup.lift(t) for t in c["rep"]]
        rows = K + rows_S
        assert ch_ok(rows)
        kd, sols = native_solutions(rows, n, max_dim=20)
        rec = {"id": ci, "n_ordered": len(c["members"]), "stab_H": c["stab_H"],
               "aut": c["aut_order"], "kernel_dim": kd, "rep_quot": list(c["rep"])}
        if sols is None:  # large kernel: exact numba Gray-code enumeration
            kd2, hist, bestd = native_histogram(rows, n)
            assert kd2 == kd and hist is not None
            if hist.sum():
                cnt = {w: int(h) for w, h in enumerate(hist) if h}
                rec.update(native=True, tdag_counts=sorted(cnt), n_native=int(hist.sum()),
                           min_tdag=min(cnt), best_delta=bestd, all_T=bool(0 in cnt),
                           tdag_hist=cnt)
            else:
                rec.update(native=False, all_T=False, n_native=0)
        elif len(sols):
            wts = np.bitwise_count(sols).astype(int)
            cnt = Counter(wts.tolist())
            rec["tdag_hist"] = dict(cnt)
            rec["native"] = True
            rec["tdag_counts"] = sorted(cnt)
            rec["n_native"] = int(len(sols))
            rec["min_tdag"] = int(wts.min())
            best = int(sols[int(np.argmin(wts))])
            rec["best_delta"] = best
            rec["all_T"] = bool(0 in cnt)
        else:
            rec.update(native=False, all_T=False, n_native=0)
        dx, dz = distances(rows, n, cols)
        rec["dX"], rec["dZ"] = dx, dz
        rec["WE"] = weight_enumerator(rows, n)
        prof = []
        for f in K:
            tabs = block_tables(sup, f)
            prof.append(tuple(quad_rank(anf(tabs[w], p), p) for w in sorted(tabs)))
        rec["quad_rank_profile"] = prof
        out.append(rec)
        if ci % 500 == 0:
            print(f"  analyzed {ci}/{len(classes)} ({time.time()-t0:.1f}s)", flush=True)
    # ---------------- Q48 witness
    q48 = None
    wf = ROOT / "data/family_witnesses/n48_fg48.npz"
    if (p, q) == (4, 2) and not tag:
        d = np.load(wf)
        G = d["G"] % 2
        Srows = [to_int(r) for r in G[3:]]
        assert Srows == rows_S, "Q48 S rows differ from coordinate functions"
        Kq = [to_int(r) for r in G[:3]]
        ck = canon(*[sup.proj(k) for k in Kq])
        q48 = key_to_class[ck]
        print("Q48 class:", q48, out[q48])
    res = {"p": p, "q": q, "n": n, "H": results["H"], "level1": results["level1"],
           "n_ordered": len(results["triples"]), "classes": out, "q48_class": q48}
    with open(HERE / f"analysis_p{p}q{q}{tag}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    print(f"done ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
