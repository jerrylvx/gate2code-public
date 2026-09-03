#!/usr/bin/env python3
"""m=8 boundary sweep: weights W in {40, 42, 44, 46} (n = 39..46 at s=8).

Covers every P_{4,2}-representable f (KTA Thm 1 form f = xg + yh + xyk;
representability at W >= 40 = the support misses some 6-flat; the
non-representable residue is documented separately). Sorted quadrant
splits (a <= b <= c), a+b+c = W <= 46 force a in {8,12} and
b in {8,12,16,18}. Pair enumeration:
  - b <= 12: g normalized to KT form, h over stabilizer-reduced w8/w12
    rep sets (reports/m8_h_orbit_reps.json) -- as in the [32,38] sweep
  - b in {16,18}: the LARGE component is pinned to its invariant-class
    representatives (reports/m6_w16_w18_invariant_classes.json; the seeds
    intersect every AGL class since every cubic part is GL-equivalent to
    a canonical one), and the SMALL component g' sweeps its full set
    (all 11160 flats for a=8; the full 1.75M weight-12 class for a=12).
Per pair: k sweeps RM(2,6) via quadratic part x Walsh spectrum; per hit:
span-8 check, then incremental packed rank of RM(2,8)|_A (base rank from
the three fixed quadrants precomputed); rank >= W-2 -> unsat (dim V_A < 3).
Survivors decided fully, plain + rank-filtered punctures.

Usage: m8_boundary_sweep.py --mode {b16a8,b18a8,b16a12,small} --task T --ntasks N [--out P]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from gate2code.alpha_linear import decide_activity
from m8_p42_lib import (B6, N6, truth6, unpack, pack, all_3flats,
                        weight12_class)

M8, N8 = 8, 256
QUAD_MONOS6 = [(i, j) for i in range(1, 7) for j in range(i + 1, 7)]
QUAD_TT6 = np.stack([B6[:, i-1] & B6[:, j-1] for i, j in QUAD_MONOS6], axis=0)
PARTAB = np.array([[bin(v & a).count("1") & 1 for v in range(N6)]
                   for a in range(N6)], dtype=np.uint8)

def wht64(tt):
    a = 1 - 2 * tt.astype(np.int16)
    h = 1
    while h < N6:
        a = a.reshape(-1, N6 // (2*h), 2, h)
        a = np.stack([a[:,:,0,:]+a[:,:,1,:], a[:,:,0,:]-a[:,:,1,:]], axis=2)
        h *= 2
    return a.reshape(-1, N6)

x8v = np.arange(N8, dtype=np.int64)
B8 = np.stack([((x8v >> (M8 - i)) & 1) for i in range(1, M8+1)],
              axis=1).astype(np.uint8)
RM28 = [np.ones(N8, dtype=np.uint8)] + [B8[:, i] for i in range(8)] + \
       [B8[:, i] & B8[:, j] for i in range(8) for j in range(i+1, 8)]
RM28 = np.stack(RM28, axis=0)
RM28_INT = [[int("".join(str(b) for b in RM28[r])), ] for r in range(37)]

def reduce_basis(basis, v):
    for b in basis:
        m = v ^ b
        if m < v:
            v = m
    return v

def packed_rank_rows(rows):
    basis = []
    for v in rows:
        v = reduce_basis(basis, v)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return basis

# --- module-level constants and per-pair worker (extracted from main so the
# --- [32,38] driver scripts/m8_small_v3.py can reuse it with k-batch sharding
# --- and per-unit checkpointing; behavior is byte-identical to the inline loop)
i6 = x8v >> 2
q7 = (B8[:, 6] & (1 ^ B8[:, 7])).astype(bool)
q8 = ((1 ^ B8[:, 6]) & B8[:, 7]).astype(bool)
q78 = (B8[:, 6] & B8[:, 7]).astype(bool)
g8 = truth6([(1, 2, 3)])
g12 = B6[:, 0] & (truth6([(2, 3)]) ^ truth6([(4, 5)]))


def count_hits(g, h, targets):
    """Cheap hit count for a pair (WHT only, no decides) -- lets a driver
    adaptively pick the number of hit-shards for heavy pairs."""
    wg, wh = int(g.sum()), int(h.sum())
    u = g ^ h
    Ts = {W - wg - wh for W in targets if W - wg - wh >= max(wh, wg)}
    if not Ts:
        return 0
    Tl = list(Ts)
    n = 0
    for start in range(0, 1 << 15, 1 << 11):
        qs = np.arange(start, start + (1 << 11), dtype=np.int64)
        sel = ((qs[:, None] >> np.arange(15)[None, :]) & 1).astype(np.uint8)
        uk = (sel @ QUAD_TT6 % 2).astype(np.uint8) ^ u[None, :]
        Wsp = wht64(uk)
        for c, wm in ((0, (N6 - Wsp) // 2), (1, (N6 + Wsp) // 2)):
            n += int(np.isin(wm, Tl).sum())
    return n


def decide_pair(g, h, targets, starts=None, qwidth=1 << 11, hit_shard=None):
    """Decide one P_{4,2} pair (g, h) over quadratic-k sub-ranges: each `start`
    covers q in [start, start+qwidth). Default = the 16 width-2^11 batches of
    the full 2^15 sweep.

    `hit_shard=(sid, n)` shards the per-hit decide work: hits are enumerated in a
    fixed order and only those with (index % n == sid) are processed (the WHT/
    setup is repeated per shard, but cheap). This is the only way to split a
    single heavy quadratic part whose hit-count alone overruns the walltime --
    q-range sharding cannot, since one q is the indivisible WHT atom. Aggregating
    all n shards reproduces the full pair. Returns a per-pair stats dict
    {hits, nonspan, rank_unsat, decided, non_unsat, deferred}."""
    if starts is None:
        starts = list(range(0, 1 << 15, 1 << 11))
    ps = {"hits": 0, "nonspan": 0, "rank_unsat": 0, "decided": 0,
          "non_unsat": [], "deferred": set()}
    wg, wh = int(g.sum()), int(h.sum())
    u = g ^ h
    Ts = {W - wg - wh for W in targets if W - wg - wh >= max(wh, wg)}
    if not Ts:
        return ps
    base_mask = (q7 & g[i6].astype(bool)) | (q8 & h[i6].astype(bool))
    base_pts = [int(v) for v in np.flatnonzero(base_mask)]
    nb = len(base_pts)
    rows_packed = []
    for r in range(37):
        v = 0
        row = RM28[r]
        for j, ptv in enumerate(base_pts):
            if row[ptv]:
                v |= 1 << j
        rows_packed.append((v, 1 << r))
    basis = []
    null_combos = []
    for v, c in rows_packed:
        for bv, bc in basis:
            if (v ^ bv) < v:
                v, c = v ^ bv, c ^ bc
        if v:
            basis.append((v, c))
            basis.sort(key=lambda x: -x[0])
        else:
            null_combos.append(c)
    d0 = len(null_combos)
    if d0:
        sel = np.array([[(c >> r) & 1 for r in range(37)]
                        for c in null_combos], dtype=np.uint8)
        N0_eval = sel @ RM28 % 2          # (d0, 256)
    span_basis = []
    bb = base_pts[0]
    for v in base_pts[1:]:
        w = v ^ bb
        for sb in span_basis:
            if (w ^ sb) < w:
                w ^= sb
        if w:
            span_basis.append(w)
            span_basis.sort(reverse=True)
    hc = 0
    for start in starts:
        qs = np.arange(start, start + qwidth, dtype=np.int64)
        sel = ((qs[:, None] >> np.arange(15)[None, :]) & 1).astype(np.uint8)
        uk = (sel @ QUAD_TT6 % 2).astype(np.uint8) ^ u[None, :]
        Wsp = wht64(uk)
        for c, wm in ((0, (N6 - Wsp)//2), (1, (N6 + Wsp)//2)):
            for bi, a in np.argwhere(np.isin(wm, list(Ts))):
                mine = hit_shard is None or hc % hit_shard[1] == hit_shard[0]
                hc += 1
                if not mine:
                    continue
                ghk = uk[bi] ^ PARTAB[a] ^ c
                new_pts = [int(v) for v in np.flatnonzero(
                    q78 & ghk[i6].astype(bool))]
                W = nb + len(new_pts)
                ps["hits"] += 1
                sb2 = list(span_basis)
                sdim = len(sb2)
                for v in new_pts:
                    w = v ^ bb
                    for s_ in sb2:
                        if (w ^ s_) < w:
                            w ^= s_
                    if w:
                        sb2.append(w)
                        sb2.sort(reverse=True)
                        sdim += 1
                if sdim < 8:
                    ps["nonspan"] += 1
                    continue
                if d0 == 0:
                    rk = 37
                else:
                    small = []
                    for irow in range(d0):
                        v = 0
                        ev = N0_eval[irow]
                        for j, ptv in enumerate(new_pts):
                            if ev[ptv]:
                                v |= 1 << j
                        small.append(v)
                    rk = 37 - (d0 - len(packed_rank_rows(small)))
                if rk >= W - 2:
                    ps["rank_unsat"] += 1
                    continue
                sup = base_pts + new_pts
                if W - rk > 12:
                    ps["deferred"].add(tuple(sup))
                    continue
                acts = ([sorted(sup)] if 0 not in sup else [])
                for p in sup:
                    psup = [v ^ p for v in sup if v != p]
                    prows = []
                    for r in range(37):
                        v = 0
                        row = RM28[r]
                        for j, ptv in enumerate(psup):
                            if row[ptv]:
                                v |= 1 << j
                        prows.append(v)
                    if len(packed_rank_rows(prows)) >= W - 3:
                        ps["rank_unsat"] += 1
                        continue
                    acts.append(sorted(psup))
                for act in acts:
                    r = decide_activity(act, M8)
                    ps["decided"] += 1
                    if r["result"] != "unsat":
                        ps["non_unsat"].append({"W": W, "n": len(act)})
    return ps


def main():
    args = sys.argv[1:]
    mode = args[args.index("--mode") + 1]
    task = int(args[args.index("--task") + 1]) if "--task" in args else 0
    ntasks = int(args[args.index("--ntasks") + 1]) if "--ntasks" in args else 1
    out_path = args[args.index("--out") + 1] if "--out" in args else None
    t0 = time.time()
    inv = json.load(open("reports/m6_w16_w18_invariant_classes.json"))
    targets = (32, 34, 36, 38, 40, 42, 44, 46) if mode == "small" else \
        (40, 42, 44, 46)
    if mode == "b16a8":     # h-rep weight 16 pinned, g' over all flats
        hreps = [int(c["rep"]) for c in inv["16"].values()]
        gset = sorted(all_3flats())
        pairs = [(hr, gv) for hr in hreps for gv in gset]
    elif mode == "b18a8":
        hreps = [int(c["rep"]) for c in inv["18"].values()]
        gset = sorted(all_3flats())
        pairs = [(hr, gv) for hr in hreps for gv in gset]
    elif mode == "b16a12":
        hreps = [int(c["rep"]) for c in inv["16"].values()]
        gset = sorted(weight12_class())
        pairs = [(hr, gv) for hr in hreps for gv in gset]
    elif mode == "small":   # b <= 12: reuse reduced rep sets, W in targets
        reps = json.load(open("reports/m8_h_orbit_reps.json"))
        pairs = [(int(v), None, "g8") for v in reps["h8_g8"]] + \
                [(int(v), None, "g8") for v in reps["h12_g8"]] + \
                [(int(v), None, "g12") for v in reps["h12_g12"]]
    else:
        raise SystemExit("bad mode")
    mine = pairs[task::ntasks]
    st = {"mode": mode, "task": task, "pairs": len(mine), "hits": 0,
          "nonspan": 0, "rank_unsat": 0, "decided": 0, "non_unsat": []}
    st["deferred"] = set()
    for item in mine:
        if mode == "small":
            hv, _, gtag = item
            g = g8 if gtag == "g8" else g12
            h = unpack(hv)
        else:
            hr, gv = item
            h = unpack(hr)      # the large pinned component
            g = unpack(gv)      # the small swept component
        tpair = time.time()
        ps = decide_pair(g, h, targets)
        for kk in ("hits", "nonspan", "rank_unsat", "decided"):
            st[kk] += ps[kk]
        st["non_unsat"].extend(ps["non_unsat"])
        st["deferred"].update(ps["deferred"])
        print(f"pair done: hits={ps['hits']} nonspan={ps['nonspan']} "
              f"rank_unsat={ps['rank_unsat']} decided={ps['decided']} "
              f"defer={len(ps['deferred'])} {time.time()-tpair:.1f}s",
              flush=True)
    st["deferred"] = [list(x) for x in sorted(st["deferred"])]
    st["all_unsat"] = not st["non_unsat"]
    st["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(st), flush=True)
    if out_path:
        json.dump(st, open(out_path, "w"), indent=1)

if __name__ == "__main__":
    main()
