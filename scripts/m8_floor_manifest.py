#!/usr/bin/env python3
"""Phase 1 of the n<=38 s=8 floor close-out: the MANIFEST.

Runs decide_pair's exact cheap pass (span check, incremental RM(2,8) rank, the
puncture rank-filter) over every P_{4,2} floor pair, but STUBS the decide so no
expensive alpha-walk runs. For each surviving activity it records, in stable
enumeration order, its dimV and whether the trilinear prefilter fires. This is:
  - the definitive dimV census (the real compute budget), and
  - the work-unit index for Phase 2 (pair_index, local activity index).

A "real-decide" activity = dimV>=3 AND prefilter does NOT fire (these need the
full alpha-walk; everything else is instant-unsat). Cheap pass only -> fast.

Output: <out>.jsonl  (one record per surviving activity)
        <out>.json   (aggregate: dimV histogram, real-decide count, est. budget)

Usage: m8_floor_manifest.py --task T --ntasks N --out PATH
"""
import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import m8_boundary_sweep as bs  # noqa: E402
from gate2code.alpha_linear import label_space, _t_tensor_even  # noqa: E402
from m8_p42_lib import unpack  # noqa: E402

TARGETS = (32, 34, 36, 38)
# per-dimV wall-clock seconds, MEASURED locally (non-prefiltered real decides):
# dimV=10 -> 1.65s, dimV=11 -> 6.7s; pairs-checked ratio ~4x per +1 dimV.
# Used only for a budget ESTIMATE, not for any decision.
SEC_EST = {d: 6.7 / (4 ** (11 - d)) for d in range(3, 13)}


def pair_list():
    reps = json.load(open("reports/m8_h_orbit_reps.json"))
    return [(int(v), "g8") for v in reps["h8_g8"]] + \
           [(int(v), "g8") for v in reps["h12_g8"]] + \
           [(int(v), "g12") for v in reps["h12_g12"]]


def main():
    a = sys.argv
    task = int(a[a.index("--task") + 1]) if "--task" in a else 0
    ntasks = int(a[a.index("--ntasks") + 1]) if "--ntasks" in a else 1
    out = a[a.index("--out") + 1] if "--out" in a else "reports/m8_floor_manifest"
    ckpt = f"{out}.task{task}.jsonl"

    pairs = pair_list()
    mine = [(pi, hv, tag) for pi, (hv, tag) in enumerate(pairs)
            if pi % ntasks == task]

    fh = open(ckpt, "w")
    hist = {}              # dimV -> total surviving activities
    real_hist = {}         # dimV -> real-decide activities (no prefilter)
    est_sec = 0.0
    t0 = time.time()

    real = bs.decide_activity
    state = {"pi": None, "li": 0}

    def stub(active, s, **kw):
        V = label_space(active, s)
        d = int(V.shape[0])
        pf = bool(d >= 3 and _t_tensor_even(V))
        real_decide = (d >= 3 and not pf)
        hist[d] = hist.get(d, 0) + 1
        if real_decide:
            real_hist[d] = real_hist.get(d, 0) + 1
        rec = {"pi": state["pi"], "li": state["li"], "n": len(active),
               "dimV": d, "prefilter": pf, "real": real_decide}
        fh.write(json.dumps(rec) + "\n")
        state["li"] += 1
        return {"result": "unsat", "dimV": d, "pairs_checked": 0}

    bs.decide_activity = stub
    try:
        for pi, hv, tag in mine:
            state["pi"], state["li"] = pi, 0
            g = bs.g8 if tag == "g8" else bs.g12
            t = time.time()
            ps = bs.decide_pair(g, unpack(hv), TARGETS)
            print(f"pi={pi} tag={tag} hits={ps['hits']} acts(decided)={ps['decided']} "
                  f"defer={len(ps['deferred'])} {time.time()-t:.1f}s", flush=True)
    finally:
        bs.decide_activity = real
        fh.close()

    for d in sorted(real_hist):
        est_sec += real_hist[d] * SEC_EST.get(d, SEC_EST[12])
    agg = {"task": task, "ntasks": ntasks, "pairs": len(mine),
           "dimV_hist": {int(d): hist[d] for d in sorted(hist)},
           "real_decide_hist": {int(d): real_hist[d] for d in sorted(real_hist)},
           "real_decides": sum(real_hist.values()),
           "total_surviving": sum(hist.values()),
           "est_decide_sec_1core": round(est_sec, 1),
           "wall_sec": round(time.time() - t0, 1)}
    json.dump(agg, open(f"{out}.task{task}.json", "w"), indent=1)
    print(json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
