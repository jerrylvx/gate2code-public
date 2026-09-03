#!/usr/bin/env python3
"""Sweep the missing (g8, h14) cells of the m=8 [32,38] floor closure.

Targets W in {36, 38} only — the cells identified in
docs/notes/m8_w14_gap_2026-07-01.md ((8,14,14) and (8,14,16); decide_pair
itself enforces the sorted-triple condition third >= max(|g|,|h|)).
Reps from reports/m8_h14_reps.json (124 Stab(g8)-orbit reps of the
22,855,680-element weight-14 class; cover audited at build time).

Checkpointed per (rep, k-batch) to reports/m8_h14_sweep.jsonl; safe to
re-run (finished units are skipped). Final verdict written to
reports/m8_h14_sweep_final.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from m8_boundary_sweep import decide_pair, g8  # noqa: E402
from m8_p42_lib import unpack  # noqa: E402

TARGETS = (36, 38)
BATCHES = list(range(0, 1 << 15, 1 << 11))  # 16 quadratic-k batches
CKPT = "reports/m8_h14_sweep.jsonl"
FINAL = "reports/m8_h14_sweep_final.json"


def main() -> None:
    reps = [int(v) for v in json.load(open("reports/m8_h14_reps.json"))["h14_g8"]]
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT) as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["rep"], r["start"]))
    print(f"reps: {len(reps)}; units total {len(reps)*len(BATCHES)}, "
          f"done {len(done)}")

    t0 = time.time()
    with open(CKPT, "a") as out:
        for ri, rep in enumerate(reps):
            h = unpack(rep)
            assert int(h.sum()) == 14
            for start in BATCHES:
                if (rep, start) in done:
                    continue
                ps = decide_pair(g8, h, TARGETS, starts=[start])
                rec = {"rep": rep, "start": start,
                       "hits": ps["hits"], "nonspan": ps["nonspan"],
                       "rank_unsat": ps["rank_unsat"], "decided": ps["decided"],
                       "non_unsat": [str(x) for x in ps["non_unsat"]],
                       "deferred": [str(x) for x in sorted(ps["deferred"])]}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if ps["non_unsat"]:
                    print(f"!! NON-UNSAT at rep={rep} start={start}: "
                          f"{ps['non_unsat']}")
            if (ri + 1) % 8 == 0:
                el = time.time() - t0
                print(f"[{el:7.0f}s] rep {ri+1}/{len(reps)}", flush=True)

    # aggregate
    tot = {"units": 0, "hits": 0, "nonspan": 0, "rank_unsat": 0,
           "decided": 0, "non_unsat": [], "deferred": []}
    with open(CKPT) as fh:
        for line in fh:
            r = json.loads(line)
            tot["units"] += 1
            for k in ("hits", "nonspan", "rank_unsat", "decided"):
                tot[k] += r[k]
            tot["non_unsat"] += r["non_unsat"]
            tot["deferred"] += r["deferred"]
    tot["all_unsat"] = not tot["non_unsat"] and not tot["deferred"]
    tot["n_reps"] = len(reps)
    tot["targets"] = list(TARGETS)
    tot["claim"] = ("(g8,h14) cells W in {36,38}: all supports decided "
                    "unsat" if tot["all_unsat"] else "SURVIVORS/DEFERRED — "
                    "see non_unsat/deferred")
    with open(FINAL, "w") as fh:
        json.dump(tot, fh, indent=1)
    print(json.dumps({k: v for k, v in tot.items()
                      if k not in ("non_unsat", "deferred")}, indent=1))
    print(f"wrote {FINAL}")


if __name__ == "__main__":
    main()
