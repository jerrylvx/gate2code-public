#!/usr/bin/env python3
"""Completeness audit for the m=8 P_{4,2} h-representative sets.

The [32,38] floor sweep reduces pairs (g,h) by fixing g to either the canonical
8-point or 12-point component, then enumerating h up to Stab(g).  This script
checks the missing load-bearing fact: the saved representative lists in
reports/m8_h_orbit_reps.json are complete orbit covers of the full A_8/A_12
sets under the relevant stabilizer subgroup.

Outputs reports/m8_h_rep_completeness_audit.json by default.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from m8_p42_lib import (  # noqa: E402
    B6,
    all_3flats,
    orbit_bfs,
    stab_perms_g8,
    stab_perms_g12,
    truth6,
    unpack,
    weight12_class,
)


def _orbit_cover_report(name: str, reps: list[int], target: set[int], perms) -> dict:
    """Close reps under perms and compare the union with target."""
    t0 = time.time()
    covered: set[int] = set()
    orbit_sizes: list[int] = []
    overlap = 0
    for idx, rep in enumerate(reps):
        orb = orbit_bfs(unpack(rep), perms)
        orbit_sizes.append(len(orb))
        overlap += len(covered & orb)
        covered |= orb
        if (idx + 1) % 250 == 0:
            print(
                f"{name}: closed {idx + 1}/{len(reps)} reps; "
                f"covered={len(covered)}",
                flush=True,
            )

    outside = covered - target
    missing = target - covered
    return {
        "name": name,
        "n_reps": len(reps),
        "target_count": len(target),
        "sum_orbit_sizes": int(sum(orbit_sizes)),
        "covered_count": len(covered),
        "overlap_count": int(overlap),
        "outside_target_count": len(outside),
        "missing_count": len(missing),
        "complete": len(missing) == 0 and len(outside) == 0,
        "orbit_size_hist": {
            str(s): orbit_sizes.count(s) for s in sorted(set(orbit_sizes))
        },
        "missing_sample": [str(v) for v in sorted(missing)[:20]],
        "outside_sample": [str(v) for v in sorted(outside)[:20]],
        "seconds": round(time.time() - t0, 3),
    }


def main() -> None:
    args = sys.argv[1:]
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path(
        "reports/m8_h_rep_completeness_audit.json"
    )
    reps_path = Path(args[args.index("--reps") + 1]) if "--reps" in args else Path(
        "reports/m8_h_orbit_reps.json"
    )

    reps_raw = json.load(open(reps_path))
    reps = {k: [int(v) for v in vals] for k, vals in reps_raw.items()}

    g8 = truth6([(1, 2, 3)])
    g12 = B6[:, 0] & (truth6([(2, 3)]) ^ truth6([(4, 5)]))
    p8 = stab_perms_g8()
    p12 = stab_perms_g12()
    sanity = {
        "g8_weight": int(g8.sum()),
        "g12_weight": int(g12.sum()),
        "stab_g8_generators": len(p8),
        "stab_g12_generators": len(p12),
        "stab_g8_preserves_g8": bool(all(np.array_equal(g8[p], g8) for p in p8)),
        "stab_g12_preserves_g12": bool(all(np.array_equal(g12[p], g12) for p in p12)),
    }
    if not sanity["stab_g8_preserves_g8"] or not sanity["stab_g12_preserves_g12"]:
        raise SystemExit(f"bad stabilizer generator sanity: {sanity}")

    print("materializing A8 and A12...", flush=True)
    t0 = time.time()
    a8 = all_3flats()
    a12 = weight12_class()
    materialize = {
        "A8_count": len(a8),
        "A12_count": len(a12),
        "seconds": round(time.time() - t0, 3),
    }

    reports = {
        "h8_g8": _orbit_cover_report("h8_g8", reps["h8_g8"], a8, p8),
        "h12_g8": _orbit_cover_report("h12_g8", reps["h12_g8"], a12, p8),
        "h12_g12": _orbit_cover_report("h12_g12", reps["h12_g12"], a12, p12),
    }
    payload = {
        "claim": (
            "Saved m=8 h-representative sets are complete Stab(g)-orbit "
            "covers of A8/A12 for the P_{4,2} floor sweep."
        ),
        "reps_path": str(reps_path),
        "sanity": sanity,
        "materialize": materialize,
        "reports": reports,
        "all_complete": all(r["complete"] for r in reports.values()),
        "seconds": round(sum(r["seconds"] for r in reports.values()) + materialize["seconds"], 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=1)
    print(json.dumps(payload, indent=1), flush=True)


if __name__ == "__main__":
    main()
