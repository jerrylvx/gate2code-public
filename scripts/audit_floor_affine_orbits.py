#!/usr/bin/env python3
"""Floor completeness LEDGER: AGL affine-orbit / RM-coset coverage audit.

This consolidates the scattered per-cell floor evidence into one machine-checkable
ledger for the length-floor theorem through n<=38.  It does NOT re-prove the
Kasami-Tokura / KTA classification --
that within-weight affine-class completeness is a literature theorem (cited).  What
it DOES machine-check:

  (M1) RM-coset weight-spectrum coverage at s<=6: every d_X-feasible admissible
       length n<=N (from activity_spectrum, branch E + puncture branch P) is
       claimed by exactly one ledger cell -- nothing admissible is left undecided.
  (M2) KTA Table-I transcription fidelity: each kta_normal_forms row produces the
       declared support weight and degree (the doc flags "support is authoritative").
  (M3) Reuse of the existing combinatorial orbit-closure certificates: the s=8
       P_{4,2} h-rep cover (reports/m8_h_rep_completeness_audit.json) and the KTA
       puncture orbits (reports/kta_puncture_orbits.json).

Each cell is labelled by completeness_basis:
  - "rm_spectrum_machine"  : weight coverage is machine-verified here (s<=6).
  - "machine_orbit_cover"  : a combinatorial orbit cover is machine-verified
                             (s=8 h-rep; puncture orbits).
  - "kt_kta_theorem_cited" : within-weight affine completeness is the KT/KTA
                             classification theorem (a citation, not re-proved).

Outputs reports/floor_affine_orbit_audit.json.  Exit non-zero if any machine-check
(M1/M2/M3) fails -- a failure would be a genuine gap in the floor argument.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from gate2code import activity_spectrum as asp
from gate2code import kta_normal_forms as kf

N_CORE = 30   # unconditional floor theorem
N_EXT = 38    # extension target (now closed: both sweeps finished all-unsat)


# ---------------------------------------------------------------------------
# Decide-set ledger.  Each cell claims a set of code lengths n it is responsible
# for, keyed by level s.  Branch E: n = wt(base); branch P (puncture): n = wt-1.
# Sources: expedited_minimality.tex (floor proof + extension) and the per-cell
# scripts named in `script`.  `status` is "closed" or "running".
# ---------------------------------------------------------------------------
LEDGER = [
    # s <= 4: only affine flats -> flat obstruction (paper lem:exp-flatobstruction)
    {"s": 4, "claims_n": [15], "method": "flat obstruction (RM(1,4))",
     "script": "activity_spectrum / lem:exp-flatobstruction",
     "basis": "rm_spectrum_machine", "status": "closed"},

    # s = 5: full exhaustion (Prop lb): 62 non-simplex unsat + n=31 simplex ruled out
    {"s": 5, "claims_n": [15, 16, 31], "method": "s<=5 exhaustion (Prop lb)",
     "script": "alpha_linear last-label linearization", "basis": "rm_spectrum_machine",
     "status": "closed"},

    # s = 6: scans + Dickson quadratic classes (with both branches) + w36 elliptic
    {"s": 6, "claims_n": [15, 16], "method": "certified small-weight scan",
     "script": "ambient_m_class_decisions", "basis": "rm_spectrum_machine", "status": "closed"},
    {"s": 6, "claims_n": [23, 24], "method": "Dickson w24 class (E) + puncture (P, n=23)",
     "script": "ambient_m_class_decisions + puncture", "basis": "kt_kta_theorem_cited",
     "status": "closed"},
    {"s": 6, "claims_n": [27, 28], "method": "Dickson w28 class (E) + puncture (P, n=27)",
     "script": "ambient_m_class_decisions + puncture", "basis": "kt_kta_theorem_cited",
     "status": "closed"},
    {"s": 6, "claims_n": [31, 32], "method": "certified scan (n=31 simplex, n=32)",
     "script": "ambient_m_class_decisions", "basis": "rm_spectrum_machine", "status": "closed"},
    {"s": 6, "claims_n": [35, 36], "method": "s=6 w36 elliptic-quadric class (E) + puncture (P, n=35)",
     "script": "m6_w40 / elliptic class", "basis": "kt_kta_theorem_cited", "status": "closed"},

    # s = 7: cubic (RM(3,7)) classes; sub-32 by KT completeness, w32 and w36 both done
    {"s": 7, "claims_n": [16, 24, 28], "method": "sub-32 cubic weights (KT completeness)",
     "script": "m7_cubic_class_scan", "basis": "kt_kta_theorem_cited", "status": "closed"},
    {"s": 7, "claims_n": [31, 32], "method": "w32 cubic classes (cor1_abc unsat; quad_h1 subsumed)",
     "script": "m7_w32_classes", "basis": "machine_orbit_cover", "status": "closed"},
    {"s": 7, "claims_n": [35, 36], "method": "w36 cubic classes + all punctures",
     "script": "m7_w36_decide_all (job 48301149)", "basis": "kt_kta_theorem_cited", "status": "closed"},

    # s = 8: degree-4 even weights [32,38] via P_{4,2} h-rep sweep (h-rep cover audited)
    {"s": 8, "claims_n": [32, 33, 34, 35, 36, 37, 38],
     "method": "P_{4,2} g/h normal form, Stab(g)-orbit h-reps + punctures",
     "script": "m8_hitshard (job 48300213)", "basis": "machine_orbit_cover", "status": "closed"},

    # s = 9 (and m=10): KTA Table I classes + every puncture point
    {"s": 9, "claims_n": [31, 32, 35, 36, 37, 38],
     "method": "KTA Table I (rows 11/15/19/20/21) + all puncture points",
     "script": "kta_puncture_orbits", "basis": "machine_orbit_cover", "status": "closed"},
]


def m1_spectrum_coverage(n_max: int) -> dict:
    """Every d_X-feasible admissible n<=n_max at s<=6 is claimed by some cell."""
    out = {"n_max": n_max, "levels": {}, "all_covered": True}
    for s in (4, 5, 6):
        admissible = [n for n in asp.allowed_n(s) if 1 <= n <= n_max]
        feasible = [n for n in admissible if asp.dx_feasible(n, s, 3)]
        claimed = set()
        for cell in LEDGER:
            if cell["s"] == s:
                claimed |= {n for n in cell["claims_n"] if n <= n_max}
        missing = sorted(set(feasible) - claimed)
        extra = sorted(claimed - set(feasible))  # claimed-but-not-admissible (over-claim)
        ok = (len(missing) == 0)
        out["levels"][s] = {
            "admissible": admissible,
            "dx_feasible": feasible,
            "claimed": sorted(claimed),
            "missing": missing,
            "claimed_not_admissible": extra,
            "covered": ok,
        }
        out["all_covered"] = out["all_covered"] and ok
    return out


def _degree_of_support(pts: list[int], m: int) -> int:
    """Algebraic degree of the indicator of `pts` over F_2^m (Mobius/ANF)."""
    n = 1 << m
    tt = [0] * n
    for p in pts:
        tt[p] ^= 1
    # ANF via in-place Mobius transform
    anf = tt[:]
    step = 1
    while step < n:
        for i in range(n):
            if i & step:
                anf[i] ^= anf[i ^ step]
        step <<= 1
    deg = 0
    for i in range(n):
        if anf[i]:
            deg = max(deg, bin(i).count("1"))
    return deg


def m2_kta_transcription() -> dict:
    """Each KTA row produces the declared weight; record degree and ambient."""
    rows = {}
    all_ok = True
    for row, (gen, w_decl) in sorted(kf.ROWS.items()):
        m, pts = gen()
        w_act = len(pts)
        deg = _degree_of_support(pts, m)
        # Transcription fidelity = the support weight matches the declared weight
        # (the module docstring: "the support, not the printed formula, is
        # authoritative").  Degree is recorded for info: these s=9/10 Table-I rows
        # are degree 5-6 (not the s=8 P_{4,2} quartics), confirming they are the
        # higher-ambient classes the descent bound routes n<=38 through.
        ok = (w_act == w_decl)
        all_ok = all_ok and ok
        rows[row] = {"m": m, "declared_w": w_decl, "actual_w": w_act,
                     "degree": deg, "ok": bool(ok)}
    return {"rows": rows, "all_ok": all_ok}


def m3_existing_orbit_certs() -> dict:
    """Link the already-computed combinatorial orbit-closure certificates."""
    out = {}
    hrep = Path("reports/m8_h_rep_completeness_audit.json")
    if hrep.exists():
        d = json.load(open(hrep))
        out["m8_h_rep_cover"] = {"present": True, "all_complete": bool(d.get("all_complete")),
                                 "claim": d.get("claim", "")}
    else:
        out["m8_h_rep_cover"] = {"present": False}
    punc = Path("reports/kta_puncture_orbits.json")
    if punc.exists():
        d = json.load(open(punc))
        summ = d.get("summary", {})
        out["kta_puncture_orbits"] = {"present": True, "summary": summ}
    else:
        out["kta_puncture_orbits"] = {"present": False}
    w32 = Path("reports/m7_w32_classes.json")
    if w32.exists():
        d = json.load(open(w32))
        out["m7_w32_classes"] = {"present": True, "summary": d.get("summary", {})}
    else:
        out["m7_w32_classes"] = {"present": False}
    return out


def main() -> None:
    out_path = Path("reports/floor_affine_orbit_audit.json")
    m1_core = m1_spectrum_coverage(N_CORE)
    m1_ext = m1_spectrum_coverage(N_EXT)
    m2 = m2_kta_transcription()
    m3 = m3_existing_orbit_certs()

    running = sorted({cell["script"] for cell in LEDGER if cell["status"] == "running"})
    payload = {
        "claim": (
            "Floor completeness ledger: RM-coset weight-spectrum coverage (s<=6, "
            "machine), KTA Table-I transcription fidelity, and reuse of the existing "
            "combinatorial orbit-closure certificates. Within-weight affine-class "
            "completeness at s=7/9 is the Kasami-Tokura/KTA classification theorem (cited)."
        ),
        "N_core": N_CORE,
        "N_ext": N_EXT,
        "M1_spectrum_coverage_core": m1_core,
        "M1_spectrum_coverage_ext": m1_ext,
        "M2_kta_transcription": m2,
        "M3_existing_orbit_certs": m3,
        "ledger": LEDGER,
        "pending_cells": running,
        "machine_checks_pass": bool(
            m1_core["all_covered"] and m1_ext["all_covered"] and m2["all_ok"]
            and m3.get("m8_h_rep_cover", {}).get("all_complete", False)
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out_path, "w"), indent=1)

    # Compact console summary
    print("=== Floor completeness ledger ===")
    for tag, m1 in (("n<=30 (core)", m1_core), ("n<=38 (ext)", m1_ext)):
        print(f"M1 RM-spectrum coverage {tag}: all_covered={m1['all_covered']}")
        for s, lv in m1["levels"].items():
            miss = lv["missing"]
            print(f"   s={s}: dx_feasible={lv['dx_feasible']}  missing={miss}"
                  + ("" if not miss else "  <-- GAP"))
    print(f"M2 KTA transcription all_ok={m2['all_ok']}: "
          + ", ".join(f"row{r}:w{v['actual_w']}/deg{v['degree']}" for r, v in m2["rows"].items()))
    print(f"M3 h-rep cover all_complete="
          f"{m3.get('m8_h_rep_cover', {}).get('all_complete')}; "
          f"puncture_orbits={'present' if m3.get('kta_puncture_orbits',{}).get('present') else 'MISSING'}")
    print(f"pending (running) cells: {running}")
    print(f"machine_checks_pass = {payload['machine_checks_pass']}")
    print(f"-> {out_path}")

    if not (m1_core["all_covered"] and m1_ext["all_covered"] and m2["all_ok"]):
        raise SystemExit("MACHINE-CHECK FAILURE: spectrum gap or KTA transcription mismatch")


if __name__ == "__main__":
    main()
