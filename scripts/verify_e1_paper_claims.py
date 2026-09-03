#!/usr/bin/env python3
"""Independently re-verify every load-bearing claim of the expedited Gates paper.

Loads the stored Q48 representative and re-derives, from scratch, the
chain .npz -> [[48,3,3]] native C=I certificate -> FG48 projective two-weight
spine -> poster->block certificate -> full-G decoration ANF.  Each claim is
recomputed with the existing verified modules (gate2code.{ccz,identity,fg48,
ks_fiber,decreasing_monomial}) and compared against the value documented in the
handoff / notebook.  A mismatch is reported verbatim (expected vs computed) and
makes the script exit non-zero -- we never silently adopt computed values.

Run:
    python scripts/verify_e1_paper_claims.py
Writes reports/e1_paper_claims_verification.json and prints a PASS/FAIL table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate2code.ccz import compute_distances, f2_rank, verify_ch_conditions
from gate2code.code_data import build_48_code
from gate2code.decreasing_monomial import anf_of_truth_table
from gate2code.fg48.code import (code_weight_distribution, generator_fg48,
                                 is_projective_generator, row_space_equal)
from gate2code.identity import no_correction, verify_no_correction
from gate2code.ks_fiber import column_ints

E1_PATH = ROOT / "code_48_3_3_fg48.npz"

CLAIMS: list[dict[str, Any]] = []


def _norm(x: Any) -> Any:
    """Make values JSON-stable and comparable (ints, sorted-key dicts, lists)."""
    if isinstance(x, dict):
        return {int(k): int(v) for k, v in sorted(x.items())}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (tuple, list)):
        return [int(v) if isinstance(v, (np.integer, bool)) else v for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def record(cid: str, desc: str, expected: Any, computed: Any) -> bool:
    exp, comp = _norm(expected), _norm(computed)
    ok = exp == comp
    CLAIMS.append({"id": cid, "desc": desc, "expected": exp,
                   "computed": comp, "pass": bool(ok)})
    return ok


# ---------------------------------------------------------------------------
# Group A -- native [[48,3,3]] certificate from the archived .npz
# ---------------------------------------------------------------------------
def group_a(G: np.ndarray) -> None:
    K, S = G[:3], G[3:]
    record("A1-shape", "G is 9x48", list(G.shape), [9, 48])
    record("A1-rankG", "f2_rank(G) = 9 (k+s independent)", 9, f2_rank(G.astype(int)))
    record("A1-rankS", "f2_rank(S) = 6 (s=6 stabilisers)", 6, f2_rank(S.astype(int)))
    record("A1-k", "K independent mod S -> k=3",
           9, f2_rank(np.vstack([S, K]).astype(int)))

    dX, dZ = compute_distances(G, k=3)
    record("A2-dist", "(d_X, d_Z) = (16, 3) -> [[48,3,3]]", (16, 3), (int(dX), int(dZ)))

    ch_ok = verify_ch_conditions(G, k=3)[0]
    record("A3-ch", "verify_ch_conditions True (quasi-transversal CCZ)", True, bool(ch_ok))

    nc = no_correction(G, k=3)
    native = bool(nc.found) and (nc.gamma is not None) and \
        bool(verify_no_correction(G, nc.gamma, k=3))
    record("A4-ci", "native-CSS C=I (no Clifford correction), brute-verified", True, native)
    record("A5-Tpattern", "Gamma sign vector = 26 T + 22 T-dagger",
           (26, 22), (int(nc.n_T), int(nc.n_Tdag)))

    # A6 negative control: the hardcoded build_48_code() is NOT the native e1.
    Gh = build_48_code()["G"]
    nc_h = no_correction(Gh, k=3)
    record("A6-negctrl", "hardcoded build_48_code() fails native C=I", False, bool(nc_h.found))
    CLAIMS[-1]["note"] = f"build_48_code status={nc_h.status}"


# ---------------------------------------------------------------------------
# Group B -- FG48 spine = projective two-weight PG(5,2)\PG(3,2)
# ---------------------------------------------------------------------------
def group_b(S: np.ndarray) -> None:
    record("B1-proj", "npz S is projective (distinct nonzero cols)",
           True, is_projective_generator(S))
    record("B1-twoweight", "WE(rowspan S) = {0:1, 24:60, 32:3}",
           {0: 1, 24: 60, 32: 3}, code_weight_distribution(S))
    record("B1-rankS", "f2_rank(S) = 6", 6, f2_rank(S.astype(int)))
    # B2 (npz S <-> abstract module) is established inside the canonical pipeline
    # via S_ordered == generator_fg48(); recorded there.
    # B3 (pytest tests/test_fg48.py) is run separately by the harness.


# ---------------------------------------------------------------------------
# Canonical FG48 pipeline (faithful port of notebook cells 5/7/9/14)
# ---------------------------------------------------------------------------
def label_to_bits(x: int, rows: int = 6) -> np.ndarray:
    return np.array([(x >> (rows - 1 - i)) & 1 for i in range(rows)], dtype=np.uint8)


def matrix_from_labels(labels: list[int], rows: int = 6) -> np.ndarray:
    M = np.zeros((rows, len(labels)), dtype=np.uint8)
    for j, x in enumerate(labels):
        M[:, j] = label_to_bits(x, rows)
    return M


def build_canonical(K: np.ndarray, S: np.ndarray):
    """Return (canonical_indices, S_ordered, S_fg48, G_canonical)."""
    s_labels = column_ints(S)
    missing = sorted(set(range(1, 64)) - set(s_labels))

    U_basis: list[int] = []
    for x in missing:
        if f2_rank(matrix_from_labels(U_basis + [x]).astype(int)) > len(U_basis):
            U_basis.append(x)
        if len(U_basis) == 4:
            break
    W_basis: list[int] = []
    for x in s_labels:
        if f2_rank(matrix_from_labels(U_basis + W_basis + [x]).astype(int)) \
                > len(U_basis) + len(W_basis):
            W_basis.append(x)
        if len(W_basis) == 2:
            break
    basis = U_basis + W_basis

    label_to_coeff: dict[int, int] = {}
    coeff_to_label: dict[int, int] = {}
    for coeff in range(64):
        v = np.zeros(6, dtype=np.uint8)
        for i, b in enumerate(basis):
            if (coeff >> i) & 1:
                v ^= label_to_bits(b)
        label = 0
        for bit in v:
            label = (label << 1) | int(bit)
        label_to_coeff[label] = coeff
        coeff_to_label[coeff] = label

    col_to_index = {x: j for j, x in enumerate(s_labels)}
    canonical_indices = []
    for w in (1, 2, 3):
        for u in range(16):
            label = coeff_to_label[u | (w << 4)]
            canonical_indices.append(col_to_index[label])

    S_ordered = S[:, canonical_indices]
    S_fg48 = generator_fg48()
    K_ordered = K[:, canonical_indices]
    G_canonical = np.vstack([K_ordered, S_fg48]).astype(np.uint8) % 2
    return canonical_indices, S_ordered, S_fg48, G_canonical


def indicator_summary(M: np.ndarray) -> dict:
    """Set-indicator truth table -> ANF weight/degree/degree-spectrum."""
    M = np.asarray(M, dtype=np.uint8) % 2
    rows = M.shape[0]
    f = np.zeros(1 << rows, dtype=np.uint8)
    for col in M.T:
        idx = 0
        for bit in col:
            idx = (idx << 1) | int(bit)
        f[idx] = 1
    anf = anf_of_truth_table(f)
    # ANF degree / degree-spectrum via 3.9-safe popcount (module uses int.bit_count,
    # which is 3.10+); identical result, no interpreter dependency.
    degs = [bin(idx).count("1") for idx, c in enumerate(anf) if c]
    spectrum: dict[int, int] = {}
    for d in degs:
        spectrum[d] = spectrum.get(d, 0) + 1
    return {"weight": int(f.sum()), "degree": max(degs, default=-1),
            "spectrum": spectrum}


# ---------------------------------------------------------------------------
# Group C -- poster -> block certificate
# ---------------------------------------------------------------------------
def group_c(S_npz: np.ndarray, S_ordered: np.ndarray, S_fg48: np.ndarray) -> None:
    S_poster_rows = [
        "000000000000000000000000111111111111111111111111",
        "000000000000111111111111000000000000111111111111",
        "000011111111000000001111000011111111000000001111",
        "001100001111000011110011001100001111000011110011",
        "010100110011001100110101010100110011001100110101",
        "110001010101010101010011110001010101010101010011",
    ]
    S_poster = np.array([[int(b) for b in r] for r in S_poster_rows], dtype=np.uint8)
    p1 = [1, 25, 23, 47, 2, 26, 24, 48, 21, 45, 3, 27, 22, 46, 4, 28,
          5, 29, 17, 41, 7, 31, 19, 43, 14, 38, 10, 34, 16, 40, 12, 36,
          6, 30, 18, 42, 8, 32, 20, 44, 13, 37, 9, 33, 15, 39, 11, 35]
    p = [j - 1 for j in p1]
    T = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0]],
                 dtype=np.uint8)
    S_from_poster = (T @ S_poster[:, p]) % 2

    record("C1-poster-block", "T @ S_poster[:,p] == S_FG48 (exact)",
           True, bool(np.array_equal(S_from_poster, S_fg48)))
    record("C1-rowspace", "rowspan(S_poster[:,p]) == rowspan(S_FG48)",
           True, bool(row_space_equal(S_poster[:, p], S_fg48)))
    # B2 / C2: equality is at the level of the stabiliser CODE (row space), the
    # physical invariant.  npz S reordered and generator_fg48() span the same
    # code; they differ by a unique invertible 6x6 row-basis R (S_fg48 = R.S_ord),
    # which is exactly the canonical basis the full-G ANF (D2) is computed in.
    record("B2-npz-module", "npz S (canonical order) spans same code as generator_fg48()",
           True, bool(row_space_equal(S_ordered, S_fg48)))
    CLAIMS[-1]["note"] = ("exact-equal only up to a 6x6 row-basis change "
                          f"(array_equal={bool(np.array_equal(S_ordered, S_fg48))})")
    record("C2-poster-eq-npz", "poster and npz give the same canonical stabiliser code",
           True, bool(row_space_equal(S_from_poster, S_ordered)))


# ---------------------------------------------------------------------------
# Group D -- full-G decoration ANF
# ---------------------------------------------------------------------------
def group_d(S_fg48: np.ndarray, G_canonical: np.ndarray) -> None:
    sd = indicator_summary(S_fg48)
    record("D1-S-weight", "S_FG48 indicator weight = 48", 48, sd["weight"])
    record("D1-S-degree", "S_FG48 indicator ANF degree = 2", 2, sd["degree"])
    record("D1-S-spectrum", "S_FG48 degree spectrum = {1:2, 2:1}",
           {1: 2, 2: 1}, sd["spectrum"])

    gd = indicator_summary(G_canonical)
    record("D2-G-weight", "full [K;S_FG48] indicator weight = 48", 48, gd["weight"])
    record("D2-G-degree", "full [K;S_FG48] indicator ANF degree = 6", 6, gd["degree"])
    record("D2-G-spectrum", "full [K;S_FG48] spectrum = {2:2,3:16,4:40,5:35,6:1}",
           {2: 2, 3: 16, 4: 40, 5: 35, 6: 1}, gd["spectrum"])


def main() -> int:
    if not E1_PATH.exists():
        print(f"FATAL: stored representative missing: {E1_PATH}", file=sys.stderr)
        return 2
    G = np.load(E1_PATH)["G"].astype(np.uint8) % 2
    K, S = G[:3], G[3:]

    group_a(G)
    group_b(S)
    _, S_ordered, S_fg48, G_canonical = build_canonical(K, S)
    group_c(S, S_ordered, S_fg48)
    group_d(S_fg48, G_canonical)

    # Report
    width = max(len(c["id"]) for c in CLAIMS)
    print(f"{'CLAIM':<{width}}  RESULT  DESCRIPTION")
    n_fail = 0
    for c in CLAIMS:
        tag = "PASS " if c["pass"] else "FAIL*"
        if not c["pass"]:
            n_fail += 1
        print(f"{c['id']:<{width}}  {tag}   {c['desc']}")
        if not c["pass"]:
            print(f"{'':<{width}}         expected={c['expected']!r}  computed={c['computed']!r}")

    out = ROOT / "reports" / "e1_paper_claims_verification.json"
    out.write_text(json.dumps(
        {"source": str(E1_PATH.relative_to(ROOT)),
         "n_claims": len(CLAIMS), "n_fail": n_fail, "claims": CLAIMS},
        indent=2) + "\n")
    print(f"\n{len(CLAIMS) - n_fail}/{len(CLAIMS)} claims PASS  ->  wrote {out.relative_to(ROOT)}")
    if n_fail:
        print(f"*** {n_fail} FAIL -- reconcile before filling the paper ***")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
