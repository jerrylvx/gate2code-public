#!/usr/bin/env python3
"""Canonical block forms [A ... A | B] for the FG48 flat-complement code family.

For each (m,a) in the family table we use the group-algebra / flat-complement tooling
(gate2code.projective_qc) to coordinatize the X-stabilizer columns as (u,w) with u in F_2^a and
w in F_2^q\\{0} (q = m-a), order the columns into the 2^q-1 W-blocks (Singer-cycle order), and
assemble the canonical block-form generator: the top a rows are the U-coordinate functionals (the
a x 2^a table A repeated across the blocks), the bottom q rows are the W-coordinate functionals
(constant w_j per block). We record the (column permutation p, row-basis change T) relating the
block form to the raw complement generator, the group-algebra generators (L_i, Q_j), and run
verifications. Positive control: (6,4) reproduces the FG48 stabilizer code generator_fg48().

Writes reports/family_block_forms.json and prints a bounded LaTeX-ready summary.
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

from gate2code.ccz import f2_inv, f2_rank
from gate2code.fg48.code import code_weight_distribution, generator_fg48, row_space_equal
from gate2code.ks_fiber import column_ints
from gate2code.projective_qc import (
    _apply_matrix_columns,
    _label_coeff_maps,
    flat_complement_qc_report,
    permute_columns,
)

FAMILY: list[tuple[int, int]] = [(6, 4), (7, 6), (7, 5), (7, 4), (8, 7)]


def subspace_complement(m: int, a: int) -> np.ndarray:
    """X-stabilizer generator of PG(m-1,2) minus the a-flat (census builder)."""
    submask = (1 << a) - 1
    cols = [v for v in range(1, 1 << m) if (v & ~submask) != 0]
    return np.array([[(v >> i) & 1 for v in cols] for i in range(m)], dtype=np.uint8)


def w_block_order(q: int, singer: Any) -> list[int]:
    """Nonzero w in F_2^q in Singer-cycle order (single block when q<2)."""
    if q < 2:
        return [1]
    orbit: list[int] = []
    x = 1
    for _ in range((1 << q) - 1):
        orbit.append(x)
        x = _apply_matrix_columns(tuple(int(c) for c in singer), x)
    return orbit


def solve_row_basis(Sp: np.ndarray, Bform: np.ndarray) -> np.ndarray | None:
    """Return T in GL_m(F_2) with Bform == (T @ Sp) % 2, or None."""
    m, n = Sp.shape
    chosen: list[int] = []
    acc = np.zeros((m, 0), dtype=np.uint8)
    for j in range(n):
        cand = np.concatenate([acc, Sp[:, j : j + 1]], axis=1)
        if f2_rank(cand.astype(int)) > acc.shape[1]:
            acc = cand
            chosen.append(j)
            if len(chosen) == m:
                break
    if len(chosen) < m:
        return None
    inv = f2_inv(Sp[:, chosen].astype(int))
    if inv is None:
        return None
    T = (Bform[:, chosen].astype(int) @ inv) % 2
    return T.astype(np.uint8)


def block_form_record(m: int, a: int) -> dict[str, Any]:
    S = subspace_complement(m, a)
    rep = flat_complement_qc_report(S)
    q = int(rep["quotient_dim"])
    assert int(rep["deleted_subspace_dim"]) == a, (rep["deleted_subspace_dim"], a)
    U_basis = [int(x) for x in rep["U_basis_column_labels"]]
    W_basis = [int(x) for x in rep["W_basis_column_labels"]]
    _, coeff_to_label = _label_coeff_maps(U_basis + W_basis, m)
    labels = column_ints(S)
    index_of = {lab: j for j, lab in enumerate(labels)}

    wlist = w_block_order(q, rep.get("outer_singer_columns"))
    n = int(S.shape[1])
    assert n == (1 << m) - (1 << a)
    assert len(wlist) == (1 << q) - 1

    p = [0] * n
    coeff_by_new = [0] * n
    for blk, w in enumerate(wlist):
        for u in range(1 << a):
            coeff = u | (w << a)
            old = index_of[coeff_to_label[coeff]]
            new = blk * (1 << a) + u
            p[old] = new
            coeff_by_new[new] = coeff

    Bform = np.zeros((m, n), dtype=np.uint8)
    for new, coeff in enumerate(coeff_by_new):
        for i in range(m):
            Bform[i, new] = (coeff >> i) & 1

    Sp = permute_columns(S, p)
    rowspace_equal = bool(row_space_equal(Bform, Sp))
    T = solve_row_basis(Sp, Bform)
    transform_ok = T is not None and bool(
        np.array_equal((T.astype(int) @ Sp.astype(int)) % 2, Bform)
    )

    A = Bform[:a, : 1 << a]
    w_pattern = [[int((w >> j) & 1) for j in range(q)] for w in wlist]
    wd = code_weight_distribution(Bform)
    nonzero_weights = sorted(w for w in wd if w != 0)

    rec: dict[str, Any] = {
        "m": m,
        "a": a,
        "q": q,
        "n": n,
        "stabilizer_dim": m,
        "blocks": len(wlist),
        "block_width": 1 << a,
        "algebra": f"F_2[C_2^{a}] (x) F_2[C_{(1 << q) - 1}]",
        "group_algebra_generators": {
            "L_i": f"(sum_{{u_i=1}} x^u) . N  for i=1..{a}  (N = norm over Singer C_{(1 << q) - 1})",
            "Q_j": f"w_j label row, constant per block  for j=1..{q}",
            "norm_order": (1 << q) - 1,
        },
        "A_table_shape": [a, 1 << a],
        "A": A.tolist(),
        "W_block_pattern": w_pattern,
        "weight_distribution": {str(int(k)): int(v) for k, v in sorted(wd.items())},
        "two_weight": len(nonzero_weights) == 2,
        "nonzero_weights": nonzero_weights,
        "expected_two_weights": sorted({1 << (m - 1), (1 << (m - 1)) - (1 << (a - 1))}),
        "flat_complement": bool(rep.get("is_flat_complement")),
        "rowspace_equal_to_raw": rowspace_equal,
        "transform_reproduces_S": transform_ok,
        "column_permutation_p": p,
        "row_basis_change_T": (T.tolist() if T is not None else None),
        "block_form_matrix": Bform.tolist(),
    }
    if (m, a) == (6, 4):
        rec["positive_control_matches_generator_fg48"] = bool(
            row_space_equal(Bform, generator_fg48())
        )
    return rec


def latex_block_form(rec: dict[str, Any]) -> str:
    a, q, blocks = rec["a"], rec["q"], rec["blocks"]
    Arep = " & ".join(["A"] * blocks)
    lines = [f"% (m,a)=({rec['m']},{a}): n={rec['n']}, q={q}, {blocks} block(s), A is {a}x{1<<a}"]
    lines.append("S_{(%d,%d)} = \\begin{bmatrix} %s \\\\ B \\end{bmatrix}" % (rec["m"], a, Arep))
    # explicit A only when narrow enough to be readable
    if (1 << a) <= 16:
        rowstrs = [" & ".join(str(b) for b in row) for row in rec["A"]]
        body = " \\\\\n".join(rowstrs)
        lines.append("A = \\begin{bmatrix}\n" + body + "\n\\end{bmatrix}")
    else:
        lines.append(f"A = (the {a}x{1<<a} table enumerating F_2^{a}; full matrix in the JSON report)")
    # W-rows: one row per W-coordinate, constant w_j across each block's 2^a columns
    if q <= 3:
        wrows = []
        for j in range(q):
            entries = " & ".join(f"{pat[j]}\\cdot\\mathbf 1" for pat in rec["W_block_pattern"])
            wrows.append(entries)
        lines.append("B = \\begin{bmatrix}\n" + " \\\\\n".join(wrows) + "\n\\end{bmatrix}")
    return "\n".join(lines)


def main() -> None:
    records = [block_form_record(m, a) for (m, a) in FAMILY]
    payload = {
        "metadata": {
            "description": "canonical block forms [A...A | B] for the FG48 flat-complement family",
            "construction": "columns U x (F_2^q \\ 0), q=m-a; top a rows = U-coordinates (A repeated), "
            "bottom q rows = W-coordinates (constant per block); algebra F_2[C_2^a] (x) F_2[C_{2^q-1}]",
        },
        "records": records,
    }
    out = ROOT / "reports" / "family_block_forms.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}\n")
    for rec in records:
        flags = (
            f"flat={rec['flat_complement']} rowspace_eq={rec['rowspace_equal_to_raw']} "
            f"T_ok={rec['transform_reproduces_S']} two_weight={rec['two_weight']}"
            f"{' fg48_ctrl=' + str(rec.get('positive_control_matches_generator_fg48')) if 'positive_control_matches_generator_fg48' in rec else ''}"
        )
        print(
            f"(m,a)=({rec['m']},{rec['a']}): n={rec['n']} q={rec['q']} blocks={rec['blocks']} "
            f"A={rec['A_table_shape'][0]}x{rec['A_table_shape'][1]} "
            f"weights={rec['nonzero_weights']} (expected {rec['expected_two_weights']}) | {flags}"
        )
    print()
    print(latex_block_form(records[0]))  # FG48 (6,4) as the worked example


if __name__ == "__main__":
    main()
