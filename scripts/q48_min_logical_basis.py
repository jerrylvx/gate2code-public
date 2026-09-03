#!/usr/bin/env python3
"""Exact minimum-weight logical-basis diagnostic for Q48.

For a CSS matrix G=[K;S], the X-distance is the minimum Hamming weight in
rowspan(G) \ rowspan(S).  This script enumerates the seven nonzero quotient
cosets of rowspan(G)/rowspan(S) for the stored Q48 representative and compares
an exact minimum-weight quotient basis with the FG48 cap normal form.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate2code.ccz import compute_distances, f2_rank, verify_ch_conditions
from gate2code.identity import no_correction, verify_no_correction
from scripts.q48_k_shift_geometry import _certificate, _col_ints, _fiber_geometry, _shifted_K


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _wt(row: np.ndarray) -> int:
    return int(np.asarray(row, dtype=np.uint8).sum())


def _bits(x: int, width: int) -> np.ndarray:
    return np.array([(x >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _rank_labels(labels: list[int]) -> int:
    if not labels:
        return 0
    M = np.array([_bits(x, 3) for x in labels], dtype=np.uint8)
    return int(f2_rank(M.astype(np.int64)))


def _word_string(row: np.ndarray) -> str:
    return "".join(str(int(x)) for x in np.asarray(row, dtype=np.uint8))


def _coset_minima(K: np.ndarray, S: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label in range(1, 8):
        a = _bits(label, 3)
        base = (a @ K) & 1
        minima: list[tuple[int, np.ndarray]] = []
        best_weight = K.shape[1] + 1
        for shift in range(1 << S.shape[0]):
            b = _bits(shift, S.shape[0])
            word = (base ^ ((b @ S) & 1)).astype(np.uint8)
            weight = _wt(word)
            if weight < best_weight:
                best_weight = weight
                minima = [(shift, word)]
            elif weight == best_weight:
                minima.append((shift, word))
        records.append(
            {
                "logical_label": format(label, "03b"),
                "minimum_weight": int(best_weight),
                "num_minimum_representatives": len(minima),
                "minimum_shift_masks": [format(shift, f"0{S.shape[0]}b") for shift, _ in minima],
                "example_word": _word_string(minima[0][1]),
            }
        )
    return records


def _minimum_quotient_basis(cosets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    labels: list[int] = []
    for rec in sorted(
        cosets,
        key=lambda r: (int(r["minimum_weight"]), int(r["logical_label"], 2), r["minimum_shift_masks"][0]),
    ):
        label = int(rec["logical_label"], 2)
        if _rank_labels(labels + [label]) > _rank_labels(labels):
            selected.append(rec)
            labels.append(label)
        if len(selected) == 3:
            break
    return selected


def _rows_from_basis(K: np.ndarray, S: np.ndarray, basis: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for rec in basis:
        label = int(rec["logical_label"], 2)
        shift = int(rec["minimum_shift_masks"][0], 2)
        base = (_bits(label, 3) @ K) & 1
        row = (base ^ ((_bits(shift, S.shape[0]) @ S) & 1)).astype(np.uint8)
        rows.append(row)
    return np.vstack(rows).astype(np.uint8)


def _native_status(K: np.ndarray, S: np.ndarray) -> dict[str, Any]:
    G = np.vstack([K, S]).astype(np.uint8)
    nc = no_correction(G, k=3)
    verified = bool(nc.found) and nc.gamma is not None and bool(verify_no_correction(G, nc.gamma, k=3))
    return {
        "found": bool(nc.found),
        "status": nc.status,
        "verified": verified,
        "n_T": int(nc.n_T),
        "n_Tdag": int(nc.n_Tdag),
    }


def _fiber_record(name: str, K: np.ndarray, S: np.ndarray) -> dict[str, Any]:
    labels = _col_ints(K)
    s_cols = _col_ints(S)
    hist = np.bincount(labels, minlength=8).astype(int).tolist()
    fibers, geom = _fiber_geometry(labels, s_cols)
    G = np.vstack([K, S]).astype(np.uint8)
    return {
        "name": name,
        "row_weights": [_wt(row) for row in K],
        "label_histogram": {format(i, "03b"): int(hist[i]) for i in range(8)},
        "fiber_geometry_summary": geom,
        "fibers": fibers,
        "rank_G": int(f2_rank(G.astype(np.int64))),
        "distances": {"d_X": int(compute_distances(G, k=3)[0]), "d_Z": int(compute_distances(G, k=3)[1])},
        "CH": bool(verify_ch_conditions(G, k=3)[0]),
        "native": _native_status(K, S),
    }


def _weight_three_z_witness(K: np.ndarray, S: np.ndarray) -> dict[str, Any]:
    targets = [
        np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8),
        np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8),
        np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8),
    ]
    indices: list[int] = []
    for target in targets:
        matches = np.flatnonzero(np.all(S.T == target, axis=1))
        assert len(matches) == 1
        indices.append(int(matches[0]))

    z = np.zeros(S.shape[1], dtype=np.uint8)
    z[indices] = 1
    stabilizer_syndrome = (S @ z) & 1
    logical_syndrome = (K @ z) & 1
    assert _wt(z) == 3
    assert not stabilizer_syndrome.any()
    assert logical_syndrome.any()
    return {
        "column_indices_1_based": [index + 1 for index in indices],
        "block_coordinates": ["(0,(1,0))", "(0,(0,1))", "(0,(1,1))"],
        "logical_labels": [K[:, index].astype(int).tolist() for index in indices],
        "stabilizer_syndrome": stabilizer_syndrome.astype(int).tolist(),
        "logical_syndrome": logical_syndrome.astype(int).tolist(),
        "weight": 3,
    }


def build_payload() -> dict[str, Any]:
    data = np.load(ROOT / "code_48_3_3_fg48.npz")
    K = np.asarray(data["K"], dtype=np.uint8) & 1
    S = np.asarray(data["S_fg48"], dtype=np.uint8) & 1
    G = np.vstack([K, S]).astype(np.uint8)

    cosets = _coset_minima(K, S)
    min_basis = _minimum_quotient_basis(cosets)
    K_min = _rows_from_basis(K, S, min_basis)
    K_cap = _shifted_K(K, S, (0b100101, 0b010100, 0b010100))

    weight16_labels = [
        rec["logical_label"] for rec in cosets if int(rec["minimum_weight"]) == int(compute_distances(G, k=3)[0])
    ]

    return {
        "source": "code_48_3_3_fg48.npz",
        "definition": {
            "d_X": "min weight in rowspan(G) \\ rowspan(S)",
            "logical_basis": "basis of the quotient rowspan(G)/rowspan(S)",
            "minimum_weight_logical_basis": "lexicographically minimum row-weight basis among quotient representatives",
        },
        "ranks": {
            "G": int(f2_rank(G.astype(np.int64))),
            "S": int(f2_rank(S.astype(np.int64))),
        },
        "distances": {"d_X": int(compute_distances(G, k=3)[0]), "d_Z": int(compute_distances(G, k=3)[1])},
        "coset_minima": cosets,
        "weight_16_logical_labels": weight16_labels,
        "weight_16_label_rank": _rank_labels([int(x, 2) for x in weight16_labels]),
        "minimum_quotient_basis": min_basis,
        "minimum_quotient_basis_matrix": K_min.tolist(),
        "z_logical_witness": _weight_three_z_witness(K, S),
        "comparison": [
            _fiber_record("original K", K, S),
            _fiber_record("cap-normal K", K_cap, S),
            _fiber_record("minimum-weight quotient basis", K_min, S),
        ],
        "cap_normal_certificate": _certificate(K_cap, S),
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = [
        "| label | min wt | # min reps | shifts |",
        "|---|---:|---:|---|",
    ]
    for rec in payload["coset_minima"]:
        shifts = ", ".join(f"`{x}`" for x in rec["minimum_shift_masks"])
        rows.append(
            f"| `{rec['logical_label']}` | {rec['minimum_weight']} | "
            f"{rec['num_minimum_representatives']} | {shifts} |"
        )

    comp = [
        "| representative | K row weights | K-fiber sizes | cap fibers | projective lines inside fibers | CH | native |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for rec in payload["comparison"]:
        hist = ",".join(str(v) for v in rec["label_histogram"].values())
        comp.append(
            f"| {rec['name']} | {rec['row_weights']} | `{hist}` | "
            f"{rec['fiber_geometry_summary']['cap_fibers']} | "
            f"{rec['fiber_geometry_summary']['total_projective_lines_inside_fibers']} | "
            f"{rec['CH']} | {rec['native']['verified']} |"
        )

    basis_rows = "\n".join(
        " ".join(str(int(x)) for x in row) for row in payload["minimum_quotient_basis_matrix"]
    )

    text = "\n".join(
        [
            "# Q48 minimum-weight logical basis",
            "",
            "This is an exact quotient-space computation.  The displayed logical rows are representatives of",
            "`rowspan(G) / rowspan(S)`; adding stabilizer rows changes representatives but not the CSS code.",
            "",
            f"- `d_X={payload['distances']['d_X']}`, `d_Z={payload['distances']['d_Z']}`",
            f"- weight-16 logical labels: `{payload['weight_16_logical_labels']}`",
            f"- rank of the weight-16 labels: `{payload['weight_16_label_rank']}`",
            f"- weight-three Z-logical columns: `{payload['z_logical_witness']['column_indices_1_based']}`",
            f"- its logical syndrome: `{payload['z_logical_witness']['logical_syndrome']}`",
            "- consequence: no basis can have all three rows of weight 16; a lexicographically minimum basis has weights `[16,16,18]`.",
            "- terminology: a projective cap is a point set with no three collinear points; in `PG(5,2)` this means no full line `{a,b,a+b}` lies inside the set.",
            "",
            "## Coset minima",
            "",
            *rows,
            "",
            "## Minimum quotient-basis rows",
            "",
            "```",
            basis_rows,
            "```",
            "",
            "## Comparison with cap normal form",
            "",
            *comp,
            "",
            "The cap-normal representative optimizes the column-fiber geometry; it is not a minimum-weight logical basis.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=ROOT / "reports/q48_min_logical_basis.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports/q48_min_logical_basis.md")
    args = parser.parse_args()

    payload = build_payload()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    write_markdown(payload, args.markdown)

    print(f"wrote {args.json}")
    print(f"wrote {args.markdown}")
    print("minimum quotient-basis row weights:", [rec["minimum_weight"] for rec in payload["minimum_quotient_basis"]])
    print("weight-16 label rank:", payload["weight_16_label_rank"])
    print("weight-three Z-logical columns:", payload["z_logical_witness"]["column_indices_1_based"])


if __name__ == "__main__":
    main()
