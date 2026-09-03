#!/usr/bin/env python3
"""Search stabilizer-shifted Q48 logical representatives with simple fibers.

The code is fixed: ``S`` is the canonical FG48 stabilizer basis and we only
replace each logical row by ``K_i + b_i S``.  This keeps the CSS code and logical
operators unchanged, but can make the logical-label fibers easier to inspect.

Run:
    python3 scripts/q48_k_shift_geometry.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate2code.ccz import compute_distances, f2_rank, verify_ch_conditions
from gate2code.decreasing_monomial import anf_of_truth_table
from gate2code.identity import no_correction, verify_no_correction
from scripts.verify_e1_paper_claims import indicator_summary


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _col_ints(M: np.ndarray) -> np.ndarray:
    labels = np.zeros(M.shape[1], dtype=np.uint8)
    for row in np.asarray(M, dtype=np.uint8) & 1:
        labels = (labels << 1) | row.astype(np.uint8)
    return labels


def _parity_table() -> np.ndarray:
    tab = np.zeros((64, 64), dtype=np.uint8)
    for mask in range(64):
        for x in range(64):
            tab[mask, x] = bin(mask & x).count("1") & 1
    return tab


def _rank_points(points: list[int]) -> int:
    if not points:
        return 0
    M = np.array([[(x >> (5 - i)) & 1 for i in range(6)] for x in points], dtype=np.int64)
    return int(f2_rank(M))


def _line_count(points: list[int]) -> int:
    pts = set(int(x) for x in points)
    total = 0
    for a, b in combinations(sorted(pts), 2):
        c = a ^ b
        if c in pts and a < b < c:
            total += 1
    return int(total)


def _fiber_geometry(labels: np.ndarray, s_cols: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    total_lines = 0
    cap_fibers = 0
    flat_fibers = 0
    rank_sum = 0
    for label in range(8):
        pts = [int(x) for x in s_cols[labels == label]]
        rank = _rank_points(pts)
        lines = _line_count(pts)
        is_flat = len(pts) == (1 << rank) - 1
        is_cap = lines == 0
        total_lines += lines
        cap_fibers += int(is_cap)
        flat_fibers += int(is_flat)
        rank_sum += rank
        records.append({
            "label": format(label, "03b"),
            "size": int(len(pts)),
            "rank": int(rank),
            "projective_line_count": int(lines),
            "projective_cap": bool(is_cap),
            "projective_flat": bool(is_flat),
        })
    return records, {
        "total_projective_lines_inside_fibers": int(total_lines),
        "cap_fibers": int(cap_fibers),
        "projective_flat_fibers": int(flat_fibers),
        "rank_sum": int(rank_sum),
    }


def _block_counts(labels: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for block, w in enumerate(("01", "10", "11")):
        local = labels[16 * block:16 * (block + 1)]
        counts = np.bincount(local, minlength=8).astype(int).tolist()
        out.append({"w": w, "counts": {format(i, "03b"): int(counts[i]) for i in range(8)}})
    return out


def _degree_spectrum(vals: np.ndarray) -> dict[str, Any]:
    anf = anf_of_truth_table(np.asarray(vals, dtype=np.uint8) & 1)
    spectrum: Counter[int] = Counter()
    for idx, coeff in enumerate(anf):
        if int(coeff):
            spectrum[bin(idx).count("1")] += 1
    return {
        "weight": int(np.asarray(vals, dtype=np.uint8).sum()),
        "degree": int(max(spectrum.keys(), default=-1)),
        "spectrum": {str(k): int(v) for k, v in sorted(spectrum.items())},
    }


def _per_block_kappa_anf(K: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for block, w in enumerate(("01", "10", "11")):
        rows = []
        for i in range(3):
            vals = K[i, 16 * block:16 * (block + 1)]
            rec = _degree_spectrum(vals)
            rec["bits"] = "".join(str(int(x)) for x in vals)
            rows.append(rec)
        out.append({"w": w, "kappa_rows": rows})
    return out


def _mask_to_row(mask: int) -> list[int]:
    return [(mask >> (5 - i)) & 1 for i in range(6)]


def _shifted_K(K: np.ndarray, S: np.ndarray, masks: tuple[int, int, int]) -> np.ndarray:
    B = np.array([_mask_to_row(m) for m in masks], dtype=np.uint8)
    return (K ^ ((B @ S) & 1)).astype(np.uint8)


def _certificate(K: np.ndarray, S: np.ndarray) -> dict[str, Any]:
    G = np.vstack([K, S]).astype(np.uint8) & 1
    d_x, d_z = compute_distances(G, k=3)
    nc = no_correction(G, k=3)
    native = bool(nc.found) and nc.gamma is not None and bool(verify_no_correction(G, nc.gamma, k=3))
    return {
        "rank_G": int(f2_rank(G.astype(np.int64))),
        "rank_S": int(f2_rank(S.astype(np.int64))),
        "d_X": int(d_x),
        "d_Z": int(d_z),
        "CH": bool(verify_ch_conditions(G, k=3)[0]),
        "native_C_eq_I": bool(native),
        "T_pattern": {"n_T": int(nc.n_T), "n_Tdag": int(nc.n_Tdag)} if nc.found else None,
        "full_G_indicator": indicator_summary(G),
    }


def build_payload() -> dict[str, Any]:
    data = np.load(ROOT / "code_48_3_3_fg48.npz")
    K = np.asarray(data["K"], dtype=np.uint8) & 1
    S = np.asarray(data["S_fg48"], dtype=np.uint8) & 1
    s_cols = _col_ints(S)
    k_labels = _col_ints(K)
    parity = _parity_table()

    original_fibers, original_geom = _fiber_geometry(k_labels, s_cols)
    original_hist = np.bincount(k_labels, minlength=8).astype(int).tolist()

    best: tuple[Any, ...] | None = None
    balanced_representatives = 0
    balanced_all_cap_representatives = 0
    cap_histogram: Counter[int] = Counter()
    target_sorted = [5, 5, 5, 5, 7, 7, 7, 7]

    for b0 in range(64):
        p0 = parity[b0, s_cols]
        for b1 in range(64):
            p1 = parity[b1, s_cols]
            prefix = (p0 << 2) | (p1 << 1)
            for b2 in range(64):
                labels = k_labels ^ (prefix | parity[b2, s_cols])
                hist = np.bincount(labels, minlength=8).astype(int).tolist()
                balance = (max(hist) - min(hist), sum((x - 6) ** 2 for x in hist), tuple(sorted(hist)))
                if sorted(hist) != target_sorted:
                    continue
                balanced_representatives += 1
                fibers, geom = _fiber_geometry(labels, s_cols)
                cap_histogram[geom["cap_fibers"]] += 1
                if geom["cap_fibers"] == 8:
                    balanced_all_cap_representatives += 1
                score = (
                    balance,
                    -geom["cap_fibers"],
                    geom["total_projective_lines_inside_fibers"],
                    geom["rank_sum"],
                    tuple(hist),
                )
                candidate = (score, (b0, b1, b2), hist, fibers, geom, _block_counts(labels))
                if best is None or score < best[0]:
                    best = candidate

    if best is None:
        raise RuntimeError("no balanced representative found")

    _, masks, hist, fibers, geom, blocks = best
    K_shifted = _shifted_K(K, S, masks)

    return {
        "source": "code_48_3_3_fg48.npz",
        "operation": "K_i -> K_i + b_i S; S basis and column order fixed",
        "search_space": {
            "stabilizer_shift_matrices": 1 << 18,
            "balanced_representatives": int(balanced_representatives),
            "balanced_all_cap_representatives": int(balanced_all_cap_representatives),
            "balanced_cap_fiber_histogram": {str(k): int(v) for k, v in sorted(cap_histogram.items())},
        },
        "original": {
            "histogram": {format(i, "03b"): int(original_hist[i]) for i in range(8)},
            "geometry_summary": original_geom,
            "fibers": original_fibers,
            "per_block_kappa_anf": _per_block_kappa_anf(K),
            "certificate": _certificate(K, S),
        },
        "best_shift": {
            "B_masks": [format(m, "06b") for m in masks],
            "B_matrix": [_mask_to_row(m) for m in masks],
            "histogram": {format(i, "03b"): int(hist[i]) for i in range(8)},
            "geometry_summary": geom,
            "fibers": fibers,
            "block_counts": blocks,
            "per_block_kappa_anf": _per_block_kappa_anf(K_shifted),
            "K_shifted": K_shifted.astype(int).tolist(),
            "certificate": _certificate(K_shifted, S),
        },
        "interpretation": [
            "Uniform 6-per-label fibers are impossible under stabilizer shifts, but many representatives attain the best 5/7 balance.",
            "The selected representative makes every logical-label fiber a projective cap inside the FG48 point set: no fiber contains a full projective line {a,b,a+b}.",
            "This cap normal form is a better finite-geometric view of the full-G indicator than the archived K basis, while preserving the same [[48,3,3]] native certificate.",
        ],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    b = payload["best_shift"]
    lines = [
        "# Q48 stabilizer-shifted K normal form",
        "",
        "Allowed operation: keep the canonical `S_FG48` basis and replace each logical row by `K_i + b_i S`.",
        "This changes logical representatives, not the CSS code.",
        "",
        "## Search summary",
        "",
        f"- searched `2^18 = {payload['search_space']['stabilizer_shift_matrices']}` stabilizer shifts",
        f"- best-balance representatives (`5,5,5,5,7,7,7,7`): {payload['search_space']['balanced_representatives']}",
        f"- best-balance representatives with all 8 fibers caps: {payload['search_space']['balanced_all_cap_representatives']}",
        f"- cap-fiber histogram among balanced representatives: `{payload['search_space']['balanced_cap_fiber_histogram']}`",
        "",
        "## Selected shift",
        "",
        f"- `B` row masks: `{b['B_masks']}`",
        f"- label histogram: `{b['histogram']}`",
        f"- geometry summary: `{b['geometry_summary']}`",
        f"- certificate: `d_X={b['certificate']['d_X']}`, `d_Z={b['certificate']['d_Z']}`, "
        f"`CH={b['certificate']['CH']}`, `native_C=I={b['certificate']['native_C_eq_I']}`",
        f"- full-G indicator after shift: `{b['certificate']['full_G_indicator']}`",
        "",
        "## Shifted K rows",
        "",
        "```",
    ]
    lines.extend(" ".join(str(int(x)) for x in row) for row in b["K_shifted"])
    lines.extend([
        "```",
        "",
        "## Logical-fiber geometry",
        "",
        "| label | size | rank | projective lines | cap? | flat? |",
        "|---|---:|---:|---:|---|---|",
    ])
    for rec in b["fibers"]:
        lines.append(
            f"| `{rec['label']}` | {rec['size']} | {rec['rank']} | "
            f"{rec['projective_line_count']} | {rec['projective_cap']} | {rec['projective_flat']} |"
        )
    lines.extend(["", "## Counts inside FG48 quotient blocks", ""])
    for block in b["block_counts"]:
        lines.append(f"- `w={block['w']}`: `{block['counts']}`")
    lines.extend(["", "## Per-block ANF of shifted kappa rows", ""])
    for block in b["per_block_kappa_anf"]:
        lines.append(f"- `w={block['w']}`")
        for i, row in enumerate(block["kappa_rows"], start=1):
            lines.append(
                f"  - `kappa_{i}`: weight {row['weight']}, degree {row['degree']}, "
                f"spectrum `{row['spectrum']}`, bits `{row['bits']}`"
            )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=ROOT / "reports" / "q48_k_shift_geometry.json")
    parser.add_argument("--md", type=Path, default=ROOT / "reports" / "q48_k_shift_geometry.md")
    args = parser.parse_args()

    payload = build_payload()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    write_markdown(payload, args.md)

    best = payload["best_shift"]
    print(f"wrote {args.json}")
    print(f"wrote {args.md}")
    print("best B masks:", best["B_masks"])
    print("histogram:", best["histogram"])
    print("geometry:", best["geometry_summary"])
    print("certificate:", best["certificate"])


if __name__ == "__main__":
    main()
