"""Fibered QC diagnostics for FG48."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from gate2code.fg48.code import row_space_equal
from gate2code.fg48.css import incidence_matrices_fg48
from gate2code.fg48.geometry import (
    decompose_x_as_u_w,
    fg48_fiber_order,
    fg48_points,
    compose_u_w,
    restricted_lines_fg48,
)


def outer_c3_permutation() -> list[int]:
    """Return old-index -> new-index permutation cycling 01 -> 10 -> 11 -> 01."""
    order = fg48_fiber_order()
    index = {x: i for i, x in enumerate(order)}
    w_next = {1: 2, 2: 3, 3: 1}
    perm = []
    for x in order:
        u, w = decompose_x_as_u_w(x)
        perm.append(index[compose_u_w(u, w_next[w])])
    return perm


def permutation_cycles(p: list[int]) -> list[list[int]]:
    """Return cycle decomposition of an old-index -> new-index permutation."""
    seen = [False] * len(p)
    cycles: list[list[int]] = []
    for start in range(len(p)):
        if seen[start]:
            continue
        cur = start
        cyc = []
        while not seen[cur]:
            seen[cur] = True
            cyc.append(cur)
            cur = int(p[cur])
        cycles.append(cyc)
    return cycles


def permute_columns(M: np.ndarray, p: list[int]) -> np.ndarray:
    """Apply old-index -> new-index column permutation."""
    M = np.asarray(M, dtype=np.uint8) % 2
    out = np.zeros_like(M)
    for old, new in enumerate(p):
        out[:, new] = M[:, old]
    return out


def is_code_invariant_under_perm(G: np.ndarray, p: list[int]) -> bool:
    """Check whether rowspace(G) is invariant under a coordinate permutation."""
    return row_space_equal(G, permute_columns(G, p))


def split_blocks(H: np.ndarray, row_block_size: int, col_block_size: int) -> list[list[np.ndarray]]:
    """Split H into rectangular blocks."""
    H = np.asarray(H, dtype=np.uint8) % 2
    if H.shape[0] % row_block_size or H.shape[1] % col_block_size:
        raise ValueError("matrix shape is not divisible by requested block sizes")
    return [
        [H[i:i + row_block_size, j:j + col_block_size] for j in range(0, H.shape[1], col_block_size)]
        for i in range(0, H.shape[0], row_block_size)
    ]


def is_circulant_block(B: np.ndarray) -> bool:
    """Check whether a square binary block is ordinary row-circulant."""
    B = np.asarray(B, dtype=np.uint8) % 2
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        return False
    if B.shape[0] == 0:
        return True
    first = B[0]
    return all(np.array_equal(B[i], np.roll(first, i)) for i in range(B.shape[0]))


def _cayley_translation_invariant_rowset(rows: np.ndarray, cols: int = 16) -> bool:
    """Check whether row supports in one 16-point U block form C2^4 translates."""
    supports = {sum(1 << j for j, bit in enumerate(row[:cols]) if bit) for row in rows}
    for shift in range(cols):
        shifted = {sum(1 << (j ^ shift) for j in range(cols) if (mask >> j) & 1) for mask in supports}
        if shifted != supports:
            return False
    return True


def block_circulant_report(H: np.ndarray, block_size: int = 16) -> dict[str, Any]:
    """Report ordinary 16-circulant blocks when row count is compatible."""
    H = np.asarray(H, dtype=np.uint8) % 2
    out: dict[str, Any] = {
        "shape": [int(H.shape[0]), int(H.shape[1])],
        "column_block_size": int(block_size),
        "ordinary_circulant_applicable": bool(H.shape[0] % block_size == 0 and H.shape[1] % block_size == 0),
    }
    if not out["ordinary_circulant_applicable"]:
        out["ordinary_circulant_note"] = "row count is not a multiple of 16; ordinary block-circulant test skipped"
        return out
    blocks = split_blocks(H, block_size, block_size)
    flags = [[is_circulant_block(B) for B in row] for row in blocks]
    out["block_rows"] = len(blocks)
    out["block_cols"] = len(blocks[0]) if blocks else 0
    out["circulant_flags"] = flags
    out["all_blocks_circulant"] = bool(all(all(row) for row in flags))
    return out


def product_cayley_report() -> dict[str, Any]:
    """Report the natural C2^4 product structure of restricted line rows."""
    grouped = restricted_lines_fg48()
    h2_type_counts: Counter[tuple[int, int, int]] = Counter()
    h2_edge_counts_by_fiber: Counter[int] = Counter()
    for row in grouped[2]:
        occ = [0, 0, 0]
        fibers = []
        for x in row:
            _, w = decompose_x_as_u_w(x)
            occ[w - 1] += 1
            fibers.append(w)
        h2_type_counts[tuple(occ)] += 1
        if len(set(fibers)) == 1:
            h2_edge_counts_by_fiber[fibers[0]] += 1

    h3_one_per_fiber = True
    h3_xor_relation = True
    for row in grouped[3]:
        by_w = {}
        for x in row:
            u, w = decompose_x_as_u_w(x)
            by_w[w] = u
        if set(by_w) != {1, 2, 3}:
            h3_one_per_fiber = False
            h3_xor_relation = False
            continue
        if by_w[1] ^ by_w[2] ^ by_w[3]:
            h3_xor_relation = False

    return {
        "H2": {
            "description": "weight-2 rows are complete-graph edges inside one U fiber",
            "fiber_occupancy_counts": {str(k): int(v) for k, v in sorted(h2_type_counts.items())},
            "edge_counts_by_w_label": {str(int(k)): int(v) for k, v in sorted(h2_edge_counts_by_fiber.items())},
            "complete_graph_edges_per_fiber": 120,
            "matches_three_K16_fibers": bool(sorted(h2_edge_counts_by_fiber.values()) == [120, 120, 120]),
        },
        "H3": {
            "description": "weight-3 rows are triples (u1,01),(u2,10),(u1+u2,11)",
            "row_count": int(len(grouped[3])),
            "one_point_per_fiber": bool(h3_one_per_fiber),
            "u_xor_relation_all_rows": bool(h3_xor_relation),
            "expected_c2_4_cayley_rows": 256,
        },
    }


def qc_report() -> dict[str, Any]:
    """Return fibered QC invariance diagnostics."""
    from gate2code.fg48.code import generator_fg48

    p = outer_c3_permutation()
    cycles = permutation_cycles(p)
    cycle_hist = Counter(len(c) for c in cycles)
    mats = incidence_matrices_fg48()
    full = np.vstack([H for _, H in sorted(mats.items()) if H.shape[0]])
    invariance = {
        "G_X": is_code_invariant_under_perm(generator_fg48(), p),
        "H2": is_code_invariant_under_perm(mats[2], p),
        "H3": is_code_invariant_under_perm(mats[3], p),
        "full_incidence": is_code_invariant_under_perm(full, p),
    }
    return {
        "coordinate_order": {
            "description": "U x {01} | U x {10} | U x {11}",
            "points_match_fg48": fg48_fiber_order() == fg48_points(),
        },
        "outer_C3": {
            "permutation": p,
            "cycle_length_histogram": dict(sorted((int(k), int(v)) for k, v in cycle_hist.items())),
        },
        "invariance": invariance,
        "block_circulant": {
            "H2": block_circulant_report(mats[2]),
            "H3": block_circulant_report(mats[3]),
            "full_incidence": block_circulant_report(full),
        },
        "product_cayley": product_cayley_report(),
        "note": "ordinary circulant blocks are diagnostic only because U is C2^4, not cyclic C16",
    }
