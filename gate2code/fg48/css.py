"""Restricted-incidence diagnostics for FG48."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from gate2code.ccz import f2_nullspace_dim, f2_rank
from gate2code.fg48.geometry import fg48_points, restricted_lines_fg48


def incidence_matrix_from_rows(rows: list[tuple[int, ...]], coord_order: list[int] | None = None) -> np.ndarray:
    """Build a binary point-incidence matrix from restricted rows."""
    if coord_order is None:
        coord_order = fg48_points()
    index = {x: i for i, x in enumerate(coord_order)}
    M = np.zeros((len(rows), len(coord_order)), dtype=np.uint8)
    for r, row in enumerate(rows):
        for x in row:
            M[r, index[x]] = 1
    return M


def incidence_matrices_fg48() -> dict[int, np.ndarray]:
    """Return incidence matrices grouped by restricted line weight."""
    grouped = restricted_lines_fg48()
    return {weight: incidence_matrix_from_rows(rows) for weight, rows in grouped.items()}


def commutation_matrix(Hx: np.ndarray, Hz: np.ndarray) -> np.ndarray:
    """Return Hx Hz^T mod 2."""
    return (np.asarray(Hx, dtype=np.uint8) @ np.asarray(Hz, dtype=np.uint8).T) % 2


def commute(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    """Check CSS commutation."""
    return bool(np.all(commutation_matrix(Hx, Hz) == 0))


def row_overlap_spectrum(H: np.ndarray) -> dict[int, int]:
    """Count pairwise row-overlap sizes."""
    H = np.asarray(H, dtype=np.uint8) % 2
    counts: Counter[int] = Counter()
    for i, j in combinations(range(H.shape[0]), 2):
        counts[int(np.sum(H[i] & H[j]))] += 1
    return dict(sorted((int(k), int(v)) for k, v in counts.items()))


def incidence_matrix_report(H: np.ndarray) -> dict[str, Any]:
    """Report rank/kernel/check-weight diagnostics for one incidence matrix."""
    H = np.asarray(H, dtype=np.uint8) % 2
    row_weights = Counter(int(w) for w in H.sum(axis=1))
    return {
        "shape": [int(H.shape[0]), int(H.shape[1])],
        "rank": int(f2_rank(H.astype(np.int64))),
        "right_kernel_dim": int(f2_nullspace_dim(H.astype(np.int64))),
        "row_weight_distribution": dict(sorted((int(k), int(v)) for k, v in row_weights.items())),
        "row_overlap_spectrum": row_overlap_spectrum(H),
    }


def css_diagnostic_report() -> dict[str, Any]:
    """Return incidence and commutation diagnostics without subset search."""
    mats = incidence_matrices_fg48()
    H2 = mats.get(2, np.zeros((0, 48), dtype=np.uint8))
    H3 = mats.get(3, np.zeros((0, 48), dtype=np.uint8))
    full = np.vstack([H for _, H in sorted(mats.items()) if H.shape[0]])
    return {
        "matrices": {str(k): incidence_matrix_report(v) for k, v in sorted(mats.items())},
        "full": incidence_matrix_report(full),
        "commutation": {
            "H2_H2": commute(H2, H2),
            "H3_H3": commute(H3, H3),
            "H2_H3": commute(H2, H3),
        },
        "note": "diagnostics only; no row-subset search or distance claim",
    }
