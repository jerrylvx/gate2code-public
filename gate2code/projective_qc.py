"""Projective flat-complement and fibered-QC diagnostics."""
from __future__ import annotations

from itertools import product
from math import log2
from typing import Any

import numpy as np

from gate2code.ccz import f2_rank
from gate2code.fg48.code import row_space_equal
from gate2code.fg48.qc import permutation_cycles
from gate2code.ks_fiber import column_ints, projective_summary


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _bits_msb(x: int, rows: int) -> np.ndarray:
    return np.array([(x >> (rows - 1 - i)) & 1 for i in range(rows)], dtype=np.uint8)


def _matrix_from_labels(labels: list[int], rows: int) -> np.ndarray:
    M = np.zeros((rows, len(labels)), dtype=np.uint8)
    for j, x in enumerate(labels):
        M[:, j] = _bits_msb(x, rows)
    return M


def _extend_basis(seed: list[int], candidates: list[int], target_rank: int, rows: int) -> list[int]:
    basis = list(seed)
    for x in candidates:
        if x in basis:
            continue
        if f2_rank(_matrix_from_labels(basis + [x], rows).astype(int)) > len(basis):
            basis.append(x)
            if len(basis) == target_rank:
                return basis
    raise ValueError("could not extend basis")


def _label_coeff_maps(basis: list[int], rows: int) -> tuple[dict[int, int], dict[int, int]]:
    label_to_coeff: dict[int, int] = {}
    coeff_to_label: dict[int, int] = {}
    for coeff in range(1 << rows):
        v = np.zeros(rows, dtype=np.uint8)
        for i, b in enumerate(basis):
            if (coeff >> i) & 1:
                v ^= _bits_msb(b, rows)
        label = 0
        for bit in v:
            label = (label << 1) | int(bit)
        label_to_coeff[label] = coeff
        coeff_to_label[coeff] = label
    return label_to_coeff, coeff_to_label


def _apply_matrix_columns(cols: tuple[int, ...], x: int) -> int:
    out = 0
    for i, col in enumerate(cols):
        if (x >> i) & 1:
            out ^= col
    return out


def _glq_singer_columns(q: int) -> tuple[int, ...] | None:
    """Return one GL(q,2) element cycling all nonzero vectors, if q>=2."""
    if q < 2:
        return None
    target = (1 << q) - 1
    for cols in product(range(1, 1 << q), repeat=q):
        if len(set(cols)) < q:
            continue
        mat = np.array([[(c >> i) & 1 for c in cols] for i in range(q)], dtype=np.uint8)
        if f2_rank(mat.astype(int)) != q:
            continue
        orbit = []
        x = 1
        for _ in range(target):
            orbit.append(x)
            x = _apply_matrix_columns(cols, x)
        if x == 1 and len(set(orbit)) == target:
            return tuple(int(c) for c in cols)
    raise ValueError(f"no Singer element found for q={q}")


def permute_columns(M: np.ndarray, p: list[int]) -> np.ndarray:
    """Apply an old-index -> new-index column permutation."""
    M = np.asarray(M, dtype=np.uint8) % 2
    out = np.zeros_like(M)
    for old, new in enumerate(p):
        out[:, new] = M[:, old]
    return out


def flat_complement_structure(M: np.ndarray) -> dict[str, Any]:
    """Identify whether projective columns are PG(r-1,2) minus a subspace."""
    M = np.asarray(M, dtype=np.uint8) % 2
    rows, n = M.shape
    proj = projective_summary(M)
    out: dict[str, Any] = {
        "rows": int(rows),
        "n": int(n),
        "projective": proj["projective"],
        "projective_summary": proj,
        "is_flat_complement": False,
    }
    if not proj["projective"]:
        out["reason"] = "columns are not projective"
        return out
    if proj["rank"] != rows:
        out["reason"] = "rows are not full rank"
        return out
    cols = set(column_ints(M))
    all_nonzero = set(range(1, 1 << rows))
    missing = sorted(all_nonzero - cols)
    deleted_size_with_zero = len(missing) + 1
    out["missing_nonzero_count"] = int(len(missing))
    if not _is_power_of_two(deleted_size_with_zero):
        out["reason"] = "missing points plus zero do not have power-of-two size"
        return out
    U = set(missing) | {0}
    if not all((a ^ b) in U for a in U for b in U):
        out["reason"] = "missing points plus zero are not closed under xor"
        return out
    a = int(log2(deleted_size_with_zero))
    q = rows - a
    U_basis = _extend_basis([], missing, a, rows)
    W_basis = _extend_basis(U_basis, sorted(cols), rows, rows)[a:]
    label_to_coeff, coeff_to_label = _label_coeff_maps(U_basis + W_basis, rows)
    fiber_counts: dict[int, int] = {}
    for x in cols:
        coeff = label_to_coeff[x]
        w = coeff >> a
        fiber_counts[w] = fiber_counts.get(w, 0) + 1
    out.update({
        "is_flat_complement": True,
        "deleted_subspace_dim": int(a),
        "quotient_dim": int(q),
        "expected_n": int((1 << rows) - (1 << a)),
        "U_basis_column_labels": [int(x) for x in U_basis],
        "W_basis_column_labels": [int(x) for x in W_basis],
        "fiber_count": int(len(fiber_counts)),
        "fiber_size_histogram": {
            str(int(size)): int(list(fiber_counts.values()).count(size))
            for size in sorted(set(fiber_counts.values()))
        },
        "fiber_counts_by_nonzero_quotient_label": {str(int(k)): int(v) for k, v in sorted(fiber_counts.items())},
        "coordinate_algebra": (
            f"F2[C2^{a}] tensor F2[C{(1 << q) - 1}] on coordinates "
            f"U x (F2^{q}\\\\{{0}}) after choosing a Singer cycle on the quotient"
        ),
    })
    return out


def flat_complement_qc_report(M: np.ndarray) -> dict[str, Any]:
    """Build and test the natural outer Singer/fiber-cycle QC action."""
    M = np.asarray(M, dtype=np.uint8) % 2
    rows, _ = M.shape
    base = flat_complement_structure(M)
    out = dict(base)
    if not base.get("is_flat_complement"):
        out["qc_tested"] = False
        return out
    a = int(base["deleted_subspace_dim"])
    q = int(base["quotient_dim"])
    singer = _glq_singer_columns(q)
    if singer is None:
        out.update({
            "qc_tested": False,
            "qc_reason": "quotient has only one nonzero label, so the outer cycle is trivial",
        })
        return out

    basis = list(base["U_basis_column_labels"]) + list(base["W_basis_column_labels"])
    label_to_coeff, coeff_to_label = _label_coeff_maps([int(x) for x in basis], rows)
    cols = column_ints(M)
    col_to_index = {x: i for i, x in enumerate(cols)}
    p: list[int] = []
    for x in cols:
        coeff = label_to_coeff[x]
        u = coeff & ((1 << a) - 1)
        w = coeff >> a
        w2 = _apply_matrix_columns(singer, w)
        y = coeff_to_label[u | (w2 << a)]
        p.append(col_to_index[y])

    hist: dict[int, int] = {}
    for cyc in permutation_cycles(p):
        hist[len(cyc)] = hist.get(len(cyc), 0) + 1

    out.update({
        "qc_tested": True,
        "outer_cycle_order": int((1 << q) - 1),
        "outer_singer_columns": [int(x) for x in singer],
        "cycle_length_histogram": {str(int(k)): int(v) for k, v in sorted(hist.items())},
        "rowspace_invariant": bool(row_space_equal(M, permute_columns(M, p))),
        "permutation_old_to_new": [int(x) for x in p],
    })
    return out
