"""K/S column-fiber diagnostics for split CSS G matrices."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from gate2code.ccz import f2_nullspace, f2_rank


def _as_binary(M: np.ndarray) -> np.ndarray:
    return np.asarray(M, dtype=np.uint8) % 2


def column_ints(M: np.ndarray) -> list[int]:
    M = _as_binary(M)
    out = []
    for col in M.T:
        x = 0
        for bit in col:
            x = (x << 1) | int(bit)
        out.append(x)
    return out


def projective_summary(M: np.ndarray) -> dict[str, Any]:
    M = _as_binary(M)
    cols = column_ints(M)
    counts = Counter(cols)
    zero = counts.get(0, 0)
    return {
        "rows": int(M.shape[0]),
        "n_columns": int(M.shape[1]),
        "rank": int(f2_rank(M.astype(np.int64))),
        "zero_columns": int(zero),
        "unique_columns": int(len(counts)),
        "unique_nonzero_columns": int(len([c for c in counts if c != 0])),
        "max_column_multiplicity": int(max(counts.values(), default=0)),
        "multiplicity_histogram": dict(sorted((int(k), int(v)) for k, v in Counter(counts.values()).items())),
        "projective": bool(zero == 0 and len(counts) == M.shape[1]),
    }


def span_weight_distribution_small(G: np.ndarray, *, max_dim: int = 16) -> dict[int, int] | None:
    G = _as_binary(G)
    r, n = G.shape
    if r > max_dim:
        return None
    counts: dict[int, int] = {0: 1}
    cur = np.zeros(n, dtype=np.uint8)
    prev_gray = 0
    for idx in range(1, 1 << r):
        gray = idx ^ (idx >> 1)
        diff = gray ^ prev_gray
        bit = diff.bit_length() - 1
        cur ^= G[bit]
        counts[int(cur.sum())] = counts.get(int(cur.sum()), 0) + 1
        prev_gray = gray
    return dict(sorted(counts.items()))


def min_distance_small(G: np.ndarray, *, max_dim: int = 16) -> int | None:
    G = _as_binary(G)
    r, n = G.shape
    if r == 0:
        return None
    if r > max_dim:
        return None
    best = n + 1
    cur = np.zeros(n, dtype=np.uint8)
    prev_gray = 0
    for idx in range(1, 1 << r):
        gray = idx ^ (idx >> 1)
        diff = gray ^ prev_gray
        bit = diff.bit_length() - 1
        cur ^= G[bit]
        wt = int(cur.sum())
        if wt and wt < best:
            best = wt
        prev_gray = gray
    return None if best == n + 1 else int(best)


def analyze_ks_fibers(
    K: np.ndarray | None,
    S: np.ndarray,
    *,
    inner_max_dim: int = 16,
    max_fibers: int = 32,
) -> dict[str, Any]:
    """Summarize full-G, K, S, and K-fiber/S-label structure."""
    S = _as_binary(S)
    if K is None:
        K = np.zeros((0, S.shape[1]), dtype=np.uint8)
    K = _as_binary(K)
    if K.shape[1] != S.shape[1]:
        raise ValueError("K and S must have the same column count")
    G = np.vstack([K, S]).astype(np.uint8) % 2
    n = int(S.shape[1])

    k_cols = column_ints(K)
    s_cols = column_ints(S)
    g_cols = column_ints(G)
    k_to_s: dict[int, list[int]] = defaultdict(list)
    s_to_k: dict[int, list[int]] = defaultdict(list)
    for k, s in zip(k_cols, s_cols):
        k_to_s[k].append(s)
        s_to_k[s].append(k)

    k_fiber_sizes = Counter(len(v) for v in k_to_s.values())
    s_fiber_sizes = Counter(len(v) for v in s_to_k.values())
    k_inner_patterns = Counter()
    fiber_records = []
    for k_label, s_labels in sorted(k_to_s.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        s_mult = Counter(s_labels)
        pattern = tuple(sorted(s_mult.values(), reverse=True))
        k_inner_patterns[pattern] += 1
        cols = [j for j, lab in enumerate(k_cols) if lab == k_label]
        S_fiber = S[:, cols]
        inner = f2_nullspace(S_fiber.astype(np.int64)).astype(np.uint8)
        rec = {
            "k_label": int(k_label),
            "k_label_bits": format(k_label, f"0{K.shape[0]}b") if K.shape[0] else "",
            "size": int(len(cols)),
            "unique_s_labels": int(len(s_mult)),
            "s_multiplicity_histogram": dict(sorted((int(a), int(b)) for a, b in Counter(s_mult.values()).items())),
            "s_labels_are_distinct": bool(len(s_mult) == len(cols)),
            "s_fiber_rank": int(f2_rank(S_fiber.astype(np.int64))),
            "inner_dimension": int(inner.shape[0]),
            "inner_min_distance": min_distance_small(inner, max_dim=inner_max_dim),
            "inner_weight_enumerator": span_weight_distribution_small(inner, max_dim=inner_max_dim),
        }
        fiber_records.append(rec)

    return {
        "n": n,
        "k_rows": int(K.shape[0]),
        "s_rows": int(S.shape[0]),
        "projective": {
            "G": projective_summary(G),
            "K": projective_summary(K) if K.shape[0] else None,
            "S": projective_summary(S),
        },
        "full_g_column_multiplicity_histogram": dict(sorted((int(k), int(v)) for k, v in Counter(Counter(g_cols).values()).items())),
        "k_fiber_count": int(len(k_to_s)),
        "k_fiber_size_histogram": dict(sorted((int(k), int(v)) for k, v in k_fiber_sizes.items())),
        "s_fiber_count": int(len(s_to_k)),
        "s_fiber_size_histogram": dict(sorted((int(k), int(v)) for k, v in s_fiber_sizes.items())),
        "k_fiber_s_multiplicity_pattern_histogram": {
            str(tuple(int(x) for x in k)): int(v) for k, v in k_inner_patterns.items()
        },
        "all_s_labels_distinct_inside_k_fibers": bool(all(len(set(v)) == len(v) for v in k_to_s.values())),
        "largest_k_fibers": fiber_records[:max_fibers],
    }
