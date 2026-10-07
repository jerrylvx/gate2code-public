#!/usr/bin/env python3
"""Recompute every entry of Table tab:exp-family from the archived witnesses.

Independent of the gate2code package: integer bit arithmetic only. For each witness in
data/family_witnesses/grid/ (listed in index.json) it checks, from the generator matrix
G = [K; S] and the flip vector gamma (gamma_j = 1 means T-dagger on column j):
  * the columns of S are the subspace complement of index.json's (p, q);
  * S has actual rank s and distinct nonzero columns, and rank G - rank S = 3;
  * the nine CH conditions: every overlap of one, two, or three distinct rows is even,
    except |K1 & K2 & K3|, which is odd;
  * the native condition (eq:exp-CI): |v| - 2|gamma & v| = 4 x1 x2 x3 (mod 8) for every
    codeword v = K^T x + S^T y;
  * d_X = min |K^T x + S^T y| over x != 0, and d_Z (eq:exp-dZ), by a breadth-first search over
    syndrome-label pairs.
Codes are counted per cell as inequivalent when (d_X, d_Z) or the weight enumerator of C1
differ. With --check, the per-cell parameters and counts must equal the printed table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "data" / "family_witnesses" / "grid"

# Table tab:exp-family: (p, q) -> (n, d, number of inequivalent codes).
PRINTED = {
    (4, 2): (48, 3, 2), (4, 3): (112, 3, 12), (4, 4): (240, 3, 17), (4, 5): (496, 3, 18),
    (5, 2): (96, 3, 11), (5, 3): (224, 3, 8), (5, 4): (480, 3, 14),
    (6, 1): (64, 4, 5), (6, 2): (192, 3, 14), (6, 3): (448, 3, 13),
    (7, 1): (128, 4, 3), (7, 2): (384, 3, 18),
    (8, 1): (256, 4, 3),
}


def to_int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def rank(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for r in rows:
        while r:
            top = r.bit_length() - 1
            if top in basis:
                r ^= basis[top]
            else:
                basis[top] = r
                break
    return len(basis)


def popcount(x: int) -> int:
    return bin(x).count("1")


def analyze(K: list[int], S: list[int], n: int, gamma: int) -> tuple[int, int, int, int]:
    s = len(S)
    assert rank(S) == s, "stabilizer rows are dependent"
    assert rank(K + S) - rank(S) == 3, "k != 3"
    columns = [sum(((S[l] >> j) & 1) << l for l in range(s)) for j in range(n)]
    assert 0 not in columns and len(set(columns)) == n, "support is not projective"
    rows = K + S
    for t in (1, 2, 3):
        for subset in combinations(range(len(rows)), t):
            overlap = (1 << n) - 1
            for r in subset:
                overlap &= rows[r]
            parity = popcount(overlap) & 1
            expected = 1 if subset == (0, 1, 2) else 0
            if subset[0] >= 3:
                expected = 0
            assert parity == expected, f"CH condition fails on rows {subset}"
    dx = n + 1
    for x in product((0, 1), repeat=3):
        base = 0
        for i in range(3):
            if x[i]:
                base ^= K[i]
        for y in range(1 << s):
            v = base
            for l in range(s):
                if (y >> l) & 1:
                    v ^= S[l]
            sw = popcount(v) - 2 * popcount(v & gamma)
            assert (sw - 4 * x[0] * x[1] * x[2]) % 8 == 0, f"native condition fails at x={x}"
            if any(x):
                dx = min(dx, popcount(v))
    # d_Z: breadth-first search over (syndrome, label) states reached by adding columns.
    labels = [sum(((K[i] >> j) & 1) << i for i in range(3)) for j in range(n)]
    start = (0, 0)
    seen = {start}
    frontier = [start]
    dz = None
    depth = 0
    while frontier and dz is None:
        depth += 1
        nxt = []
        for syn, lab in frontier:
            for j in range(n):
                state = (syn ^ columns[j], lab ^ labels[j])
                if state in seen:
                    continue
                if state[0] == 0 and state[1] != 0:
                    dz = depth
                    break
                seen.add(state)
                nxt.append(state)
            if dz is not None:
                break
        frontier = nxt
    assert dz is not None
    flips = popcount(gamma)
    return n - flips, flips, dx, dz


def enumerator(rows: list[int]) -> tuple:
    counts: Counter = Counter()
    for m in range(1 << len(rows)):
        w = 0
        for i, r in enumerate(rows):
            if (m >> i) & 1:
                w ^= r
        counts[popcount(w)] += 1
    return tuple(sorted(counts.items()))


def is_complement(S: list[int], n: int, p: int, q: int) -> bool:
    s = len(S)
    cols = {sum(((S[l] >> j) & 1) << l for l in range(s)) for j in range(n)}
    removed = set(range(1 << s)) - cols
    return (s == p + q and len(removed) == 1 << p
            and all((a ^ b) in removed for a in removed for b in removed))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="compare the per-cell parameters and counts with the printed table")
    args = parser.parse_args()
    index = json.loads((GRID / "index.json").read_text())
    cells: dict = defaultdict(set)
    dz_by_cell: dict = defaultdict(set)
    for entry in index:
        data = np.load(GRID / entry["file"])
        G = np.asarray(data["G"], dtype=np.int64) % 2
        rows = [to_int(r) for r in G]
        n, p, q = int(data["n"]), int(data["p"]), int(data["q"])
        K, S = rows[:3], rows[3:]
        assert is_complement(S, n, p, q), entry["file"]
        _, _, dx, dz = analyze(K, S, n, to_int(np.asarray(data["gamma01"]) % 2))
        enum = enumerator(rows)
        assert (dx, dz) == (entry["d_X"], entry["d_Z"]), entry["file"]
        assert hashlib.sha1(str(enum).encode()).hexdigest() == entry["enumerator_sha1"], entry["file"]
        cells[(p, q)].add((dx, dz, enum))
        dz_by_cell[(p, q)].add(dz)
    computed = {}
    for (p, q), codes in sorted(cells.items()):
        d = min(dz_by_cell[(p, q)])
        computed[(p, q)] = (2 ** (p + q) - 2 ** p, d, len(codes))
        print(f"(p,q)=({p},{q}): [[{computed[(p, q)][0]},3,{d}]], {len(codes)} codes")
    print(f"{len(index)} witnesses verified")
    if args.check:
        if computed != PRINTED:
            for key in sorted(set(computed) | set(PRINTED)):
                if computed.get(key) != PRINTED.get(key):
                    print(f"MISMATCH {key}: printed {PRINTED.get(key)}, computed {computed.get(key)}",
                          file=sys.stderr)
            return 1
        print("all printed values match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
