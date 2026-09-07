#!/usr/bin/env python3
"""Exact missing weight-32 cell: quadratic normal forms, all origins, CH tests.

Run from the repository root with PYTHONPATH=. and print the JSON receipt.
The three forms are x6, x6+x1*x2, x6+x1*x2+x3*x4. The affine class
reduces analytically to the rank-five simplex; the other two are tested
without sharding or sampling, both unpunctured and punctured. Every choice
of origin is transported by an explicitly checked invertible linear map.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from time import monotonic

from gate2code.alpha_linear import decide_activity, label_space


def parity(v):
    return bin(v).count("1") & 1


def q(v, r):
    return sum((v >> (2*i) & 1) * (v >> (2*i+1) & 1)
               for i in range(r)) & 1


def row_basis(rows):
    basis = {}
    for row in rows:
        while row:
            pivot = row.bit_length()-1
            if pivot in basis:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
    return basis


def in_span(row, basis):
    while row:
        pivot = row.bit_length()-1
        if pivot not in basis:
            return False
        row ^= basis[pivot]
    return True


def packed_label_basis(support):
    return [sum(int(v) << j for j, v in enumerate(row))
            for row in label_space(support, 6)]


def linear_map(v, a, r):
    # (u,z) -> (u,z+B_q(a,u)); this map is its own inverse.
    return v ^ ((q((v & 31) ^ a, r) ^ q(v & 31, r) ^ q(a, r)) << 5)


def verify_transports(support, r):
    canonical = {False: support, True: sorted(v ^ 32 for v in support if v != 32)}
    vbases = {key: row_basis(packed_label_basis(val)) for key, val in canonical.items()}
    receipts = []
    for p in range(64):
        punctured = p in support
        actual = sorted(v ^ p for v in support if v != p)
        a = p & 31
        columns = [linear_map(1 << j, a, r) for j in range(6)]
        assert len(row_basis(columns)) == 6
        assert all(linear_map(linear_map(v, a, r), a, r) == v for v in range(64))
        assert sorted(linear_map(v, a, r) for v in actual) == canonical[punctured]
        where = {v: i for i, v in enumerate(canonical[punctured])}
        perm = [where[linear_map(v, a, r)] for v in actual]
        V = packed_label_basis(actual)
        for word in V:
            moved = sum(((word >> j) & 1) << perm[j] for j in range(len(actual)))
            assert in_span(moved, vbases[punctured])
        assert len(V) == len(vbases[punctured])
        receipts.append({"origin": p, "punctured": punctured,
                         "linear_map_columns": columns, "label_space_verified": True})
    return canonical, receipts


def main():
    start = monotonic()
    results = []
    for r in range(3):
        support = [v for v in range(64) if (v >> 5 & 1) ^ q(v, r)]
        assert len(support) == 32 and 0 not in support
        # The geometric support obeys S1--S3 independently of the label solver.
        for degree in (1, 2, 3):
            for indices in combinations(range(6), degree):
                mask = sum(1 << i for i in indices)
                assert sum(v & mask == mask for v in support) % 2 == 0
        canonical, transports = verify_transports(support, r)
        entry = {"quadratic_rank": 2*r, "support": support,
                 "origin_transports": transports, "tests": []}
        if r == 0:
            assert support == list(range(32, 64))
            assert canonical[True] == list(range(1, 32))
            assert sorted(v & 31 for v in support if v != 32) == list(range(1, 32))
            entry["result"] = "analytic reduction to rank-five simplex"
        else:
            for punctured in (False, True):
                active = canonical[punctured]
                before = monotonic()
                verdict = decide_activity(active, 6, prefilter=False)
                assert verdict["result"] == "unsat"
                entry["tests"].append({"punctured": punctured,
                    "support": active, "seconds": round(monotonic()-before, 6), **verdict})
            entry["result"] = "no third logical row"
        results.append(entry)
    root = Path(__file__).resolve().parents[1]
    print(json.dumps({"classification": "MacWilliams-Sloane Ch. 15 Theorems 4--5",
        "scope": {"s": 6, "weight": 32}, "sampling": False,
        "affine_classes": 3, "analytically_reduced_classes": 1,
        "unpunctured_tests": 2, "punctured_tests": 2,
        "origin_positions_verified": 192, "all_unsat_or_reduced": True,
        "algorithm_sha256": hashlib.sha256((root / 'gate2code/alpha_linear.py').read_bytes()).hexdigest(),
        "seconds": round(monotonic()-start, 6), "classes": results}, indent=2))


if __name__ == '__main__':
    main()
