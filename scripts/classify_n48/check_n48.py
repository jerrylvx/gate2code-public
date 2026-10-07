#!/usr/bin/env python3
"""Reproduce the complete classification at n = 48 documented in the repository README.

Runs the exact classification of CH label maps on PG(5,2) minus PG(3,2) (classify.py), the
exact single-qubit transversal test of every class (analyze_classes.py), and the grouping of classes into codes C1
(code_groups.py). Asserts 4,152 classes, 17 single-qubit transversal classes, and exactly two codes C1 with a single-qubit transversal
pattern, and checks that their weight enumerators are those of the two archived n = 48 codes of
Table 3. Needs numpy, numba and sympy. Run time about seven minutes.
Usage: python scripts/classify_n48/check_n48.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_classes  # noqa: E402
import code_groups  # noqa: E402
from verify_family_table import enumerator, to_int  # noqa: E402


def main() -> int:
    analyze_classes.main(4, 2)
    code_groups.main(4, 2)
    ana = pickle.load(open(HERE / "analysis_p4q2.pkl", "rb"))["classes"]
    groups = pickle.load(open(HERE / "codes_p4q2.pkl", "rb"))["groups"]
    native = [c for c in ana if c["native"]]
    native_codes = [g for g in groups if any(ana[ci]["native"] for ci in g)]
    assert len(ana) == 4152, len(ana)
    assert len(native) == 17, len(native)
    assert len(native_codes) == 2, len(native_codes)
    assert all(ana[ci]["native"] for g in native_codes for ci in g)
    found = sorted(ana[g[0]]["WE"] for g in native_codes)
    archived = sorted(enumerator([to_int(r) for r in np.load(ROOT / "data/family_witnesses/grid" / f)["G"] % 2])
                      for f in ("p4q2_01.npz", "p4q2_02.npz"))
    assert found == archived, (found, archived)
    print(f"n=48: {len(ana)} classes, {len(native)} with a single-qubit transversal pattern, "
          f"{len(native_codes)} such codes C1, weight enumerators match the archived Table 3 codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
