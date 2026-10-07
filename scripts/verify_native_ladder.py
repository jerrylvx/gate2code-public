#!/usr/bin/env python3
"""Check the signed-weight conditions and 8-divisibility on the stored Q48 code.

The check compares the quasi-transversal criterion, physical single-qubit
Clifford corrections, the single-qubit transversal T/T-dagger criterion, and
the 8-divisibility of C_2. The binary flip vector gamma is the paper's notation;
physical signs satisfy Gamma_j = (-1)^gamma_j.

Run from the repository root: python scripts/verify_native_ladder.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate2code.identity import no_correction, verify_no_correction  # noqa: E402

CODE_PATH = ROOT / "code_48_3_3_fg48.npz"
K_LOGICAL = 3

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def load() -> tuple[np.ndarray, np.ndarray]:
    """Return the 9x48 generator G = [K;S] and the binary flip vector gamma."""
    G = np.load(CODE_PATH)["G"] % 2
    res = no_correction(G)
    if not res.found:
        raise SystemExit(f"no single-qubit transversal solution found: {res.status}")
    # gate2code stores Gamma as omega exponents 1 (T) and 7 (T-dagger);
    # delta is the same object as the paper's flip vector gamma in {0,1}^n.
    return G, res.delta.astype(int) % 2


def all_codewords(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every codeword of C_1 = rowspan(G), with the selector bits that made it."""
    m = G.shape[0]
    Z = np.array([[(z >> i) & 1 for i in range(m)] for z in range(1 << m)], dtype=int)
    return (Z @ G) % 2, Z


def main() -> int:
    G, gamma = load()
    m, n = G.shape
    S = G[K_LOGICAL:]
    C, Z = all_codewords(G)
    x = Z[:, :K_LOGICAL]
    weight = C.sum(1)
    target = (4 * x[:, 0] * x[:, 1] * x[:, 2]) % 8
    overlap = (C * gamma).sum(1)                       # integer |c and gamma|
    sw = weight - 2 * overlap                          # eq:exp-sw
    stab = Z[:, :K_LOGICAL].sum(1) == 0                # the x = 0 slice, c in C_2

    check("stored Q48 admits a single-qubit transversal T/T-dagger pattern",
          verify_no_correction(G, np.where(gamma == 1, 7, 1)),
          f"{int((gamma == 0).sum())} T + {int(gamma.sum())} T-dagger")

    # --- Section 2: 8-divisibility and the divisibility ladder -------------
    stab_weights = weight[stab]
    check("C_2 is 8-divisible", bool((stab_weights % 8 == 0).all()),
          f"{len(stab_weights)} stabilizer codewords")
    ladder = (
        all(int(S[l].sum()) % 8 == 0 for l in range(len(S)))
        and all(int((S[a] & S[b]).sum()) % 4 == 0 for a, b in combinations(range(len(S)), 2))
        and all(int((S[a] & S[b] & S[c]).sum()) % 2 == 0
                for a, b, c in combinations(range(len(S)), 3))
    )
    check("eq:exp-divladder holds for S", ladder, "|S_l| mod 8, pairs mod 4, triples mod 2")
    s123 = (
        all(int(S[l].sum()) % 2 == 0 for l in range(len(S)))
        and all(int((S[a] & S[b]).sum()) % 2 == 0 for a, b in combinations(range(len(S)), 2))
        and all(int((S[a] & S[b] & S[c]).sum()) % 2 == 0
                for a, b, c in combinations(range(len(S)), 3))
    )
    check("the ladder reduces mod 2 to S1-S3", ladder and s123, "S1-S3 are the reductions mod 2")

    # --- def:exp-CI and its two rewritings ---------------------------------
    check("eq:exp-CI: sw_Gamma(c) == 4 x1x2x3 (mod 8)", bool((sw % 8 == target).all()),
          f"{len(sw)} codewords")
    check("eq:exp-Z4: 2(gamma.c) == |c| - 4 x1x2x3 (mod 8)",
          bool(((2 * overlap) % 8 == (weight - target) % 8).all()), "")
    check("eq:exp-Z4half: gamma.c == |c|/2 - 2 x1x2x3 (mod 4)",
          bool((weight % 2 == 0).all()
               and (overlap % 4 == (weight // 2 - 2 * x[:, 0] * x[:, 1] * x[:, 2]) % 4).all()),
          "halving is legitimate because every codeword weight is even")

    # --- the necessity budget with the CZ terms deleted --------------------
    lam = (-gamma) % 4                                  # lam_j in {0,3}
    budget = (weight + 2 * (C * lam).sum(1)) % 8
    check("necessity budget with lam_jj' = 0 reproduces eq:exp-CI",
          bool((budget == target).all()),
          "|c| + 2 sum_j lam_j c_j == 4 x1x2x3, lam = -gamma mod 4")

    # --- the x = 0 slice ----------------------------------------------------
    check("x=0 slice: sw_Gamma(c) == 0 (mod 8) on C_2", bool((sw[stab] % 8 == 0).all()), "")
    check("x=0 slice at Gamma = 1 is exactly 8-divisibility",
          bool((weight[stab] % 8 == 0).all()), "sw_1(c) = |c|")
    check("8-divisible C_2 forces gamma.c == 0 (mod 4) on C_2",
          bool((overlap[stab] % 4 == 0).all()), "")
    check("hence gamma lies in C_2-perp", bool(((S @ gamma) % 2 == 0).all()),
          "mod-2 reduction of the previous line")

    # --- the coset (translation) form ---------------------------------------
    translated = ((C + gamma) % 2).sum(1)
    check("sw_Gamma(c) = |c xor gamma| - |gamma|",
          bool((translated - int(gamma.sum()) == sw).all()), "")
    per_coset = all(
        bool(((translated[(x == xv).all(1)] - int(gamma.sum())) % 8
              == (4 * xv[0] * xv[1] * xv[2]) % 8).all())
        for xv in np.array([[(v >> i) & 1 for i in range(K_LOGICAL)]
                            for v in range(1 << K_LOGICAL)]))
    check("each coset gamma + K^T x + C_2 has constant weight mod 8", per_coset,
          "the constant is |gamma| + 4 x1x2x3")

    width = max(len(nm) for nm, _, _ in CHECKS)
    for nm, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {nm.ljust(width)}  {detail}")
    failed = sum(1 for _, ok, _ in CHECKS if not ok)
    report = {
        "source": str(CODE_PATH.relative_to(ROOT)),
        "codewords_checked": int(len(C)),
        "checks": [
            {"name": name, "passed": bool(ok), "detail": detail}
            for name, ok, detail in CHECKS
        ],
        "all_passed": failed == 0,
    }
    report_path = ROOT / "reports/q48_native_ladder.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(f"wrote {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
