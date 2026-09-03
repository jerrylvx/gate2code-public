"""Assert the exact polynomials in the paper reproduce the flip vector.

Indexing convention used in the paper text: u = (u_1,u_2,u_3,u_4) lists the four rows
of A in order, so u_i is bit (i-1) of the within-coset coordinate.
"""
import json
from pathlib import Path

import numpy as np
from gate2code.identity import no_correction

ROOT = Path(__file__).resolve().parents[1]
npz = np.load(ROOT / "code_48_3_3_fg48.npz")
G = npz["G"].astype(np.uint8) % 2
delta = np.asarray(no_correction(G, k=3).delta, dtype=np.uint8)
S = G[3:]


def bits(j):
    u = [int(S[i, j]) for i in range(4)]          # u_1..u_4
    w = [int(S[4, j]), int(S[5, j])]              # w_1, w_2
    return u, w


def Q1(u):
    u1, u2, u3, u4 = u
    return (u1 + u2 + u3 + u1 * u3 + u2 * u3 + u2 * u4) % 2


def Q2(u):
    u1, u2, u3, u4 = u
    return (u3 + u4 + u1 * u2 + u1 * u4 + u3 * u4) % 2


def g1(u):
    u1, u2, u3, u4 = u
    return (u1 + u2 + u1 * u3 + u2 * u3 + u2 * u4) % 2


def g2(u):
    u1, u2, u3, u4 = u
    return (u4 + u1 * u2 + u1 * u4 + u3 * u4) % 2


def g3(u):
    u1, u2, u3, u4 = u
    return (u1 + u2 + u3 + u4 + u1 * u2 + u1 * u3 + u2 * u3
            + u1 * u4 + u2 * u4 + u3 * u4) % 2


bad = 0
counts = {(1, 0): 0, (0, 1): 0, (1, 1): 0}
for j in range(48):
    u, w = bits(j)
    gamma = (u[2] + w[0] * Q1(u) + w[1] * Q2(u)) % 2
    assert gamma == delta[j], f"closed form fails at column {j}"
    per_coset = {(1, 0): g1, (0, 1): g2, (1, 1): g3}[(w[0], w[1])](u)
    assert per_coset == delta[j], f"coset restriction fails at column {j}"
    counts[(w[0], w[1])] += int(delta[j])
    bad += int(gamma != delta[j])

print("closed form gamma(u,w) = u_3 + w_1 Q_1 + w_2 Q_2 reproduces delta on all 48 columns")
print("per-coset dagger counts:", {f"w={k}": v for k, v in counts.items()})
print("total daggers:", sum(counts.values()), " T sites:", 48 - sum(counts.values()))
assert sum(counts.values()) == 22
assert sorted(counts.values()) == [6, 6, 10]
report = {
    "source": "code_48_3_3_fg48.npz",
    "columns_checked": 48,
    "closed_form_matches_flip_vector": bad == 0,
    "block_Tdag_counts": {f"{key[0]}{key[1]}": value for key, value in counts.items()},
    "n_Tdag": int(sum(counts.values())),
    "n_T": int(48 - sum(counts.values())),
}
report_path = ROOT / "reports/q48_tpattern_paper_form.json"
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("all paper-form assertions passed")
print(f"wrote {report_path}")
