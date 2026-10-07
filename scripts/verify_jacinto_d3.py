#!/usr/bin/env python3
"""Verify the 47-qubit code of Jacinto et al., arXiv:2606.07734v1, Appendix D, Eq. (D3).

The stored matrix has logical rows 1-3 and stabilizer rows 4-9. The check verifies
the nine CH conditions, ranks, d_X=16 and d_Z=3, then exhibits a binary linear
contradiction excluding a single-qubit transversal T/T-dagger implementation.
Writes reports/jacinto_d3_47_verification.json.
"""
import json
import numpy as np

ROWS = [
    "01111101011011011111010011101001100001001011101",
    "10010011110000101100111101110100111010010110000",
    "00010011100101100000010001110111101111101010110",
    "10011101001010011101001010011101001010011101001",
    "01001011011001001011011001001011011001001011011",
    "00111000111000111000111000111000111000111000111",
    "00000111111000000111111000000111111000000111111",
    "00000000000111111111111000000000000111111111111",
    "00000000000000000000000111111111111111111111111",
]
G = np.array([[int(c) for c in r] for r in ROWS], dtype=int)
assert G.shape == (9, 47)

from gate2code.ccz import compute_distances, f2_rank, verify_ch_conditions
from gate2code.identity import no_correction
from gate2code.fg48.code import code_weight_distribution

S = G[3:]
ch = verify_ch_conditions(G)
dX, dZ = compute_distances(G)
res = no_correction(G)

# A short contradiction extracted from the modulo-two reductions of the
# one-row and two-row native equations. Indices here are zero-based.
single_rows = [2, 3, 7]
pair_rows = [
    (0, 6), (0, 7), (0, 8), (1, 3), (1, 4), (2, 3),
    (3, 7), (4, 5), (4, 7), (4, 8), (5, 6),
]
coefficient_sum = np.zeros(G.shape[1], dtype=int)
for i in single_rows:
    coefficient_sum ^= G[i]
for i, j in pair_rows:
    coefficient_sum ^= G[i] & G[j]
single_weights = [int(G[i].sum()) for i in single_rows]
pair_overlap_weights = [int((G[i] & G[j]).sum()) for i, j in pair_rows]
right_hand_side = (
    sum(weight // 2 for weight in single_weights)
    + sum(weight // 2 for weight in pair_overlap_weights)
) % 2
assert not coefficient_sum.any()
assert right_hand_side == 1

out = {
    "source": "arXiv:2606.07734v1 Appendix D, Eq. (D3): 47T->1CCZ on N=9",
    "shape": list(G.shape),
    "rank_G": int(f2_rank(G)),
    "rank_S": int(f2_rank(S)),
    "ch_conditions_pass": bool(ch[0]),
    "dX": int(dX), "dZ": int(dZ),
    "G_distinct_cols": len(set(tuple(c) for c in G.T)),
    "WE_S": {str(w): c for w, c in sorted(code_weight_distribution(S.astype(np.uint8)).items())},
    "native_no_correction": {"found": res.found, "status": res.status},
    "mod2_contradiction": {
        "single_rows_1_based": [i + 1 for i in single_rows],
        "pair_rows_1_based": [[i + 1, j + 1] for i, j in pair_rows],
        "single_weights": single_weights,
        "pair_overlap_weights": pair_overlap_weights,
        "coefficient_xor_is_zero": True,
        "right_hand_side_parity": int(right_hand_side),
    },
}
print(json.dumps(out, indent=1))
json.dump(out, open("reports/jacinto_d3_47_verification.json", "w"), indent=1)
