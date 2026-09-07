# Reproducing the computational results

This guide maps the computational claims in *Native CCZ codes from projective geometry: a
48-qubit code and a length lower bound* and its supplementary material to the corresponding
programs and retained outputs. Run the programs from the repository root with `PYTHONPATH=.`.
Each program exits with a nonzero status when a checked assertion fails.

If the local Numba cache is unavailable, prefix a command with `NUMBA_DISABLE_JIT=1`. This changes
performance but not the exact arithmetic.

## Supplementary Note A

| Result | Program | Retained output | Expected result | Approximate cost |
|---|---|---|---|---|
| Parameters and CH conditions of the 47-qubit code | `scripts/verify_jacinto_d3.py` | `reports/jacinto_d3_47_verification.json` | ranks 9 and 6, `dX=16`, `dZ=3`, all CH checks pass | seconds |
| No native physical sign pattern | `scripts/verify_jacinto_d3.py` | `reports/jacinto_d3_47_verification.json` | native solve is `gf2_unsat` | seconds |
| Exact contradiction printed in Note A | `scripts/verify_jacinto_d3.py` | `mod2_contradiction` in the same report | coefficient XOR is zero and right-hand side parity is one | included in the same run |

## Supplementary Note B

| Result | Program | Retained output | Expected result | Approximate cost |
|---|---|---|---|---|
| Ranks, exact logical-coset minima, and distances of Q48 | `scripts/q48_min_logical_basis.py` | `reports/q48_min_logical_basis.json` and `.md` | ranks 9 and 6, `dX=16`, `dZ=3`, coset minima `18,16,18,16,18,16,18` | seconds |
| Explicit weight-three Z-logical | `scripts/q48_min_logical_basis.py` | `z_logical_witness` in the same JSON report | zero stabilizer syndrome and nonzero logical syndrome | included in the same run |
| Polynomial for the physical signs and the `26T+22Tdag` count | `scripts/verify_tpattern_paper_form.py` | `reports/q48_tpattern_paper_form.json` | all 48 values agree, with block counts 6, 6, and 10 | seconds |
| Native signed-weight identity for every codeword | `scripts/verify_native_ladder.py` | `reports/q48_native_ladder.json` | all checks pass on 512 codewords | seconds |

The two sign checks serve different purposes. The first compares the formula printed in the paper
with the stored flip vector. The second evaluates the native congruence on every codeword.

## Supplementary Note C

| Result | Program | Retained output | Expected result | Approximate cost |
|---|---|---|---|---|
| Initial lower-bound reductions and candidate weights | `scripts/kt_floor_theorem.py` | `reports/kt_floor_theorem.json` | the stated candidate list and reductions | seconds |
| Rank-five closure | `scripts/close_s5_corrected.py` and `scripts/simplex_s5_via_library.py` | `reports/s5_corrected_closure.json` and `reports/s5_simplex_linearized_decision.json` | 62 non-simplex activities rejected and all 49,741,825 simplex pairs rejected | about 1 minute for the non-simplex cases and 9 minutes for the vectorized simplex check |
| Rank-six and rank-seven coverage | `scripts/audit_floor_affine_orbits.py` and `scripts/m7_w32_classes.py` | `reports/floor_affine_orbit_audit.json` and `reports/m7_w32_classes.json` | the coverage ledger is complete and no tested rank-seven label map survives | minutes |
| Rank-six weight-32 case | `scripts/verify_s6_weight32.py` | `reports/s6_weight32_exact.json` | three affine classes: one analytic rank-five reduction and four exhaustive unpunctured/punctured tests, all unsatisfiable; every origin transport verified | about 20 seconds with compilation enabled |
| Rank-eight orbit coverage | `scripts/m8_h_rep_completeness_audit.py` | `reports/m8_h_rep_completeness_audit.json` | every listed orbit is disjoint, contained in its class, and complete | about 30 seconds with compilation enabled, or about 2 minutes with JIT disabled |
| Rank-eight label-map systems | `scripts/m8_floor_manifest.py` | `reports/m8_floor_manifest.task0.json` through `task3.json` | 16,798,112 base systems, with 3,523,584 full third-label solves, all inconsistent | 12,868.9 recorded task-seconds in total |
| Separate weight-14 cases | `scripts/m8_h14_sweep.py` | `reports/m8_h14_sweep_final.json` | 124 representatives and 840,384 decided systems, with no missing, deferred, or surviving case | cluster-scale sweep, timing not retained in the final summary |
| Combined rank-eight totals | inspect the two preceding result groups | the same retained outputs | 17,638,496 systems, of which 13,274,528 fail K3 and 4,363,968 reach an inconsistent third-label system | no additional computation |
| Receipt consistency | `scripts/verify_supplement_receipts.py` | `reports/supplement_receipt_audit.json` | every Note C total and completeness field agrees | seconds |

The four rank-eight task summaries and the weight-14 JSONL file are the retained machine-readable
search outputs. The compact JSON files record the aggregate checks used in the paper and supplement.
The heavy sweeps are not part of a routine local test run.

For the rank-six weight-32 case, use the new exact receipt rather than the historical
coverage summary or sampled invariant buckets. The script prints its full JSON receipt.
The two quadratic classes test respectively 12,042,241 and 2,224,129 unpunctured pairs,
and 3,006,465 and 555,009 punctured pairs. Explicit linear maps cover all 32 punctures
within each class; no additional punctured label solve is needed. Supplementary Note C
gives the affine-hyperplane reduction and the coordinate-change argument.

## Other independent checks

The following checks support displayed constructions whose mathematical derivations already appear
in the paper.

| Result | Program | Retained output | Expected result | Approximate cost |
|---|---|---|---|---|
| FG48 support, ranks, distances, and two-weight enumerator | `scripts/verify_e1_paper_claims.py` | `reports/e1_paper_claims_verification.json` | 48 columns and stabilizer weights 24 and 32 | seconds |
| FG48 block generator form | `scripts/family_block_forms.py` | program assertions | row spaces and block sizes agree | seconds |
| Full and stabilizer indicator identities | `scripts/verify_alpha_indicators.py` and `scripts/verify_sec31_claims.py` | program assertions | all stated indicator checks pass | seconds |

The analytic uniqueness proof in Supplementary Note D has no computational dependency.

## Minimal reproduction sequence

Run the four quick checks used by Supplementary Notes A and B:

```bash
PYTHONPATH=. python scripts/verify_jacinto_d3.py
PYTHONPATH=. python scripts/q48_min_logical_basis.py
PYTHONPATH=. python scripts/verify_tpattern_paper_form.py
PYTHONPATH=. python scripts/verify_native_ladder.py
```

Then inspect the retained Note C outputs before deciding whether to repeat a heavy sweep. The stored
task summaries and weight-14 records preserve the exact terminal outcomes used by the receipt audit.
