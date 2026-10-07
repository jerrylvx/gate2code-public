# gate2code reproducibility package

Computational companion to *Building codes with transversal CCZ using projective
geometry and SAT solvers*.

This README identifies the public programs, retained outputs, and expected results.
The supplementary material accompanying the current paper contains lower-bound computational details
(Note A) and the family verification and classification explanation (Note B).
The 47-qubit obstruction and the 48-qubit distance checks are documented here
rather than in separate supplementary notes.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Each verification program exits with a nonzero status if an assertion fails.
Some longer searches use Numba. If its cache is unavailable, set
`NUMBA_DISABLE_JIT=1`; this changes performance, not the exact arithmetic.

## 47-qubit code of Jacinto et al.

```bash
PYTHONPATH=. python scripts/verify_jacinto_d3.py
```

Expected: generator and stabilizer ranks 9 and 6, distances `dX=16` and `dZ=3`,
and all nine Campbell--Howard conditions satisfied. The equations for a physical
`T/Tdagger` implementation of logical CCZ have no binary solution (`gf2_unsat`).
The output includes an explicit contradiction: selected coefficient rows sum to
zero over GF(2), while their right-hand sides sum to one.

Retained output: [jacinto_d3_47_verification.json](reports/jacinto_d3_47_verification.json),
including `mod2_contradiction`. The check takes seconds.

## 48-qubit code and physical signs

```bash
PYTHONPATH=. python scripts/q48_min_logical_basis.py
PYTHONPATH=. python scripts/verify_tpattern_paper_form.py
PYTHONPATH=. python scripts/verify_native_ladder.py
```

The first program checks ranks 9 and 6 and computes `dX=16`, `dZ=3`.
For logical coefficients `001,010,011,100,101,110,111`, the minimum coset weights
are `18,16,18,16,18,16,18`. The retained output includes a weight-three Z-logical
with zero stabilizer syndrome and nonzero logical label.

The second program compares the paper's physical-sign formula with all 48 stored
signs, including the `26T+22Tdag` count. The third checks the signed-weight
congruence on every one of the 512 codewords. Each check takes seconds.

Retained outputs:
[distance checks](reports/q48_min_logical_basis.json),
[physical-sign formula](reports/q48_tpattern_paper_form.json), and
[signed-weight checks](reports/q48_native_ladder.json).

## Length lower bound: Supplementary Note A

Programs below reproduce or check the finite calculations supporting the length
lower bound. Read the retained outputs before repeating the longer searches.

| Calculation | Programs | Retained outputs |
|---|---|---|
| Initial reductions and candidate weights | `scripts/kt_floor_theorem.py` | `reports/kt_floor_theorem.json` |
| Stabilizer rank 5 | `scripts/close_s5_corrected.py`, `scripts/simplex_s5_via_library.py` | `reports/s5_corrected_closure.json`, `reports/s5_simplex_linearized_decision.json` |
| Stabilizer ranks 6 and 7 | `scripts/audit_floor_affine_orbits.py`, `scripts/m7_w32_classes.py` | `reports/floor_affine_orbit_audit.json`, `reports/m7_w32_classes.json` |
| Stabilizer rank 6, weight 32 | `scripts/verify_s6_weight32.py` | `reports/s6_weight32_exact.json` |
| Stabilizer rank 8 orbit coverage | `scripts/m8_h_rep_completeness_audit.py` | `reports/m8_h_rep_completeness_audit.json` |
| Stabilizer rank 8 label systems | `scripts/m8_floor_manifest.py`, `scripts/m8_h14_sweep.py` | `reports/m8_floor_manifest.task0.json` through `task3.json`, `reports/m8_h14_sweep_final.json` |
| Aggregate consistency | `scripts/verify_supplement_receipts.py` | `reports/supplement_receipt_audit.json` |

For rank 6 and weight 32, the exact check tests 12,042,241 and 2,224,129
unpunctured logical-row pairs in the two quadratic classes, with no surviving
third row. Use this exact output rather than historical sampled summaries.

The rank-8 outputs together account for 17,638,496 systems: 13,274,528 fail
condition K3, and the remaining 4,363,968 have an inconsistent third-row system.
The aggregate consistency check compares these totals with the retained outputs.

The longer rank-8 searches are not routine quick checks. Four archived tasks
record 12,868.9 task-seconds in total; the separate weight-14 sweep is also a
substantial computation. Successful execution verifies the implemented checks.
The mathematical coverage arguments are in Appendix D of the paper and Note A
of the supplement.

## Family and classification: Supplementary Note B

The family data, SAT search programs, and complete length-48 classification
programs are not included in this public revision. Those materials are available
from the authors on request. The supplement states the classification scope and
equivalence operations behind the exact two-code count in Table 3 of the paper.

## License

The code is available under the [MIT License](LICENSE).
