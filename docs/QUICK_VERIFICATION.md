# Quick verification

The quick-check runner executes the existing verification programs in a temporary
copy of the repository. Archived reports are not overwritten. The runner checks
program exit status and reported quantities that some programs only print.

| Program | Checked property |
|---|---|
| `verify_jacinto_d3.py` | The stored 47-qubit comparison code has ranks 9 and 6, satisfies the CH conditions, has distances 16 and 3, and gives an explicit binary contradiction excluding a physical sign pattern. |
| `q48_min_logical_basis.py` | The stored 48-qubit code has ranks 9 and 6, distances 16 and 3, and a weight-three Z-logical witness. |
| `verify_tpattern_paper_form.py` | The physical-sign formula matches all 48 signs, with 26 T and 22 T-dagger gates. |
| `verify_native_ladder.py` | All 14 signed-weight checks pass, including evaluation on all 512 codewords. |
| `verify_family_table.py --check` | All 138 family witnesses satisfy the stated geometry, CH conditions, physical signs and distances; enumerator comparisons reproduce the printed family table. |

## Local verification

A fresh Python 3.11.8 virtual environment, without system site packages, passed
installation and `pip check` with the constraints in
[requirements-quick.txt](../requirements-quick.txt). All five programs and the
runner's report checks passed on macOS ARM64. The receipt is
[quick_verification_2026-10-06.json](../reports/quick_verification_2026-10-06.json).
The receipt identifies the exact program and data digest and dependency versions.

The runner rejects Python optimization, which would disable assertions in the
existing programs. A fresh receipt must be written outside the archived
`reports/` directory. The checked-in receipt above was copied from a separate
verification run, rather than replacing an earlier record.

## Automated verification

The [Quick verification workflow](https://github.com/jerrylvx/gate2code-public/actions/workflows/quick-verification.yml)
installs the same constrained dependencies on Ubuntu with Python 3.11 and runs the
same checks. Each run retains its own receipt as an artifact and checks that
archived reports remain unchanged. Consult the workflow result for evidence of a
particular remote run; the local receipt is not evidence of remote success.

## Verification limits

Quick verification checks the stored witnesses and the properties listed above.
The runner does not repeat the exhaustive length-48 classification, large
lower-bound sweeps, or SAT search. Earlier receipts remain historical evidence
for their named snapshots. Mathematical reductions and computational coverage
arguments remain in the paper and the [README](../README.md).
