# Building codes with transversal CCZ using projective geometry and SAT solvers

Public reproducibility package for the paper by Bohan Lu and Kenneth R. Brown.

This README contains the reproduction instructions, computational coverage arguments,
and classification details supporting the paper. Equation, table, algorithm, and
appendix numbers refer to the paper.

## Quick start

The quick checks were tested with Python 3.11.8 and the dependency versions in
[requirements-quick.txt](requirements-quick.txt): NumPy 2.2.6, galois 0.4.6,
Numba 0.61.2, llvmlite 0.44.0, z3-solver 4.16.0.0, and typing_extensions 4.15.0.
From a fresh clone of this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-quick.txt -e .
python -m pip check
python scripts/verify_quick.py --report .quick-check/verification.json
```

Expected final output:

```text
All five quick verification programs and report checks passed. Archived reports unchanged.
```

The runner checks the 47-qubit comparison code, the distances and a logical
witness of the 48-qubit code, its physical-sign formula and signed-weight
conditions, and all 138 family witnesses against Table 3. Existing programs run
in a temporary copy, so the archived reports stay unchanged. The fresh receipt
records the program and data digest, interpreter, dependencies, and check results.
Run without Python's `-O` option: the existing verification programs use assertions.

The [automated quick checks](https://github.com/jerrylvx/gate2code-public/actions/workflows/quick-verification.yml)
use the same dependency constraints on Ubuntu with Python 3.11. Passing quick
checks does not establish exhaustive search coverage. The full length-48
classification, large lower-bound sweeps, and SAT search remain separate manual
workflows documented below.

The latest local quick-check results and their limits are documented in
[Quick verification](docs/QUICK_VERIFICATION.md). Archived outputs elsewhere in
the repository retain their original scope.

## Terminology and notation

A *single-qubit transversal* implementation of logical CCZ uses physical
$`T/T^\dagger`$ gates, as in Definition 3.4. A *quasi-transversal* implementation
allows a diagonal Clifford correction. Satisfying the nine Campbell–Howard (CH)
conditions in Table 1 does not by itself imply a single-qubit transversal
implementation.

For $`G=[K;S]`$, the three rows of $`K`$ represent the $`X`$-logical generators, and the
$`s`$ independent rows of $`S`$ generate the classical code
$`\mathcal C_2\subseteq\mathcal C_1=\mathrm{rowspan}(G)`$.
The stabilizer rank is $`s=\mathrm{rank}\,S=\dim\mathcal C_2`$.
In label-map calculations, $`\kappa_i`$ denotes the logical row $`K_i`$.
The binary flip vector $`\gamma`$ specifies the physical signs
$`\Gamma_j=(-1)^{\gamma_j}`$: $`\gamma_j=0`$ selects $`T`$, and $`\gamma_j=1`$ selects
$`T^\dagger`$.

Existing program names and stored output fields retain their identifiers. The
following table relates those identifiers to the paper's notation.

| Program identifier | Meaning in the paper |
|---|---|
| `native`, `native_CI`, `native_no_correction` | Status of the search for a single-qubit transversal $`T/T^\dagger`$ implementation. |
| `NoCorrResult.delta`, classification bit mask `delta` | The binary flip vector $`\gamma`$. |
| `NoCorrResult.gamma` | Physical exponents in $`\{1,7\}`$, congruent to the signs $`\Gamma\in\{1,-1\}^n`$ mod 8. |

The status `lift_unsat_sampled` means that a bounded search found no pattern
and leaves existence unresolved. The complete length-48 classification below
enumerates all solutions for each class.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# For the complete length-48 classification:
pip install -e '.[classification]'
# For the CryptoMiniSat/PySAT search:
pip install -e '.[sat]'
```

The quick-check runner also validates reported ranks, distances and exclusion
results that individual programs print without asserting. Any failed assertion,
unexpected report value, or nonzero program exit makes the runner fail.
Some longer searches use Numba. If its cache is unavailable, set
`NUMBA_DISABLE_JIT=1`; this changes performance, not the exact arithmetic.

## 47-qubit code of Jacinto et al.

```bash
PYTHONPATH=. python scripts/verify_jacinto_d3.py
```

Expected: generator and stabilizer ranks 9 and 6, distances $`d_X=16`$ and $`d_Z=3`$,
and all nine CH conditions satisfied. The equations for a single-qubit transversal
$`T/T^\dagger`$ implementation of logical CCZ have no binary solution (`gf2_unsat`).
The output includes an explicit contradiction: selected coefficient rows sum to
zero over $`\mathbb F_2`$, while their right-hand sides sum to one.

Retained output: [jacinto_d3_47_verification.json](reports/jacinto_d3_47_verification.json),
including `mod2_contradiction`. The check takes seconds.

## 48-qubit code and physical signs

```bash
PYTHONPATH=. python scripts/q48_min_logical_basis.py
PYTHONPATH=. python scripts/verify_tpattern_paper_form.py
PYTHONPATH=. python scripts/verify_native_ladder.py
```

The first program checks ranks 9 and 6 and computes $`d_X=16`$, $`d_Z=3`$.
For logical coefficients `001,010,011,100,101,110,111`, the minimum coset weights
are `18,16,18,16,18,16,18`. The retained output includes a weight-three $`Z`$-logical
with zero stabilizer syndrome and nonzero logical label.

The second program compares the paper's physical-sign formula with all 48 stored
signs, including the $`26\,T+22\,T^\dagger`$ count. The third checks the signed-weight
congruence on every one of the 512 codewords. Each check takes seconds.

Retained outputs:
[distance checks](reports/q48_min_logical_basis.json),
[physical-sign formula](reports/q48_tpattern_paper_form.json), and
[signed-weight checks](reports/q48_native_ladder.json).

## Length lower bound

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
Appendix D of the paper gives the mathematical reductions. The coverage details
below explain the finite calculations and the origins counted in Table 10.

### Notation and finite-search coverage

Algorithm 1 tests label maps on each syndrome support
$`\mathcal P\subseteq\mathbb F_2^s\setminus\{0\}`$, using $`V_{\mathcal P}`$
of equation (35). The pair counts below refer to ordered pairs
$`(\kappa_1,\kappa_2)`$ of nonzero logical rows in $`V_{\mathcal P}`$ satisfying
K2 and KS2. Each linear system for $`\kappa_3`$ is solved over $`\mathbb F_2`$.

At $`s=6`$ and $`|c|=32`$, Appendix D.4, Step 2 excludes the degree-one class
analytically. The two remaining classes have $`c=y_6+q`$:

| $`q`$ | $`\dim V_{\mathcal P}`$ | Unpunctured pairs tested |
|---|---:|---:|
| $`y_1y_2`$ | 13 | 12,042,241 |
| $`y_1y_2+y_3y_4`$ | 11 | 2,224,129 |

Every linear system for $`\kappa_3`$ is inconsistent. Appendix D.2.2 shows that
excluding CH triples on one full support with $`c(a_0)=0`$ excludes them on
every translated or punctured support.

For $`s=7`$ and $`|c|=36`$, the enumeration fixes the cubic terms and varies all
lower-degree terms. The degree-three part, modulo polynomials of degree at most
two, transforms as an alternating trilinear form: associate $`y_i y_j y_k`$ with
$`e_i^*\wedge e_j^*\wedge e_k^*`$, where the $`e_i^*`$ are the coordinate
functionals on $`\mathbb F_2^7`$.
[Cohen–Helminck, Theorem 2.2, Table 1 and Corollary 2.4](https://doi.org/10.1080/00927878808823558)
give the 11 nonzero alternating-form representatives over $`\mathbb F_2`$.
The direct Boolean quotient classification lists twelve cubic parts, including
zero: [Braeken et al., Section 5.4 and Table 11](https://eprint.iacr.org/2004/248).
Enumerating every quadratic, linear and constant correction to each
representative covers every cubic polynomial up to a linear coordinate change.
The zero cubic part contributes no weight-36 function by the quadratic weight
formula in [MacWilliams–Sloane, Chapter 15, Theorems 4–5](https://doi.org/10.1016/S0924-6509(08)X7030-8).
The retained enumeration contains 9,792 weight-36 functions, counted separately
even when affine-equivalent.

Translations preserve the fixed cubic part and the weight, so the enumerated
family is closed under translations. Every point belongs to the same number
of its supports. Counting incidences between functions and their 36 support
points gives

```math
\#\{c:c(0)=1\}=\frac{9792\cdot36}{128}=2754,
\qquad
\#\{c:c(0)=0\}=9792-2754=7038.
```

The unpunctured tests use these 7,038 functions with $`c(0)=0`$, as counted in
Table 10. Every possible origin outside a support gives one of these functions
after translation. Appendix D.2.2 covers the punctured supports.

The unpunctured $`s=6`$, $`|c|=36`$ case tests 43,450,369 nonzero ordered pairs
satisfying K2 and KS2. Across the four $`s=8`$ cases with
$`|c|=32,34,36,38`$, 17,638,496 systems reach the label-map tests.
K3 rejects 13,274,528 systems before solving for $`\kappa_3`$; the remaining
4,363,968 systems are inconsistent.

For every orbit partition in Table 9, the verification program checks that the
orbits are disjoint, lie in the stated weight class, and have total size equal
to that class. These checks and the mathematical reductions specify the
coverage claimed by the finite calculations.

## Family verification and complete length-48 classification

Install the `classification` extra above, then run:

```bash
PYTHONPATH=. python scripts/verify_family_table.py --check
PYTHONPATH=. python scripts/classify_n48/check_n48.py
```

The family check verifies the 138 archived generator matrices and binary flip
vectors in [data/family_witnesses/grid](data/family_witnesses/grid), indexed by
[index.json](data/family_witnesses/grid/index.json). Expected terminal output:

```text
138 witnesses verified
all printed values match
```

The classification regenerates its intermediate files locally. Expected final
output:

```text
n=48: 4152 classes, 17 with a single-qubit transversal pattern, 2 such codes C1, weight enumerators match the archived Table 3 codes
```

The family check takes seconds. The full classification is a longer computation;
its running time depends on the environment. Earlier checks are recorded in
[reports/release_verification.json](reports/release_verification.json), a historical
receipt for the source snapshot named in that file. A new quick-check receipt
does not replace that record or rerun the full classification.
The large lower-bound sweeps use the separately retained outputs listed above.

### What is verified for each family code

For Table 3, the archive retains 138 matrices $`G=[K;S]`$ and flip vectors
$`\gamma`$. Distances or weight enumerators distinguish the representatives on
each support, so the counts are lower bounds except at $`n=48`$.

A program independent of the search code verifies that the columns of $`S`$ form
the stated subspace complement, $`\mathrm{rank}\,S=s`$, and
$`\mathrm{rank}\,G=s+3`$. Every overlap of one, two, or three distinct rows has
even weight except $`|K_1\wedge K_2\wedge K_3|`$, which is odd. Equation (19)
holds for every one of the $`2^{3+s}`$ codewords.

The program computes $`d_X`$ by minimizing
$`|K^{\top}\vec x+S^{\top}\vec y|`$ over nonzero
$`\vec x\in\mathbb F_2^3`$ and $`\vec y\in\mathbb F_2^s`$.
For $`d_Z`$, a breadth-first search starts at
$`(0,0)\in\mathbb F_2^3\times\mathbb F_2^s`$. Each step adds a full column
$`(\alpha_j,\beta_j)`$, with logical label $`\alpha_j`$ and syndrome $`\beta_j`$.
The first reached pair $`(\alpha,0)`$ with $`\alpha\ne0`$ gives the minimum
weight of a $`Z`$-logical, as in Section 2 of the paper.

### Classification domain and equivalence

At $`n=48`$, fix $`\mathcal C_2=\mathrm{rowspan}(S_{\mathrm{FG48}})`$ and
consider all $`\mathcal C_1\supset\mathcal C_2`$ with
$`\dim(\mathcal C_1/\mathcal C_2)=3`$ that admit logical rows satisfying the nine
CH conditions. Codes are identified up to a coordinate permutation.
Proposition 2.3 shows that the fixed choice covers every projective 8-divisible
$`[48,6]`$ stabilizer code up to such a permutation.

With $`S=S_{\mathrm{FG48}}`$ fixed, a code is determined by its logical rows
$`K_1,K_2,K_3`$. In the coordinates $`(u,w)`$ of Corollary 2.4, the removed subspace
is $`P=\{(u,0):u\in\mathbb F_2^4\}`$. Its preserving subgroup
$`H\le\mathrm{GL}(6,2)`$ consists of

```math
(u,w)\longmapsto(Mu+Bw,Cw),\qquad
M\in\mathrm{GL}(4,2),\quad C\in\mathrm{GL}(2,2),\quad
B\in\mathbb F_2^{4\times2}.
```

The group has order 30,965,760 and permutes the 48 surviving columns.
Every coordinate permutation preserving $`\mathcal C_2`$ induces an invertible
change of its six generator rows, so $`H`$ is the full coordinate-permutation
group preserving $`\mathcal C_2`$. Adding an element of $`\mathcal C_2`$ to a
logical row leaves $`\mathcal C_1`$ unchanged.

A CH triple $`(\kappa_1,\kappa_2,\kappa_3)`$ satisfies the CH conditions on
$`\mathcal P`$, as in Appendix B. By equation (35), a logical row satisfies K1,
KS1 and KS3 exactly when it lies in $`V_{\mathcal P}`$. This space has dimension
27, and dimension 21 modulo $`\mathcal C_2`$.

### Exhaustive orbit enumeration

Candidates are cosets $`\kappa_i+\mathcal C_2`$ in
$`V_{\mathcal P}/\mathcal C_2`$; stabilizers fix these cosets. The program splits
all $`2^{21}`$ candidates for $`\kappa_1`$ into $`H`$-orbits. For each representative,
candidates for $`\kappa_2`$ satisfying the pairwise CH conditions are split into
orbits under the stabilizer of $`\kappa_1+\mathcal C_2`$. Candidates for
$`\kappa_3`$ satisfying the remaining conditions, including odd
$`|\kappa_1\wedge\kappa_2\wedge\kappa_3|`$, are split into orbits under the
subgroup fixing both previous cosets.

At each stage, breadth-first enumeration visits every candidate and assigns
it to one orbit. The program uses Schreier–Sims to check that every generated
stabilizer subgroup has order equal to the parent-group order divided by the
orbit size. The result is 23,940 orbits of ordered triples under $`H`$.
Merging the six permutations of each triple gives 4,152 classes of CH triples
under $`H`$, additions from $`\mathcal C_2`$, and permutations of the three logical
rows. General logical-basis changes have not yet been identified at this stage.
Every CH triple gives a code with $`d_X=16`$ and $`d_Z=3`$.

### Physical-sign test and grouping into codes

For every CH-triple class, the program enumerates the full solution set of
equation (19). Exactly 17 classes admit a $`T/T^\dagger`$ implementation of
logical CCZ. For every representative triple, the program applies all 168
elements of $`\mathrm{GL}(3,2)`$, finds each resulting triple's class,
and merges the related classes.

Logical-basis changes preserve the CH conditions by Corollary A.6: the
triple-overlap parity on $`\mathcal C_1/\mathcal C_2`$ is an alternating
trilinear form, and every invertible binary $`3\times3`$ matrix has determinant 1.
The merged groups give 192 classical codes $`\mathcal C_1`$ up to $`H`$:

| CH-triple classes per code | 4 | 7 | 10 | 16 | 28 |
|---|---:|---:|---:|---:|---:|
| Number of codes | 3 | 4 | 24 | 53 | 108 |

The 17 classes admitting a single-qubit transversal implementation form exactly two of these
classical codes, with 10 and 7 CH-triple classes respectively. Their CSS codes
include $`\mathcal Q_{48}`$. The two CSS codes are inequivalent under every
coordinate permutation: their classical codes $`\mathcal C_1`$ have different
weight enumerators, with 10 versus 18 codewords of weight 16. The classification
check repeats the enumeration and checks these counts against the two archived
length-48 codes.

## SAT search and provenance of the 48-qubit code

With the `sat` extra installed, run:

```bash
PYTHONPATH=. python scripts/cms_verify_instance.py --task 48 --s 6 --threads 1 --out reports/cms_s6_n48.json
```

The program uses the column indicator $`\chi_S`$ of Section 3.1, with
$`\beta\in\mathbb F_2^s`$ and block length $`n`$. It sends the CH parity equations
to CryptoMiniSat and the fixed-length equation
$`\sum_{\beta\ne0}\chi_S(\beta)=n`$ to a PySAT cardinality encoding.
Software references: [Soos et al. (2009)](https://doi.org/10.1007/978-3-642-02777-2_24)
and [Ignatiev et al. (2018)](https://doi.org/10.1007/978-3-319-94144-8_26).

A solution satisfies the nine CH conditions but need not admit a
$`T/T^\dagger`$ implementation of logical CCZ as defined in Definition 3.4.
The archived first solution does not admit such an implementation; the first
solution from a new run may depend on the solver version. The logical rows of
$`\mathcal Q_{48}`$ were found by enumerating
solutions of the same encoding with [Z3](https://doi.org/10.1007/978-3-540-78800-3_24)
and retaining a solution for which equation (19) has a binary solution.

The cluster enumeration program is
[scripts/provenance/cluster_enumerate_48.py](scripts/provenance/cluster_enumerate_48.py).
Its retained solution is
[data/provenance/e1_code_t005_c0048.npz](data/provenance/e1_code_t005_c0048.npz).
[code_48_3_3_fg48.npz](code_48_3_3_fg48.npz) stores the same code in the paper's
FG48 coordinates. The stored SAT output and timing record are
[reports/cms_s6_n48.json](reports/cms_s6_n48.json) and
[reports/timing/cms_s6_n48_timing.json](reports/timing/cms_s6_n48_timing.json).
The cluster search itself is not a quick local check.

## Citation

Please cite the paper when using the constructions or results:

```text
Bohan Lu and Kenneth R. Brown.
Building codes with transversal CCZ using projective geometry and SAT solvers.
2026.
```

The permanent arXiv link will be added after the article is publicly announced.

## License

The code is available under the [MIT License](LICENSE).
