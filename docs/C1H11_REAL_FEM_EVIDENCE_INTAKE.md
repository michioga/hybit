# C1-H11 — Real FEM evidence intake and same-budget PCG comparison

**Base:** `69dfa7463bf9b53a491f2f80e76ab18fe892f902` (`develop/0.10.0`).

## Purpose and limitations

C1-H2–H10 demonstrated experimental ABTM local-direct routing and prepared
RHS reuse on constructed SPD problems. C1-H11 is **data intake and a falsifiable
comparison protocol**, not proof of real-FEM rescue. No real FEM sample or
physical RHS is bundled or fabricated. The default solver is unchanged.

Use a *complete single-process* FEM operator after Dirichlet constraints and
its **matching physical RHS from the same solve call**, plus an independent
record of SPD justification and provenance. FrontISTR serial dumps are eligible
only after the operator verifies one rank, no unsatisfied ghost ownership and
a matching call number. This tool refuses non-zero-rank files in the same call.

`mtx_fem_audit` applies numerical sufficient SPD certificates for exactly
symmetric strictly/irreducibly weakly diagonally dominant systems. A symmetric
matrix not covered by those sufficient conditions is **SPD_UNVERIFIED**, even
if Jacobi-PCG converges: human-supplied independent evidence is required and
is recorded as a declaration, not magically converted into a certificate.
An approximatively symmetric matrix is not proven exactly symmetric; review
numerical discrepancies before trusting PCG. Audit the FEM constraints
outside this tool. In particular, do not treat a positive diagonal as SPD.

## JSON manifest

Create an operator-owned manifest outside the repository (real paths only):

```json
{
  "cases": [
    {
      "id": "serial_frontistr_case01",
      "origin": "frontistr_serial",
      "matrix": "/absolute/path/dump_matrix_1_0.mm",
      "rhs": "/absolute/path/dump_matrix_1_0.rhs",
      "xref": null,
      "provenance": "/absolute/path/case01_provenance.txt",
      "spd_evidence": "/absolute/path/case01_spd_justification.txt"
    }
  ]
}
```

`origin` can also be `external_fem_serial` with its own matrix/RHS pair.
`provenance` should state FEM host/version, mesh ID/hash, DOF order, constraints,
load case, solver snapshot, MPI rank count=1 and whether RHS was physical or
manufactured. `spd_evidence` documents independent SPD reasoning, including
material positivity, constrained null modes, matrix symmetry and boundary
conditions; a text assertion by itself is not a mathematical proof.
`xref` is optional: absent reference means **forward error is unknown**, not
zero. Zero RHS is excluded from rescue statistics.

## Commands

```bash
cd ~/hybit
bash benchmarks/scripts/C1H11_run_real_fem.sh build
bash benchmarks/scripts/C1H11_run_real_fem.sh audit \
  --manifest ~/fem_evidence/cases.json \
  --out ~/c1h11_evidence/audit01
bash benchmarks/scripts/C1H11_run_real_fem.sh run \
  --manifest ~/fem_evidence/cases.json \
  --out ~/c1h11_evidence/run01 \
  --budgets 120 360 1200 --caps 128 256 512 \
  --tol 1e-10 --factor-mib 64
```

Output directories **must be new**; no benchmark output enters the source
integrity manifest. The run reports all 9 budget/cap pairs per case, including
negative results. All arms share the same matrix, physical RHS, initial guess
and iteration ceiling; repeated baseline measurements across caps may vary.
This is an initial functional screen, not a statistically paired runtime study.

Output: `mtx_audit.csv`, `intake.json` (SHA-256 for inputs and provenance),
`invocation.json` (git HEAD and dirty state), raw per-configuration `.log`
with command and STDERR, `results.csv`, and `summary.json`. The **peak RSS**
from `/usr/bin/time` is whole-process maximum, not per-arm peak or Cholesky
storage. `local_factor_bytes` and RSS are different measures.

## Classifications and fail-closed behavior

`RESCUE` requires Jacobi to miss tolerance within the budget and Hybrid to
converge with original-CSR verified relative residual at/below the same
`tolerance` **and an actual local escalation**. `BOTH_NOT_CONVERGED`,
`BOTH_CONVERGED`, `RELIABILITY_REGRESSION`, `FALSE_REPORTED_CONVERGENCE` and
`AUTO_CONVERGED_WITHOUT_LOCAL_FACTOR` stay visible, as do process and parse
failures. Ordinary numerical MaxIterations is **not** silently renamed success.
The Python tool returns nonzero on error/regression, with full results preserved.

`STATUS_RESIDUAL_MISMATCH` is also an error: a non-Converged status with an
original-CSR residual already at/below tolerance is not counted as a Hybrid
rescue. A Converged status with residual above tolerance remains
`FALSE_REPORTED_CONVERGENCE`. Duplicate budgets or caps are rejected before
output creation to prevent log-name collisions. On parse errors and timeouts,
the runner retains already captured output or partial subprocess output.

For a verified real-FEM study, compare repeated physical RHS and appropriate
external PETSc GAMG / hypre BoomerAMG separately, with their setup, converged
true residual and peak RSS. These comparator implementations are **not run**
by C1-H11. Distinguish a fully assembled serial matrix from a rank-local
FrontISTR operator; no MPI gluing, Fortran embedding or new production
self-routing is in this patch.
