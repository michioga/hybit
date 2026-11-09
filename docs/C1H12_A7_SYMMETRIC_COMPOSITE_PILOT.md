# C1-H12-A7: Native two-level symmetric composite numerical pilot

Status: **owner WSL results pending** (2026-11-09). No production change or
performance claim. The 36-arm experiment is opt-in and isolated in a Rust
example. A2–A5 historical evidence remains unmodified. A6 is uncommitted.

## Rationale

A6 periodic residual checks did not establish a general convergence gain.
On its 63 process records, `p16` regressed one `multi_patch` arm from
`Converged` to `MaxIterations`, improved another, and `p32` reproduced the
A5 numerical trajectories. This motivates testing a numerically different
preconditioner before modifying restart controls or public defaults.

For the original symmetric positive definite system `A`, let `B^-1` denote
the positive Jacobi inverse and `M_add^-1` the existing positive two-level
block-Jacobi plus direct coarse correction. The control is `M_add^-1`.
The experimental composite is

```text
M_A7^-1 = B^-1 + (I - B^-1 A) M_add^-1 (I - A B^-1).
```

For exactly symmetric `A` and SPD `B^-1`, `M_add^-1`, this is SPD:

```text
x^T M_A7^-1 x = x^T B^-1 x
                 + [(I - A B^-1) x]^T M_add^-1 [(I - A B^-1) x] > 0.
```

The factors are kept fixed during each PCG solve. It is a *symmetric
composite two-level preconditioner*, **not a multilevel AMG V-cycle**.

## Owner WSL preflight and experiment

Apply the separate A7 archive using its `C1H12_A7_apply.sh`. It requires
branch `develop/0.10.0`, checkpoint `1f3eaa5`, an unstaged A6 experiment,
and zero staged files. It does not commit or push changes.

```bash
cd ~/hybit
BASE="$HOME/c1h11_evidence/c1h12_a6_periodic_1CKka9"
bash benchmarks/scripts/C1H12_A7_symmetric_pilot.sh --dry-run "$BASE"
bash benchmarks/scripts/C1H12_A7_symmetric_pilot.sh --run "$BASE"
```

It runs 3 original FEM matrices x 6 coarse aggregation/basis arms x 2
policies (additive and symmetric), for 36 processes. Each uses a generated
`b=A*1`, target coarse dimension 512, `F64` transfers, one process thread,
original-CSR certified PCG, `tol=1e-10`, and 600 total iterations. The
comparison gate requires all records, matching A6 inputs and full source
commit, accurate execution statuses and original-CSR true residuals.

The additive control must reproduce the A6 A5 outcome status and iteration
count for each corresponding arm. If it does not, review experiment setup
before interpreting the symmetric results. Both positive and negative
numerical outcomes are retained.

The symmetric arm costs *two additional sparse matrix-vector products* and
five full-length scratch arrays per preconditioner application. This
extra work must be recorded in the evidence. A decrease in iterations does
not alone establish a total-solve or end-to-end speedup.

## Next decision gate

1. Verify `36/36` raw numerical/log/provenance PASS and the independent
   tests/Clippy gates. Do not accept a false-convergence claim.
2. Compare convergence count and per-case iteration counts, and then
   `setup_ms`, `solve_ms` and total wall, taking the two extra SpMVs into
   account.
3. If the native composite is promising, repeat timings and evaluate a
   genuine multilevel aggregation/restriction/coarse-numeric cycle in a
   separate feature behind the stable CSR32 API. ABTM A2/A3 symbolic graph
   kernels may optimize hierarchy construction, but no symbolic result on
   its own establishes PCG convergence or AMG equivalence.

No PETSc/hypre dependency or production/public-ABI modification is made.
