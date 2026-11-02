# C1-H10 — Prepared opt-in ABTM factor reuse: multi-RHS cost experiment

**Base:** `bdc3c59f33e15779f28a6d7fe23514b646f32936` (`develop/0.10.0`).
**Scope:** evidence-only synthetic SPD timing; no solver/controller/API/FFI changes.

## Question

C1-H9 proved that, on a fixed SPD matrix, each new RHS is first diagnosed and
only an exactly matching strong-component candidate may reuse a previously
verified local Cholesky factor. This does **not** establish whether repeated
RHS solves amortize the initial factorization and the unavoidable per-RHS
ABTM diagnostic. Measure that trade-off instead of assuming a speedup.

## Comparison and fairness

The new `c1h10_rhs_amortization` Rust example exercises three real HyBIT paths:

1. **plain:** build one Jacobi preconditioner per batch, run 30-iteration PCG
   independently for each RHS, and independently compute original-CSR residual.
2. **fresh:** call `HybitSolver::solve_csr32` on each RHS, forcing fresh
   analysis/prepare and local factorization when promoted.
3. **prepared:** build one `HybitPreparedSystem` for the batch and call its
   `solve` repeatedly. Every RHS still goes through C1-H9 probe/diagnostics;
   a verified local factor is reused only for an exact region match.

All arms use the **same original SPD CSR matrix**, RHS order, exact reference,
zero initial guess, `1e-10` tolerance, 30-iteration ceiling, one 128-DOF
candidate maximum, one local region, zero overlap, one escalation, and 1 MiB
factor budget. A 96-DOF Dirichlet chain and a weak positive Laplacian bridge
create a locally hard component. The RHS set includes a canonical hard load,
its exact `-2` scaling for the C1-H9 reuse invariant, and independent
manufactured smooth/harmonic loads; every eighth is an exterior-only easy
control. The distinct harmonic loads are **not independent real FEM loads**.

For all modes, compute original-CSR true residual and known-reference forward
error. A `Converged` claim with residual above `1e-10` or relative forward
error above `1e-6` is a hard failure. Fresh and prepared must agree in
status, iterations, promotion decisions, and numerical solution (tight
error bound). The first RHS must reproduce the C1-H9 rescue. On batches of 2+
RHS, the second, exactly scaled RHS must reuse the factor.

The cumulative wall time includes creation of each arm's solver/preconditioner,
all solve calls, and the extra independent residual/forward-error audit. In
addition, Auto telemetry records analysis, prepare, probe, diagnostics,
local-factor, total-solve times, iteration sum, cache reuse count and peak
persistent factor bytes. Differences in repeated per-call overhead and
additional Auto certification mean these are **end-to-end research paths**,
not an isolated numerical-kernel microbenchmark. Matrix creation and RHS
construction are excluded because inputs are identical across arms.

## Run

```bash
cd ~/hybit
export PYTHONDONTWRITEBYTECODE=1
bash benchmarks/scripts/C1H10_run_rhs_amortization.sh smoke
bash benchmarks/scripts/C1H10_run_rhs_amortization.sh full
cargo +1.99.1 test --release --workspace --locked
cargo +1.99.1 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.99.1 fmt --all -- --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
```

The full run measures 1, 2, 4, 8, 16 and 32 RHS, each with two excluded
warm-up runs and eleven recorded repetitions. Arm order rotates every run.
The smoke check uses 1, 2 and 4 RHS with one warm-up and three repetitions.
Output:

- `$HOME/c1h10_rhs_amortization/{smoke,full}/raw.csv`: each measured arm,
  iteration, convergence, factorization/reuse events and timing, including
  warmups flagged separately.
- `$HOME/c1h10_rhs_amortization/{smoke,full}/summary.csv`: medians and the
  **descriptive** prepared/fresh and prepared/plain ratios.

The number of RHS required to offset initial preparation is not predefined;
inspect the measured medians and repeat on a quiescent machine. Ratios near
one can be dominated by timer noise. No inference of a statistically reliable
crossover, physical-FEM performance, or practical acceleration is authorized
from a single short synthetic study. Even if prepared beats fresh, it may
still be slower than plain PCG on this very small matrix.

## Next evidence requirement

After this cost audit, select independent **real FEM SPD matrices** (including
localized stiffness/constraint pathologies), multiple physically meaningful
loads, and compare to PETSc/hypre on the same hardware/precision and residual
criterion. Report full time including assembly/matrix read where appropriate,
factor memory, peak RSS, and independently checked residuals. Existing
C1-G pilot FEM-like matrices are useful negative controls but do not by
themselves establish general real-FEM speedup or rescue.
