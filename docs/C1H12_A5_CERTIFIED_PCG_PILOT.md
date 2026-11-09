# C1-H12-A5: true-residual-certified PCG restart (opt-in experiment)

Status: **assistant candidate; owner WSL 21-arm measurements pending**.
The A4 owner run showed six `multi_patch` cases returning `Converged`
from the recursively updated PCG residual while the original CSR true
relative residual remained 3.42e-10 to 7.16e-10, above tol=1e-10.
See `docs/C1H12_A4_OWNER_NUMERICAL_EVIDENCE.md`.

## Hypothesis and scope

PCG updates an inexpensive recursive residual `r <- r - alpha A p`.
At stringent tolerances this can drift from independently recomputed
`b - A*x`. A5 tests a **wrapper only in `fem_coarse_crosscheck`**, enabled
via `--certified-restart`. Without that flag, A4 and production PCG paths
are unchanged. No PETSc/hypre dependence, no new public solver/ABI policy.

Each segment calls the unchanged native `hybit::pcg` with the same `A`,
preconditioner `M`, RHS, and current `x`. On every segment return, A5
recomputes `||b-A*x|| / ||b||` on the original CSR. A recursive
`Converged` claim is accepted **only** if its true relative residual is
at/below the target. Otherwise A5 rejects that claim, and (if budget
remains and iterations advanced) makes a new PCG solve from the current
solution; this resets conjugacy with the newly recomputed initial
residual. Every subsequent segment deducts its iterations from the
**same global 600-iteration budget**.

If the budget is exhausted, the status is `MaxIterations`. A zero-step
false convergence instead becomes `Breakdown` to avoid an infinite loop.
Numerical exceptions remain visible as process failures. This is a
targeted diagnostic for reliable residual correction, **not** a proof of
robustness or a general AMG V-cycle implementation. True-residual
certification may not achieve tol=1e-10 when floating-point residual
evaluation reaches a practical floor.

`C1H12A5|` logs record `true_checks`, `restarts`,
`rejected_recursive_convergences`, final original-CSR relative residual,
total iterations and status. A4's existing `C1H12A4|` row remains
backward compatible for the numerical gate, now reflecting the opt-in
certified status and total segment iterations.

## WSL owner acceptance

Apply the incremental A5 patch over **A2+A3+A4 uncommitted** on
`develop/0.10.0` at HEAD `4593c20`. A5 includes exact A4 evidence
transcribed from the owner; the owner's existing raw A4 folder is not
modified. Then:

```bash
cd ~/hybit
BASE="$HOME/c1h11_evidence/c1h12_a4_native_OPAkkT"
bash benchmarks/scripts/C1H12_A5_certified_pcg.sh --dry-run "$BASE"
bash benchmarks/scripts/C1H12_A5_certified_pcg.sh --run "$BASE"
```

Run 3 matrices × 7 identical arms × one process repeat with the same
known `b=A*ones`, `x=0`, tol 1e-10, 600 iterations, 512 coarse target,
F64 transfer, one CPU thread. Before running, check the old matrix
SHA-256 manifest against disk and ensure the new run matches that
manifest **byte for byte**. Preserve all 21 `.log`/`.time` files,
`HEAD.txt`, hashes, `native_coarse_summary.csv`, and the additional
`certified_restart_comparison.csv` under a new
`~/c1h11_evidence/c1h12_a5_certified_XXXXXX/` directory.

The A4 gate checks strict independently verified convergence and the
integrity of the entire 21-arm run. The A5 gate additionally checks
old/new provenance, completion, iteration-count/status agreement and
certification metadata for each pair. A correctly reported
`MaxIterations` or `Breakdown` remains useful negative evidence.

## Decision after owner measurements

For the six `multi_patch` arms, report rejected recursive convergences,
extra iterations, restart count, final true residual, solution error,
setup and solve timing. Do not relax the `1e-10` gate to make a pilot pass.
Compare both the true-residual rescue rate and added operator work.
If no arm passes, investigate floating-point attainable tolerance,
residual replacement cadence, compensated/reproducible SpMV and better
native multilevel coarse correction in separate controlled experiments.
Evaluate PETSc only with verified identical matrix/RHS/tolerance.
