# C1-H6 — Experimental weighted ABTM routing in HyBIT Auto

**Base:** `d012e5190faf4ec242866a8f5d19591fc587e8e5` on `develop/0.10.0`.
**Status:** candidate patch — actual WSL2 compiler, Clippy, workspace and corpus gates NOT YET RUN.

## Scope and contract

This adds an explicitly enabled, **FEM-host-neutral** experiment to the
ordinary SPD `HybitSolver` / `HybitPreparedSystem` controller. It does not
change any default solver path or the C ABI. `GeneralSquare` / contact / MINRES
are outside this experiment. FrontISTR is an intended future FEM host, not a
HyBIT core dependency.

- New `HybridOptions::abtm_weighted_routing_enabled` defaults to `false`.
- After the existing 12-iteration Jacobi PCG probe, recompute original CSR
  residual and select its highest Jacobi-scaled residual DOF.
- Grow one ABTM-topology connected component through edges whose normalized
  numeric coupling `|A_ij|/sqrt(A_ii A_jj)` is >= 0.1, up to
  `max_local_region_size` DOFs.
- Only admit a candidate if it exactly fills that bounded region, has **zero**
  strong edges and at least one positive weak edge crossing its boundary, and
  its conservative dense byte bound `8 * region_size^2` fits the configured
  local factor budget. Sorted, unique CSR columns and finite positive
  diagonal/finite residual are required; otherwise keep PCG.
- When rejected: preserve the *original* PCG session, do not prepare local
  factors, do not restart, and do not fall through to the old residual-only
  escalation policy.
- When admitted: reuse HyBIT's factor-budget selection and
  `HybridPreconditioner::from_csr32`; restart PCG **only** after changing the
  SPD preconditioner; true-residual certification is unchanged.
- Restrict this experiment to a single local region and one escalation, with
  zero overlap layers, no algebraic coarse base, and no progress monitor.
  Validation fails on incompatible options rather than silently changing
  semantics.
- In this experimental mode, **disable persistent local factor reuse across
  later RHS calls**. Each RHS must be diagnosed separately. Existing default
  prepared solve-many caching remains unchanged.

The `0.1` edge threshold and exact-size/weak-boundary criteria are **pilot
heuristics** from the 160 DOF C1-H5 corpus. They are not a general stiffness
classifier, SPD proof, convergence guarantee or verified optimal policy.
Numeric coupling alone is insufficient to identify a global low-energy mode.

## Validation

```bash
cd ~/hybit
bash benchmarks/scripts/C1H6_run_auto_route.sh
cargo +1.99.1 test --release --workspace --locked
cargo +1.99.1 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.99.1 fmt --all -- --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
```

`C1H6_run_auto_route.sh` generates the same two controlled SPD problem
families as C1-H5 and executes the **real HyBIT Auto code**, checking:

1. `localized_chain`: Jacobi MaxIterations, opt-in Auto Converged, one local
   escalation, true relative residual <= 1e-10; auto unit tests compare the
   known-reference forward error and the exact 14-iteration result.
2. `global_chain`: no local factorization; Auto recurrence, status, iterations
   and final solution bits match uninterrupted Jacobi-PCG in the unit test.
3. All five C1-G P1 FEM pilot matrices: execute the identical C1-H6 routing
   option and fail if the previously convergent case triggers any local factor
   escalation. C1-H3 offline shadow logs are **not** a substitute for this.

The C1-G gate is a *strict holdout smoke test* for the experimental heuristic;
it may fail and should drive revision, not be bypassed or described as a success.
Record factor memory, actual wall, residual and forward error for every case.

## Limitations before claiming any practical success

The 160 DOF synthetic matrix is not a realistic real-FEM workload, and its
Cholesky setup is much slower than the entire plain PCG run. Do not claim a
wall-clock speedup. The new `HybridOptions` public field alters Rust struct
literal construction compatibility; the default value does not make this
source-compatible for callers who construct the full struct explicitly.
Hold the patch uncommitted until all workspace tests, Clippy and C1-G checks
pass on WSL2. Before actual production routing, validate on independent
external FEM systems with known SPD class and, where possible, reference x.
