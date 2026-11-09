# C1-H12-A10B: prepared ABTM-backed multilevel AMG reuse

**Status:** isolated research probe, pending owner WSL pilot; not production evidence.

## Why

The A8 native multilevel hierarchy uses ABTM word masks for symbolic
Galerkin candidate support and numeric CSR accumulation for values.
A10A exposed large forward errors at a true-residual tolerance of
1e-10 on weak modes. This trial uses 1e-12 for all runs.

Question: how much hierarchy setup cost can be amortized over a batch
of distinct RHS solves on the same fixed matrix?

An A8-prepared MultiLevelAmg already retains its ABTM-derived hierarchy,
numeric coarse matrices, bottom Cholesky factor and V-cycle scratch.
Both lifecycle arms use exactly the same ABTM Galerkin kernel. Thus this
trial does NOT isolate an ABTM-only speedup.

## Design

* Input: three n=20736 original FEM matrices, four deterministic A9
  RHS modes (ones, index_smooth, seeded_random, point_load).
* Aggregation/basis: Graph/Piecewise, StrongGraph/Piecewise,
  StrongGraph/JacobiSmoothed.
* Preconditioners: additive two-level control and native A8 multilevel V-cycle.
* Lifecycle: fresh builds a preconditioner for every RHS; reuse builds
  one instance before all repeated RHS solves.
* Five measured rounds of four RHS after one unmeasured warmup round.
  Each solve starts with x=0. Reuse setup time is charged exactly once
  in the first measured sample.
* 1200 maximum PCG iterations, tolerance 1e-12, single thread.
  Independent original-CSR and Kahan residual, plus manufactured forward
  error for the three exact-solution RHS modes.
* 3 x 3 x 2 x 2 = 36 processes, 20 measured samples each:
  **720 measured samples** and 144 warmup solves.

## Measurement interpretation

measured_total_ms = setup_ms_total + solve_ms_total. This excludes matrix
load, external process startup, warmup, diagnostic recomputation and logs.
/usr/bin/time records whole-process elapsed time and RSS including all of
the above. Do not present these timing definitions as interchangeable.

The gate requires 720 complete samples and validates matched fresh/reuse
iterations, original A9 RHS fingerprints, frozen A9/A10A matrix checksums
and owner HEAD, numerical status/work, hierarchy and ABTM accounting,
and timing sums. Converged implies original-CSR residual <= 1e-12.
Kahan checks are diagnostic, not a forward-error proof.

No assertion of speed improvement is encoded: slowdowns and lack of
convergence remain visible.

## Boundaries

This trial reuses a fixed matrix. Changing numeric coefficients requires
refreshing Galerkin values and factors; changing strength-graph membership
or sparse topology may require symbolic reconstruction.

A10B2 can separately benchmark cached ABTM scatter maps versus CSR
symbolic methods for coefficient updates with invariant sparsity.
A10C will inspect ABTM strong-graph components and independently
computed small Ritz modes; masks alone cannot determine eigenvalues.

Production PCG, public API, and solver defaults remain unchanged.
