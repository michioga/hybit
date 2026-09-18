# HyBIT 0.5 architecture

HyBIT 0.5 keeps the staged-PCG safety rule from 0.2/0.3 and adds reusable matrix-dependent execution state.

## One-shot path

`HybitSolver::solve_csr32` is still available. Internally it now performs:

1. analyze;
2. prepare;
3. one prepared solve.

This preserves the simple API while keeping a single implementation path.

## Prepared path

1. **Analyze** CSR32 structure, SPD baseline requirements, backend policy, and exact structure/value signatures.
2. **Prepare** Jacobi and a reusable five-vector PCG workspace; ABTM is built eagerly only for an ABTM backend and otherwise lazily on hybrid escalation.
3. **Solve #1** with a short Jacobi-PCG probe.
4. If progress is poor, compute residual/numerical-risk masks and split hard DOFs into connected components.
5. Expand each component through ABTM topology, bound the local regions, and build dense Cholesky factors.
6. Build symmetric weighted Schwarz terms and restart PCG.
7. Cache the resulting Hybrid preconditioner.
8. **Solve #2..N** against the same matrix using the cached local factors and PCG workspace without repeating probe, diagnostics, or factorization.

ABTM remains internal. Applications continue to supply ordinary CSR32.

## Allocation policy

Repeated PCG execution is allocation-free with respect to HyBIT work vectors:

- five global `n`-length vectors live in `PcgWorkspace`;
- each local Schwarz factor owns fixed local RHS and solution scratch vectors;
- local `apply()` performs no `Vec` allocation.

The first adaptive diagnostic/factor-construction pass may still allocate masks and region bookkeeping outside the Krylov inner loop.

## Reuse safety

HyBIT 0.5 hashes both CSR structure and exact floating-point coefficient bits. A prepared context rejects changed matrix structure or values. This is intentionally conservative for the first solve-many implementation.
