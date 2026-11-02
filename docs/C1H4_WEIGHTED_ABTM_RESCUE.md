# C1-H4 — weighted ABTM local-direct rescue positive/negative control

**Experimental checkpoint for HyBIT 0.10.0.** No change to production `HybitSolver` defaults, local Hybrid implementation, PCG implementation, ABI or MPI. FEM-host neutral; synthetic SPD algebraic controls are **not real FrontISTR FEM measurements**.

## Purpose and limitations

C1-H3 shadow stall guard rejected all hypothetical promotions in the five easy-to-moderate SPD-FEM corpus cases (15 threshold variants, 87 checkpoints). This is negative evidence; it is **not** proof that local direct correction can rescue a baseline that fails under the same budget.

C1-H4 adds a fixed, independently executable Rust example that compares:

1. Jacobi-PCG from zero initial guess for **30 global Krylov iterations**.
2. Jacobi-PCG from zero for **12 probe iterations**, followed by **one** ABTM-derived, weighted-graph-local Cholesky preconditioner and **at most 18** more PCG iterations. Restart on the changed preconditioner is mathematically necessary and budgeted.

Both arms use identical SPD matrix, RHS, reference solution, tolerance `1e-10`, precision, and 30-iteration cap. The *plain* and *two-stage* arms deliberately do not have equal setup/runtime costs: the hybrid additionally builds ABTM and factors one local block. Wall times are separated and must not be interpreted as a speedup without accounting for setup and repeated trials. Persistent factor bytes are reported.

## Synthetic controls and exact SPD contract

Dimension **160**. Every diagonal starts at 2; a strong chain has offdiagonal -1; Dirichlet tridiagonal chain has strictly positive eigenvalues. On the localized arm, only DOFs `[0..96)` form the 96-DOF strong chain, DOFs 96..159 are diagonal-only (diagonal 2), and a single weak edge of 1e-4 connects DOFs 95 and 96. The weak edge adds +1e-4 to **both endpoint diagonals**, i.e. it is a positive semidefinite edge Laplacian added to an SPD block-diagonal matrix. Hence the resulting matrix is SPD by construction.

In the global control arm, all 160 DOFs are joined by strong -1 edges with tridiagonal Dirichlet diagonals 2. It is SPD, but a local 96-DOF factor cannot cover the entire chain.

Both RHS are independently generated as `b=A*xref`, using a nonconstant, deterministic reference vector. The experiment reports original-CSR true residual and forward error for both arms. The local case is pre-registered to pass only if plain-PCG exhausts 30 iterations above tolerance and 12+18-iteration Hybrid-PCG converges with true residual <=1e-10 **and** forward error <=1e-6. It also requires the selected ABTM region equal `[0..96)`, zero strong boundary edges and exactly one weak boundary edge.

In the global arm the same information is printed, but no hard convergence or nonconvergence outcome is asserted. This is a *negative control for the locality assumption*, not a proof that globally hard modes are incurable by local direct methods. The optional coarse-space and KrylovScope comparisons remain future work.

## Weighted ABTM topology selection

- Build the **structure-only** `AbtmTopology` once; it retains explicitly stored zeros.
- At probe iteration 12, form true residual `r = b - Ax`, then select the DOF maximizing `|r_i|/sqrt(A_ii)`.
- Expand a connected local region through **ABTM word masks**, but accept neighbor `j` only if the canonical CSR coefficient satisfies:

  `|A_ij| / sqrt(A_ii A_jj) >= 0.1`.

- Region cardinality capped at 96 DOFs. Construct the existing `HybridPreconditioner::from_csr32(A,[region])`: this uses local packed Cholesky and Jacobi on uncovered DOFs.
- The weighted graph is **an experiment-specific rule for the constructed SPD matrix**, not a generally valid spectral predictor. In a nonsymmetric or inconsistent CSR matrix, the binary-search assumption and undirected interpretation would need additional canonicalization and validation. Do not apply this SPD PCG path to general indefinite or contact systems.

For the constructed local family the normalized weak edge is approximately 5e-5, well below the 0.1 threshold; strong interior chain edges are approximately 0.5. The connectivity-based region should therefore cut the weak bridge and select the 96-DOF local chain. A corresponding pure topology-only flood would cross the weak edge.

## Running and interpreting

```bash
cd ~/hybit
bash benchmarks/scripts/C1H4_run_positive.sh
```

The script requires `local rescue gate  : PASS` and verifies the source manifest. The Rust example's `#[test]` cases independently verify the numeric cut, the strong-only global chain and symmetric/known-solution construction.

**Passing this gate proves a *manually orchestrated* local preconditioner rescue under a specified iteration budget**, not that the existing automatic `HybitSolver` chooses that region or that an ABTM policy generalizes across FEM codes. A follow-up experiment must compare the existing Auto route and a topology-aware, opt-in candidate selector, then test held-out FEM problems with independent truth/reference and memory budgets.

## Acceptance gates

- Original-CSR true relative residual and known forward error both PASS for localized rescued arm.
- Baseline Jacobi-PCG fails under same 30-iteration cap.
- Local factor nonzero, factor bytes disclosed, total Krylov iterations counted as probe + stage (not just stage).
- No numerical global-case claims based solely on a test's exit status.
- Workspace Release tests, strict Clippy, rustfmt and source SHA checks PASS.
- Source-control checkpoint only after user WSL execution and review. No release or publish.
