# HyBIT 0.9 H1-C3K8-C — Large FEM candidate PCG (experimental)

## Baseline and scope

Baseline: `develop/0.9.0` `3daae848be7cf9172b29cff186e172eff33e6e0a` (K8-B committed). Changes to `mpi_k8b_fem_pcg.rs` are *experimental example code only*: production solver selection and the distributed PCG library are unchanged.

- Existing K8-A / K8-B 10-argument invocation stays fail-closed: K8-B's strict diagonal dominance, anchored M-matrix, or bounded <=512 dense Cholesky acceptance is unchanged.
- Only 11th argument `--experimental-large-spd` enables candidate screening for non-DD/non-M-matrix with `n > 512`.
- **Candidate screening is NOT SPD certification**. Requires finite symmetric, sorted unique CSR, positive finite diagonal and four deterministic positive Rayleigh probes; an untested negative eigenvector may remain. There is no automatic selection or fallback.
- PCG rejects nonpositive curvature `p^T A p <= 0`, nonfinite iteration values, failure to converge, and failed terminal *true residual*. Experimental path also performs a true residual after the first iteration and every 32 iterations; these checkpoints incur SpMV/collective overhead.
- Root gathers noncontiguous solution and evaluates `b-Ax` with the original (pre-scatter) CSR, independent from Halo/owner reconstruction.
- Preconditioners: existing Jacobi, `LocalBlockIc0`, `LocalBlockCholesky` with temporary local CSR bridge. Local factor SPD does not prove global SPD.
- The `experimental_symmetric_positive_diagonal_curvature_screen` label is explicit and must never be presented as an SPD certificate.

## Test matrix

`make-k8c-large-fem-spring.ps1` generates a two-component spring lattice in sparse *lower triangular* Matrix Market symmetric coordinate form. Every spring adds a positive-semidefinite rank-one outer product; the positive uniform grounding shift makes the assembled exact mathematical matrix SPD. The generator uses sparse storage, not K8-B's bounded dense array, and has independent bounds (Side 17..60); default Side 32 => 2,048 DOFs. Positive coupling / non-DD structure exercises the new screen.

## Native validation / diagnostics

1. Strict MS-MPI Clippy `--all-targets -D warnings`, 95 library unit tests, K8-B example tests (11 after patch), K8-A example regression tests, release binary, source-integrity gate.
2. Generate `k8c_fem32.mtx`, Side 32, Shift 0.5.
3. Run `bench-k8c-large-fem-pcg.ps1 -Ranks @(1,4) -Policies @('equal','region','multisource') -Preconditioners @('jacobi','ic0','block') -BlockSize 16 -Warmups 1 -Repeats 3 -MaxIters 1500 -Tolerance 1e-8`. This produces **one** `${Prefix}_summary.csv` by default; per-case CSVs are temporary. 18 combinations.
4. Require exact original CSR, experimental label/flag, at least one periodic/early true residual checkpoint, true terminal residual <= tolerance and Root-original verification.
5. Test fail-closed path by running old 10-argument K8-B benchmark for a 128-DOF model (no experimental flag). Do not merge/commit until native checks and numerical results pass.

## Caveats

- This is a *prototype candidate-testing mode* intended only for matrices whose SPD nature comes from external mathematical provenance (e.g., constrained spring energy + positive grounding). Passing probes and PCG does **not** establish SPD.
- For arbitrary non-SPD matrices: require GMRES/other appropriate path, not opportunistic PCG retry.
- MPI runtime Microsoft MS-MPI/MSVC, not MinGW or Open MPI.
- No automatic commit or push. The APPLY script requires exact committed K8-B HEAD, clean worktree, known blobs, payload digests, and stages 5 paths only.
