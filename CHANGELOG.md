# Changelog

## 0.6.0-r25 development snapshot

- Promote the validated parallel/fused PCG dense-vector path into Structural Auto through `StructuralPcgVectorPolicy::{Auto, Serial, Parallel}`.
- `Auto` enables parallel/fused PCG vector kernels only when the structural system has at least 131072 unknowns and the shared Rayon pool has at least 4 workers; small systems and 1-2 worker pools remain on the established serial vector recurrence.
- Keep the sparse operator and rigid-body preconditioner policies independent from the PCG-vector policy, so SpMV, preconditioner, and vector-kernel A/B tests remain isolated.
- Add `bench-fem-structural-auto-pcg.ps1` for an end-to-end production-path comparison between serial and Auto PCG vector kernels with parallel SpMV/preconditioning fixed.
- Preserve `solve_csr32` behavior; only the explicit structural solve path can select the new vector policy.
- Validated the integrated r25 Structural Auto path on the physical-load L-angle system (358065 DOF, 28239653 CSR nnz) at 8 Rayon workers: Graph aggregation, coarse dimension 1398, 220 PCG iterations, verified relative residual `9.378557e-9`, 1.726 s solve, and 2.797 s analysis+prepare+solve.
- Enter feature freeze for the 0.6.0 structural production path on `develop/0.6.0`; the next milestone is release-candidate regression, packaging, ABI, documentation, and publication-gate validation rather than additional solver features.

## 0.6.0-r24 development snapshot

- Add an experimental `pcg_with_workspace_parallel_vectors` path using the shared Rayon pool for large dense Krylov vector kernels.
- Fuse `x += alpha*p`, `r -= alpha*Ap`, and the residual-norm reduction into one parallel traversal, removing one full residual-vector pass per PCG iteration.
- Parallelize PCG dot products, initial residual construction, and search-direction updates without changing the sparse operator or rigid-body preconditioner.
- Add `fem_structural_pcg_parallel` / `bench-fem-structural-pcg-parallel.ps1` for thread-count A/B against the production PCG vector path.
- Keep Production Structural Auto unchanged pending real FEM numerical-equivalence and wall-time results.

## 0.6.0-r23 development snapshot

- Add an exact nested PCG stage profiler for the structural parallel path (`fem_structural_pcg_profile` / `bench-fem-structural-pcg-profile.ps1`).
- Separate actual operator, preconditioner, dot/norm reductions, fused x/r update, and search-direction update costs before changing Krylov kernels.
- Keep the Production PCG recurrence unchanged in the profiled control path.

## 0.6.0-r22 development snapshot

- Integrate the validated parallel rigid-body preconditioner into Structural Auto through `StructuralPreconditionerPolicy::{Auto, Serial, Parallel}`.
- Large structural CSR systems may automatically use Parallel CSR SpMV plus parallel block-Jacobi/restriction/prolongation; small systems remain serial.
- Preserve the packed coarse triangular solve as serial and cache the aggregate-to-node parallel index for prepared solve-many reuse.
- Keep Rayon thread-count selection external rather than hard-coding the development machine's optimum.

## 0.6.0-r21 development snapshot

- Add an experimental `ParallelRigidBodyTwoLevelPreconditioner` view that reuses the already prepared Graph/Additive rigid-body coarse factor without duplicating Cholesky storage.
- Parallelize the independent 3x3 block-Jacobi solves, aggregate-local rigid-body restriction, and node-local prolongation with Rayon; keep the packed dense coarse triangular solve serial.
- Precompute an aggregate-to-node index for conflict-free parallel restriction; report its additional storage separately from the existing preconditioner factor bytes.
- Add `fem_structural_precond_parallel` and `bench-fem-structural-precond-parallel.ps1` to compare serial and parallel preconditioners under the same `ParallelCsr32Operator` and sweep the shared Rayon thread pool.
- Keep the production Structural Auto preconditioner policy unchanged until the L-angle wall-time A/B establishes that the extra Rayon work is beneficial.

## 0.6.0-r20 development snapshot

- Integrate the validated Rayon-parallel CSR operator into the Structural Auto/prepared solve path through `StructuralSpmvPolicy::{Auto, Serial, Parallel}`.
- `StructuralSpmvPolicy::Auto` keeps small matrices serial and selects parallel CSR for CSR32 systems with at least 1,000,000 nonzeros; generic `solve_csr32` remains unchanged.
- Preserve external Rayon thread control rather than hard-coding the L-angle development machine's 4-thread optimum; PowerShell structural wrappers accept `-RayonThreads`.
- Expose the effective prepared SpMV policy and add regression coverage for small-matrix Auto-serial behavior and explicit parallel execution.
- Add `bench-fem-structural-auto-spmv.ps1` for an end-to-end serial-vs-Structural-Auto comparison using the public structural API.

## 0.6.0-r19 development snapshot

- Add an explicit `ParallelCsr32Operator` using Rayon row parallelism for controlled CSR SpMV experiments; existing `Csr32Matrix` execution policy remains serial.
- Use the validated/private CSR invariants to remove repeated hot-loop bounds checks in both serial and parallel SpMV kernels.
- Add `fem_structural_spmv` and `bench-fem-structural-spmv.ps1` to sweep Rayon thread counts and compare SpMV microbenchmarks plus full structural PCG wall time.
- Keep Graph rigid-body Structural Auto, target coarse dimension 1536, additive two-level correction, and packed coarse Cholesky unchanged while measuring the dominant fine-grid kernel.

## 0.6.0-r18 development snapshot

- Add a zero-overhead-by-default structural kernel profiler for the rigid-body two-level path.
- Report CSR SpMV, 3x3 block-Jacobi, coarse restriction, packed coarse solve, and prolongation timings separately.
- Add `bench-fem-structural-profile.ps1` for 1536/3072 coarse-size bottleneck diagnosis.
- Keep the packed coarse Cholesky introduced in r17; no production solve algorithm changes in this snapshot.

All notable changes to HyBIT are documented here.

## 0.6.0 (development)

- Added Matrix Market coordinate import for real/integer `general` and `symmetric` matrices.
- Added duplicate-entry coalescing and symmetric expansion into CSR32.
- Added Matrix Market general-coordinate export for assembled CSR32 matrices.
- Added `fem_bench`, a real-matrix benchmark path comparing plain Jacobi-PCG with HyBIT Auto under the same tolerance and initial guess.
- Added independent `||Ax-b||/||b||` verification and optional known-solution `b=A*1` generation.
- Added real-matrix reporting for matrix storage, iteration counts, wall time, adaptive regions, local-factor memory, and Krylov workspace memory.
- Added benchmark documentation and a bundled Matrix Market smoke matrix.
- Added contiguous SPD block-Jacobi preconditioning with dense Cholesky factors and allocation-free application.
- Added `fem_block_bench` and `bench-fem-block.ps1` to measure vector-FEM block preconditioning independently from the current Auto policy.
- Added an SPD additive two-level preconditioner combining contiguous block Jacobi with a Galerkin aggregation coarse correction.
- Added `fem_twolevel_bench` / `bench-fem-twolevel.ps1` to diagnose global low-frequency behavior on real structural FEM matrices before integrating coarse correction into Auto policy.
- Added geometry-aware six-mode rigid-body aggregation (Tx/Ty/Tz/Rx/Ry/Rz) with RMS-radius normalization for 3-D structural coarse correction.
- Added `fem_rigid_bench` / `bench-fem-rigid.ps1` and coordinate-sidecar support for exact reduced/RCM node ordering.
- Extended the rigid-body FEM benchmark with optional physical RHS input so synthetic `b=A*1` and application loads can use the identical matrix/preconditioner path.
- Added an experimental Structural Auto policy that selects a six-mode rigid-body aggregate size from a bounded target coarse dimension; the 119355-node L-angle case selects 512 nodes/aggregate and a 1404-dimensional coarse space.
- Integrated the validated Structural Auto path into the Rust solver API with `StructuralOptions`, `solve_structural_csr32`, and reusable `HybitPreparedStructuralSystem`; generic `solve_csr32` behavior remains unchanged.
- Added prepared reuse of the geometry-aware rigid-body coarse factor and PCG workspace across multiple RHS vectors.
- Added graph-connected rigid-body aggregation inferred from the structural CSR sparsity, with deterministic BFS regions and tail merging; contiguous RCM aggregation remains available for controlled A/B comparison.
- Added `bench-fem-structural-aggregation.ps1` to run contiguous-vs-graph Structural Auto comparisons under identical solver settings.
- Hardened graph aggregation by merging small BFS remainder islands (below one quarter of the requested aggregate size) into their strongest adjacent aggregate before constructing six-mode rigid-body coarse spaces.
- Added a prepared structural solve-many benchmark that resets the initial guess for every solve and reports one-time setup separately from reusable solve cost.
- Promoted graph-connected structural aggregation into the default `RigidBodyAggregation::Auto` policy after the L-angle physical-load benchmark reduced PCG from 1370 iterations (contiguous, coarse dimension 1404) to 220 iterations (graph, coarse dimension 1398); explicit `Graph` and `Contiguous` modes remain available for controlled benchmarks.
- Added Structural Auto fallback from graph aggregation to contiguous aggregation when graph coarse Cholesky reports numerical breakdown or graph topology contains a disconnected component too small for a six-mode rigid-body aggregate; explicit Graph remains strict and no diagonal regularization is used.
- Added regression coverage separating disconnected-identity fallback from connected-graph selection so Structural Auto remains correct on valid SPD matrices with no inter-node graph edges.
- Moved strict-Graph rejection coverage into the Rust integration suite so the build harness no longer treats an intentionally failing CLI smoke test as a build failure.
- Added `PreconditionerKind::RigidBodyTwoLevel` and corrected the repository-only C ABI version query to report 0.6.0.
- Added `bench-fem-structural-coarse-sweep.ps1` to measure Graph coarse-space size against setup, iteration count, solve time, and memory; the L-angle physical-load sweep identifies target coarse dimension 1536 as the best tested single-RHS wall-time point.
- Added an experimental symmetric balanced rigid-body two-level preconditioner and `bench-fem-structural-balanced.ps1` for controlled additive-vs-balanced A/B testing without changing the Structural Auto default.
- Kept additive rigid-body two-level as the Structural Auto default after the L-angle balanced experiment reduced iterations only from 220 to 196 while increasing solve time from about 5.37 s to 12.45 s because of two extra sparse matrix-vector products per preconditioner application.
- Changed the structural rigid-body coarse Cholesky factor from full `n x n` lower storage to packed lower-triangular storage with precomputed row offsets, halving coarse-factor memory while preserving the same Galerkin operator and solve semantics.

## 0.5.0

First public release candidate for GitHub and crates.io.

- Preserved the numerical and ABI implementation validated by the 0.4.1 release gate.
- Added crates.io-ready package metadata and versioned internal path dependencies.
- Added the public repository URL `https://github.com/michioga/hybit`.
- Added English and Japanese public READMEs.
- Added publication, contribution, roadmap, and release documentation.
- Added GitHub Actions CI for Rust workspace tests on Linux and Windows.
- Kept `hybit-ffi` repository-only while publishing the Rust facade and its internal Rust dependencies to crates.io.
- Removed release-candidate wording and version-specific validation text from runtime errors.

## 0.4.1

- Hardened MinGW external-language examples against PATH-dependent GNU runtime DLL mismatches.
- C++ example links libstdc++ and libgcc statically while keeping `hybit.dll` as the public ABI boundary.
- Fortran example links libgfortran and libgcc statically while keeping `hybit.dll` as the public ABI boundary.
- Release gate prints PE DLL imports when `objdump` is available.
- Added reusable `analyze -> prepare -> solve-many` execution through `HybitAnalysis` and `HybitPreparedSystem`.
- Added reusable `PcgWorkspace` and allocation-free prepared PCG work vectors.
- Made weighted local Schwarz application allocation-free with preallocated per-region scratch.
- Added lazy Hybrid preconditioner learning and local Cholesky factor reuse across subsequent RHS vectors for an unchanged matrix.
- Added matrix structure/value signatures that reject stale prepared contexts when coefficients change.
- Added `prepare_seconds`, `preconditioner_reused`, `solve_sequence`, and `krylov_workspace_bytes` diagnostics.
- Added C ABI prepared handles: `hybit_prepare`, `hybit_solve_prepared`, and `hybit_prepared_destroy`.
- Added C++ RAII `Prepared` wrapper and matching Fortran bindings.

## 0.3.0

- Generalized selective-direct PCG to multiple ABTM-expanded local subdomains.
- Added topology overlap and symmetric `1/sqrt(multiplicity)` weighted Schwarz correction.
- Added local-factor memory and per-stage timing diagnostics.

## 0.2.0

- Added poor-progress probing and automatic selective-direct escalation for SPD problems.
- Added residual/risk hard-DOF selection, ABTM one-hop topology expansion, and local dense Cholesky.
- Restarted PCG after changing the preconditioner.

## 0.1.1

- Stabilized Windows DLL/import-library handling for MSVC Rust with MinGW C/C++/Fortran consumers.

## 0.1.0

- Initial Rust workspace, CSR32/ABTM, Jacobi-PCG, C ABI, C++ wrapper, and Fortran binding.
