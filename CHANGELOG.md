## 0.7.0-r30 - algebraic coarse cross-validation harness

- Added `fem_coarse_crosscheck`, a benchmark-only example that constructs the generic Graph + JacobiSmoothed + Parallel + Wide/F32 algebraic two-level preconditioner directly, independent of selective-direct escalation.
- Added direct-coarse and Jacobi-probe-then-coarse modes so cross-matrix tests can separate coarse-space effectiveness from the current escalation controller.
- Added `bench-fem-coarse-crosscheck.ps1`.
- No production solver/API/default changes from r29.

### 0.7.0-r29

- Add `bench-fem-hybrid-initial-regions.ps1` as a benchmark-only repeated comparison of first-stage `max_local_regions` 1, 2, 4, 8 and 16 with `max_escalations = 1`.
- Keep the validated Graph + JacobiSmoothed + Parallel + Wide/F32 + target-1792 coarse path fixed so the experiment isolates the amount of selective-direct strengthening performed before the final PCG continuation.
- r28 showed that one escalation (8 regions) was fastest: median 870 iterations and 20.711 s wall, 5.94% faster than depth 3; additional preconditioner changes restarted PCG and did not recover their restart/setup cost.
- Include 16 as a saturation check because the current 1024-DOF hard core and 128-DOF region cap may limit the first harvest to eight full-size regions.
- No Rust solver, preconditioner, API, or default-policy changes from r27/r28.

### 0.7.0-r28

- Add `bench-fem-hybrid-escalation-depth.ps1` as a benchmark-only repeated comparison of `max_escalations` 1, 2, 3 and 4 on the validated Graph + JacobiSmoothed + Parallel + Wide/F32 + target-1792 generic FEM path.
- Keep all Rust solver, preconditioner, API and default-policy code unchanged from r27 so the experiment isolates selective-direct escalation depth.
- Interpret depth 1 as one 8-region strengthening followed by the remaining Krylov budget, depth 2 as the existing 8→16-region progression with the second stage final, depth 3 as the current production experiment baseline, and depth 4 as a probe of whether a short third stage justifies one further strengthening.
- Report actual escalations, local-region count, factor memory, diagnostics/factor setup cost, iterations, milliseconds per iteration, solver time and wall time, with depth 3 as the relative baseline.
- r27 validation passed `cargo fmt`, `hybit-precond` 22/22, `hybit-auto` 27/27, the full workspace suite and Clippy `-D warnings`; an option-omitting FEM run resolved the promoted smoothed transfer defaults to Parallel/Wide/Auto→F32 and converged in 876 iterations with 59.71 MiB coarse memory.
- r26 established Wide/F32 as the measured speed/memory Pareto default and Compact/F32 as the low-memory alternative; r28 therefore moves optimization attention from coarse transfer representation to the selective-direct controller.

### 0.7.0-r27

- Promote the high-level generic algebraic coarse transfer defaults used by `AlgebraicCoarseOptions` to `Parallel` application, `Wide` indices and `Auto` value storage after r26 established Wide/F32 as the best speed/memory Pareto point on the repeated L-angle benchmark.
- Keep the low-level `TwoLevelTransferOptions::default()` compatibility path unchanged at Serial/Wide/F64; only the generic auto-layer defaults and `fem_bench` defaults are promoted.
- `Auto` transfer-value storage resolves to F32 whenever all smoothed-transfer weights remain finite after quantization, otherwise falling back to F64; Wide indices remain the speed-oriented default.
- Keep aggregation and basis defaults unchanged (`Contiguous` + `PiecewiseConstant`), so the promoted transfer defaults take effect only when a caller explicitly selects `JacobiSmoothed`.
- r26 four-way result at Graph + JacobiSmoothed + Parallel + target 1792: all layouts converged in 876 iterations; Wide/F32 was fastest at median wall 21.808 s with 59.71 MiB coarse memory, while Compact/F32 minimized coarse memory at 52.52 MiB with only 0.63% slower median wall than Wide/F32.
- Wide/F64 and Compact/F64 are dominated by Wide/F32 in the measured speed/memory plane; retain both explicit policies for reproducibility and compatibility rather than using them as generic smoothed defaults.

### 0.7.0-r26

- Add `bench-fem-hybrid-coarse-transfer-layout.ps1` as a benchmark-only four-way comparison of Wide/F64, Compact/F64, Wide/F32 and Compact/F32 Jacobi-smoothed parallel transfer storage at fixed Graph aggregation and coarse target 1792.
- Keep all Rust solver, preconditioner, API and default-policy code unchanged from r25 so the experiment isolates persistent transfer layout only.
- Rotate/reverse run order across repeats and report median convergence, iterations, coarse memory, setup, milliseconds per iteration, solver time and wall time, plus deltas relative to Wide/F64.
- r25 result: F32 transfer values preserved 876 iterations and reduced coarse memory from 71.37 to 59.71 MiB (11.66 MiB) with only 0.59%/0.38% median solver/wall slowdown versus F64.
- r24 result: Compact indices preserved 876 iterations and reduced coarse memory by 7.20 MiB but slowed median solver/wall by 1.33%/1.03%; r26 measures whether combining Compact indices with F32 values gives a useful low-memory Pareto point.

### 0.7.0-r25

- Add experimental `TwoLevelTransferValueStoragePolicy::{F64, F32, Auto}` for Jacobi-smoothed generic coarse transfers while preserving `F64` as the default.
- Store F32 transfer weights persistently as `f32` but promote each weight to `f64` during restriction/prolongation; keep matrix, coarse solve, Krylov vectors and local factors in `f64`.
- Build the Galerkin coarse operator `P^T A P` from the same quantized transfer weights used during PCG when F32 storage is selected, preserving a symmetric coarse correction rather than mixing F64 setup with F32 application.
- Keep transfer-index storage independently selectable; the r25 benchmark fixes Wide indices so the F64/F32 A/B isolates transfer-weight bandwidth and storage effects.
- Include transfer-value bytes in persistent coarse-memory accounting and expose effective value-storage policy/value-byte telemetry on `TwoLevelBlockJacobiPreconditioner`.
- Extend `AlgebraicCoarseOptions` and `fem_bench` with `--coarse-transfer-values f64|f32|auto`; keep F64 as the generic default pending real-FEM validation.
- Add regression coverage checking positive F32-smoothed action, F32/F64 closeness, halved transfer-value storage, and Auto resolution to F32 for representable weights.
- Add `bench-fem-hybrid-coarse-transfer-values.ps1` for repeated F64-vs-F32 A/B runs with Graph aggregation, Jacobi-smoothed basis, Parallel transfer, Wide indices, target 1792 and coarse-apply `Auto` fixed.
- r24 result: Compact indices preserved 876 iterations and reduced coarse memory from 71.37 to 64.17 MiB (7.20 MiB), but median solver/wall time worsened by 1.33%/1.03%; retain Wide indices as the speed-oriented default.

### 0.7.0-r24

- Add experimental `TwoLevelTransferStoragePolicy::{Wide, Compact, Auto}` for the Jacobi-smoothed generic coarse transfer while preserving `Wide` as the default.
- Keep all transfer weights in `f64`; `Compact` changes only sparse-transfer indices, storing row offsets as `u32` and coarse columns as `u16` when representable.
- Preserve the historical wide constructor/API path and add an explicit storage-policy constructor; `Auto` selects compact indices when the constructed transfer fits and otherwise falls back to wide storage.
- Include the effective compact index footprint in persistent coarse-memory accounting and expose transfer-storage/index-byte telemetry on `TwoLevelBlockJacobiPreconditioner`.
- Extend `AlgebraicCoarseOptions` and `fem_bench` with `--coarse-transfer-storage wide|compact|auto`; keep `Wide` as the generic default pending real-FEM A/B validation.
- Add regression coverage proving wide/compact smoothed transfers produce equivalent preconditioner action while compact storage uses fewer persistent index bytes.
- Add `bench-fem-hybrid-coarse-transfer-storage.ps1` for repeated Wide vs Compact A/B runs with Graph aggregation, Jacobi-smoothed basis, Parallel transfer, target 1792 and coarse-apply `Auto` fixed.
- r23 result: Graph + JacobiSmoothed + Parallel remained fastest at target 1792 / actual 1785 with 876 iterations, median solver 17.319 s and wall 21.755 s; target 2048 reduced iterations to 846 but did not recover its additional setup/apply cost and raised coarse memory to about 80.15 MiB.

### 0.7.0-r23

- Add `bench-fem-hybrid-smoothed-coarse-sweep.ps1` to re-optimize generic coarse dimension after r22 made Jacobi-smoothed transfer application competitive in wall time.
- Fix Graph aggregation, Jacobi-smoothed basis and Parallel transfer while sweeping targets 512, 768, 1024, 1152, 1280, 1536, 1792 and 2048 with three alternating-order repeats.
- Report actual coarse dimension, effective coarse-apply policy, iterations, escalations, local regions, factor/coarse memory, coarse setup, milliseconds per iteration, solver time and wall time; identify the fastest all-converged target by median wall time.
- r22 result: Parallel smoothed transfer preserved 876 iterations while reducing median milliseconds/iteration from 24.630 to 19.609, solver time by 20.39% and wall time by 17.05% versus serial transfer.
- Relative to the r21a piecewise Graph baseline at target 1792, r22 smoothed+parallel reduced median wall time from 23.770 s to 21.545 s (about 9.36%) while increasing coarse memory from 33.44 MiB to 71.37 MiB; r23 therefore targets the memory/setup tradeoff without changing Rust numerics.
- Make no Rust, solver, preconditioner, public API or default-policy changes from r22; this checkpoint is benchmark-only.

### 0.7.0-r23

- Add `bench-fem-hybrid-smoothed-coarse-sweep.ps1` to re-optimize coarse dimension after r22 made the Jacobi-smoothed Graph coarse basis faster than the piecewise-constant Graph baseline by parallelizing sparse transfer application.
- Sweep targets 512, 768, 1024, 1152, 1280, 1536, 1792, and 2048 with Graph aggregation, Jacobi-smoothed basis, parallel transfer, and coarse-apply `Auto`, using three alternating-order repeats.
- Report actual coarse dimension, effective apply policy, convergence, escalation/local-region state, coarse memory/setup, solver milliseconds per iteration, solver time, wall time, and wall-time range; select the best all-converged target by median wall time.
- Keep r22 Rust code, solver/preconditioner behavior, public API, and defaults unchanged; this checkpoint is benchmark-only.
- r22 result: parallel smoothed transfer preserved the 876-iteration solve while reducing median solver time by 20.39% and wall time by 17.05% versus serial transfer; median wall reached 21.545 s, beating the piecewise Graph baseline (~23.77 s), at a coarse-memory cost of ~71.37 MiB.

### 0.7.0-r22

- Add an experimental `TwoLevelTransferApplyPolicy::{Serial, Parallel}` for the Jacobi-smoothed generic coarse basis while preserving serial transfer application as the default.
- Parallelize smoothed restriction with one persistent coarse accumulation buffer per Rayon worker, followed by a deterministic worker-order reduction; parallelize smoothed prolongation across independent fine rows.
- Keep the smoothed transfer, damping factor, Galerkin operator `P^T A P`, coarse solve and selective-direct logic numerically unchanged.
- Include persistent per-worker restriction buffers in coarse-memory accounting.
- Extend `AlgebraicCoarseOptions` and `fem_bench` with `--coarse-transfer serial|parallel`.
- Add regression coverage comparing serial and parallel smoothed-transfer application and preserving serial as the generic default.
- Add `bench-fem-hybrid-coarse-transfer.ps1` for repeated serial-vs-parallel A/B runs with Graph aggregation, Jacobi-smoothed basis and target 1792 fixed.
- r21a result: Jacobi smoothing reduced L-angle iterations from 1044 to 876 (16.09%) but raised median milliseconds/iteration from 19.464 to 24.806, worsening solver/wall time by 6.94%/10.04% and increasing coarse memory by 37.71 MiB; r22 targets this hot-path transfer overhead without changing the coarse space.

### 0.7.0-r21a

- Fix the r21 workspace Clippy gate without suppressing lints or changing smoothed-basis numerics.
- Replace the four-element smoothed-transfer tuple return with a private `JacobiSmoothedTransfer` struct to resolve `clippy::type_complexity`.
- Iterate over the extracted diagonal with `enumerate()` when building the Jacobi-smoothed transfer to resolve `clippy::needless_range_loop`.
- Preserve the r21 Graph + target-1792 Jacobi-smoothed Galerkin experiment unchanged.

### 0.7.0-r21

- Add an experimental `TwoLevelBasis::JacobiSmoothed` coarse transfer for the generic algebraic two-level preconditioner while preserving `PiecewiseConstant` as the default.
- Build the smoothed transfer with one damped Jacobi step `P = (I - omega D^-1 A) P_tent`, where `omega = 4 / (3 rho)` and `rho` is estimated by deterministic power iteration on `D^-1/2 A D^-1/2`.
- Form the true Galerkin coarse operator `P^T A P` from the sparse smoothed transfer without materializing `A P`, and retain the sparse transfer for restriction/prolongation during PCG.
- Extend `AlgebraicCoarseOptions` and `fem_bench` with `--coarse-basis piecewise|smoothed`; keep aggregation, coarse dimension and coarse-apply policy independently selectable.
- Add regression tests for positive smoothed-basis action and the default piecewise-constant policy.
- Add `bench-fem-hybrid-coarse-basis.ps1` for repeated Graph + target-1792 piecewise-vs-smoothed A/B runs.
- r20 result: StrongGraph reduced iterations only 0.29% versus Graph and increased median solver/wall time by 1.30%/1.36%, so strength-prioritized aggregation remains experimental rather than becoming the default.

### 0.7.0-r20

- Add `TwoLevelAggregation::StrongGraph`, a geometry-free strength-of-connection variant that grows deterministic graph aggregates by normalized node-block Frobenius coupling strength.
- Keep the existing `Graph` and `Contiguous` aggregation policies unchanged and preserve `Contiguous` as the generic default while the new policy is benchmarked.
- Extend `fem_bench --coarse-aggregation` with `strong-graph` / `graph-strong` / `strength`.
- Add a regression test proving that StrongGraph prefers the stronger block coupling on a diagonally dominant SPD graph while preserving positive preconditioner action.
- Add `bench-fem-hybrid-coarse-strength.ps1` for repeated Graph vs StrongGraph A/B runs at fixed coarse target, apply policy, and local-factor settings.

# Changelog
### 0.7.0-r19

- Add `bench-fem-hybrid-graph-coarse-sweep.ps1` to re-optimize coarse-space size after r18b demonstrated a 55.15% iteration reduction and 52.88% median wall-time reduction from graph-aware aggregation at the previous contiguous optimum.
- Sweep graph aggregation by default at targets 512, 768, 1024, 1152, 1280, 1536, 1792, and 2048 with three alternating-order repeats, keeping coarse-apply `Auto` enabled.
- Report effective coarse-apply policy, actual coarse dimension, convergence, escalations, local-region count, local-factor memory, coarse memory/setup, solver milliseconds per iteration, solver time, wall time, and wall-time range.
- Identify the best all-converged graph target by median wall time and compare it with graph target 1792 for wall-time and coarse-memory improvement.
- Make no Rust, solver, preconditioner, public API, default-policy, or numerical changes from r18b; this checkpoint is benchmark-only.

### 0.7.0-r18b

- Normalize the r18a Rust sources to the workspace rustfmt layout so `cargo fmt --all -- --check` passes from the distributed archive.
- Apply formatting-only changes reported by the user's Rust 1.98 toolchain; no solver, preconditioner, graph aggregation, benchmark logic, API semantics, or numerical behavior changes.
- Preserve the r18a Clippy fix and the r18 graph-aware aggregation experiment unchanged.

### 0.7.0-r18a

- Fix the r18 graph-aggregation regression test so the workspace-wide Clippy `needless_range_loop` gate passes without adding a lint suppression.
- Replace the indexed six-row test-data initialization with `iter_mut().enumerate()`; no solver, preconditioner, graph aggregation, benchmark, or numerical behavior changes.

### 0.7.0-r18

- Add an experimental graph-aware aggregation mode for the generic geometry-free algebraic two-level preconditioner while preserving contiguous aggregation as the default.
- Infer node adjacency from CSR block sparsity for arbitrary `dofs_per_node`, then form deterministic breadth-first piecewise-constant aggregates without requiring coordinates or structural rigid-body modes.
- Merge undersized graph fragments into their strongest adjacent aggregate while keeping disconnected components valid; store the graph assignment only for graph mode and include it in persistent preconditioner memory accounting.
- Extend `AlgebraicCoarseOptions` with `TwoLevelAggregation` and add `fem_bench --coarse-aggregation contiguous|graph`.
- Add `bench-fem-hybrid-coarse-aggregation.ps1` for repeated alternating-order A/B testing at the validated L-angle coarse target 1792 using the existing coarse-apply Auto policy.
- Keep generic Auto on contiguous aggregation until the L-angle A/B result demonstrates a graph advantage; this checkpoint changes no default solver path.
- Add regression coverage that verifies graph aggregation follows matrix connectivity rather than consecutive node numbering while preserving a positive preconditioner application.

### 0.7.0-r17a

- Fix the r17 resumable-PCG API so `pcg_continue_with_workspace` satisfies the workspace-wide Clippy `too_many_arguments` gate without adding a lint suppression.
- Move continuation tolerance ownership fully into `PcgSession`; a continuation no longer accepts redundant `SolverOptions` because its convergence target was fixed when the session started.
- Replace the private many-argument hybrid continuation helper with `HybridPcgContext`, grouping the fixed operator, algebraic coarse correction, local preconditioner, and RHS while keeping `x`, the iteration budget, session state, and workspace explicit.
- Preserve r17 numerical behavior and restart semantics: a preconditioner change starts a new session, while controller-only segmentation with an unchanged preconditioner preserves the Krylov recurrence.

### 0.7.0-r17

- Add resumable serial PCG sessions in `hybit-krylov` so controller/telemetry boundaries can advance the same Krylov recurrence without recomputing the residual or discarding the conjugate search direction.
- Preserve the strict PCG invariant that a preconditioner change starts a new session; continuation is used only while the operator and SPD preconditioner remain unchanged.
- Update generic selective-direct escalation so an acceptable stage, or a later diagnostic that finds no stronger region set, spends the remaining iteration budget by continuing the current PCG session instead of restarting from the current `x`.
- Keep ordinary `pcg_with_workspace` on the same implementation by expressing it as one start plus one continuation, avoiding duplicate serial-PCG recurrence logic.
- Add a Krylov regression test that compares one uninterrupted solve against a segmented 2-iteration + continuation solve and requires matching iterations, residual, and solution.
- The current L-angle three-stage benchmark is not expected to improve materially from this checkpoint because its final escalation already receives the entire remaining iteration budget; r17 primarily fixes the general controller semantics.

### 0.7.0-r16

- Add `TwoLevelCoarseApplyPolicy::Auto` for the generic algebraic two-level coarse correction.
- Resolve `Auto` from the **actual** post-aggregation coarse dimension: dimensions below 1024 use packed `FactorSolve`, while dimensions at or above 1024 use `ExplicitInverse`.
- Base the 1024 threshold on the repeated L-angle crossover benchmark: actual dimension 768 favored factor solve slightly, while actual 1275, 1791, and 2037 favored explicit inverse by about 4.2%, 7.7%, and 7.5% median wall time respectively.
- Make `AlgebraicCoarseOptions` and `fem_bench --coarse-apply` default to `Auto`, while keeping the low-level `TwoLevelBlockJacobiPreconditioner::from_csr32()` constructor backward-compatible with `FactorSolve`.
- Preserve explicit `factor` and `inverse` overrides and print both requested and effective coarse-apply policies in `fem_bench`.
- Add regression tests for the Auto crossover resolution and the generic algebraic-coarse default policy.

### 0.7.0-r15

- Add `bench-fem-hybrid-coarse-apply-crossover.ps1` to measure `FactorSolve` versus `ExplicitInverse` across several algebraic coarse dimensions with repeated, alternating-order runs.
- Default crossover targets are 768, 1280, 1792, and 2048 with three repeats per policy/target.
- Report per-target median setup cost, solver milliseconds per iteration, solver time, wall time, factor-to-inverse speedup, and an approximate break-even iteration count.
- Identify the first measured coarse target where `ExplicitInverse` wins on median wall time; no solver, preconditioner, numerical, or Rust API changes from r14a.

### 0.7.0-r14a

- Fix the `ExplicitInverse` Rayon apply path so its parallel closure captures only the immutable coarse-inverse slice, the immutable coarse RHS, and the coarse dimension instead of capturing `&TwoLevelBlockJacobiPreconditioner`.
- Avoid imposing `Sync` on the preconditioner-wide `RefCell<CoarseScratch>`; no lock or scratch-layout change is required.
- Preserve the r14 coarse inverse algorithm, public API, numerical operation, memory model, and benchmark interface.

### 0.7.0-r14

- Add an experimental `TwoLevelCoarseApplyPolicy::ExplicitInverse` path for the geometry-free algebraic coarse correction.
- Preserve the existing packed-Cholesky `FactorSolve` path as the default.
- For explicit-inverse mode, build the dense symmetric coarse inverse once during setup, discard the temporary packed factors, and apply the inverse with a row-major Rayon-parallel dense matvec in each PCG iteration.
- Keep persistent coarse storage approximately `O(n_coarse^2)` in both policies; explicit inverse trades additional setup work for lower iteration-time dependency/latency.
- Extend `AlgebraicCoarseOptions` and `fem_bench` with a coarse-apply policy (`--coarse-apply factor|inverse`).
- Add `bench-fem-hybrid-coarse-apply.ps1` for repeated factor-vs-inverse A/B testing at the empirically selected L-angle target 1792.
- Add a numerical equivalence/positivity regression test comparing explicit-inverse and factor-solve coarse application.

### 0.7.0-r13

- Refine the generic hybrid algebraic-coarse sweep around the observed L-angle optimum with targets 1280, 1536, 1792, and 2048.
- Repeat each coarse target three times by default and compare medians instead of single-run wall times; alternate ascending and descending target order between repeats to reduce systematic order bias.
- Require every repeat to converge before a target is eligible for the automatic best-target selection.
- Report per-run measurements plus median solver/wall time, median iterations, median coarse setup, median milliseconds per iteration, maximum verified residual, and wall-time range.
- Fix the sweep summary so `TargetCoarseDim` and the best-target message retain the requested target value instead of rendering blank.
- Export both raw per-run CSV data and an aggregated median summary CSV.
- No solver, preconditioner, numerical, or Rust API changes from r12.

### 0.7.0-r12

- Extend the hybrid algebraic-coarse sweep upward to targets 1280, 1536, 1792, 2048, 2304, and 2560 after the L-angle sweep showed wall time still improving through the 1533-dimensional coarse space.
- Preserve the numeric CSV output while formatting the console residual column explicitly in scientific notation so near-tolerance values no longer render as `0.00`.
- Add solver milliseconds per iteration to the sweep summary and automatically identify the lowest-wall-time converged target.
- No solver, preconditioner, numerical, or Rust API changes from r11a.

### 0.7.0-r11a

- Fix `bench-fem-hybrid-coarse-sweep.ps1` so the local-only reference run can pass an empty `ExtraArgs` collection on PowerShell.
- No solver, preconditioner, numerical, or Rust API changes from r11.


## 0.7.0-r10 - bidirectional packed algebraic coarse solve

- Store the generic algebraic coarse Cholesky factor as packed rows of both `L` and `L^T`.
- Keep persistent coarse-factor memory approximately unchanged while eliminating column-stride reads in backward substitution.
- Preserve the coarse space, additive SPD composition, public solver semantics, and numerical result.
- Add a storage-layout regression test for the bidirectional packed factor.


## 0.7.0 development checkpoints

- r9: added an opt-in geometry-free algebraic two-level base for the generic hybrid path, composed additively with selective-direct local corrections so PCG remains SPD-compatible; the coarse factor is built lazily on first escalation, reused across later stages and prepared solve-many RHS vectors, and reported separately from the local-direct memory budget. Added `bench-fem-hybrid-coarse.ps1` for selective-direct-only vs algebraic-two-level+selective-direct A/B testing.
- r8: promoted `JacobiEnergyPerByte` to the default local-factor selector after the constrained L-angle FEM A/B/C benchmark, added per-escalation telemetry (iterations, residual ratio, active regions, unique covered DOFs, and persistent local-factor bytes) to `SolveReport` and `fem_bench`, and stabilized the selector CLI parser against the rustfmt match-arm oscillation seen on Windows.
- r7: added an experimental `JacobiEnergyPerByte` selector using `sum(r_i^2 / A_ii)` per incremental factor byte, restored `CandidateOrder` as the conservative default after the 1 MiB L-angle benchmark showed raw residual-energy/byte underperforming candidate order, and extended the FEM selector benchmark to three-way A/B/C comparison.
- r6: harvested multiple residual-centered connected local regions from large seeded numerical-risk components so the local-direct memory budget and benefit/byte selector receive a meaningful candidate pool on large FEM systems.
- r5: added an explicit local-factor selection policy (`CandidateOrder` or `BenefitPerByte`) and an A/B FEM benchmark path that holds matrix, RHS, tolerance, escalation settings, and persistent local-factor memory budget constant.
- r4: ranked newly discovered local-direct regions by uncovered residual energy per incremental persistent factor byte while preserving already learned regions.
- r3: added bounded multi-stage selective-direct escalation with PCG restart only when the SPD preconditioner is strengthened.
- r2: added a persistent local-direct factor-memory budget.
- r1: packed local Cholesky storage.


## 0.6.0-r26g MSRV dependency pin

- Preserve the declared Rust 1.73 MSRV by pinning `rayon-core = "=1.12.1"` alongside `rayon = "=1.10.0"` in every published HyBIT crate that uses Rayon (`hybit-matrix`, `hybit-krylov`, and `hybit-precond`).
- Prevent Cargo from resolving Rayon 1.10.0's compatible `rayon-core ^1.12.1` requirement to `rayon-core 1.13.0`, which requires Rust 1.80.
- Extend the workspace metadata release gate to verify both exact pins so a future manifest change cannot silently raise the effective MSRV.
- No solver algorithm, numerical operation ordering, or public HyBIT API behavior is changed.

## 0.6.0-r26f rustdoc/doctest cleanup

- Mark structural two-level preconditioner equations as `text` code fences so rustdoc does not compile mathematical notation as Rust doctests.
- Fix the four release-gate doctest failures in `hybit-precond` without changing executable code, public API behavior, or numerical operation ordering.
- Preserve the r26e rustfmt stabilization and the Clippy-clean workspace state.

## 0.6.0-r26e rustfmt stabilization

- Rewrite the structural example `--precond` parser arms with an explicit local value so rustfmt cannot oscillate between direct-expression and block-arm forms.
- Preserve parser behavior, solver algorithms, structural Auto policy, and numerical operation ordering.
- Carry forward the r26d FFI safety documentation/fix and the Clippy-clean workspace state.

## 0.6.0-r26d RC FFI safety cleanup

- Synchronize the two remaining structural example parser forms with the Windows rustfmt output used by the release-candidate gate.
- Document explicit `# Safety` contracts for all 14 public `unsafe extern "C"` entry points in `hybit-ffi`.
- Make `hybit_matrix_create_csr_f64` safely honor its existing zero-nnz API behavior by avoiding `slice::from_raw_parts` on null `col_idx`/`values` pointers when `nnz == 0`.
- Keep solver algorithms, structural Auto policy, and numerical operation ordering unchanged.

## 0.6.0-r26c RC lint follow-up

- Apply the remaining rustfmt changes reported by the Windows release-candidate gate.
- Keep the internal `run_continuation` helper unchanged and narrowly allow Clippy's `too_many_arguments` lint rather than refactoring release-candidate control flow.
- No solver algorithm, numerical operation ordering, structural Auto policy, or public API behavior is changed.

## 0.6.0-r26b RC lint cleanup

- Apply the full workspace `rustfmt` normalization required by the 0.6.0 release-candidate gate.
- Complete the public `Preconditioner` collection-style API with a default `is_empty()` implementation.
- Resolve current stable Clippy `-D warnings` findings in matrix and preconditioner code using semantics-preserving iterator forms, `div_ceil`, and `sort_unstable_by_key`.
- Keep solver algorithms, structural Auto policy, and numerical operation ordering unchanged.

## 0.6.0-r26 release-gate hardening

- Feature freeze remains in effect; solver algorithms are unchanged from r25.
- Added `release-candidate-gate.ps1` as the authoritative clean-tree RC orchestration gate.
- Added source-integrity and workspace-metadata gates.
- Added a real L-angle FEM regression gate that verifies Auto policy selection, convergence/residual, iteration guard, and prepared solve-many reuse without imposing machine-dependent timing thresholds.
- Added local Rust 1.73 MSRV verification and stable `cargo fmt`/Clippy checks to the RC gate.
- CI now runs on `develop/0.6.0` pushes and includes formatting/Clippy checks.


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

## 0.7.0-r11 (development checkpoint)

- Add `--skip-plain` to `fem_bench` so parameter sweeps can avoid rerunning the expensive Plain Jacobi-PCG baseline.
- Add `bench-fem-hybrid-coarse-sweep.ps1` for local-only + algebraic-coarse target sweeps.
- Default sweep targets are 384, 512, 768, 1024, and 1536 coarse dimensions.
- The sweep prints a compact Pareto table and exports status, iterations, verified residual, actual coarse dimension, coarse memory/setup cost, solver time, total wall time, and per-stage residual ratios to CSV.
