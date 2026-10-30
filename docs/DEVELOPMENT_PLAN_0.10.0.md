# HyBIT 0.10.0 development plan — correctness first, then measured value

**Status:** development proposal, not a release announcement (2026-10-28).
**Base:** published `v0.9.0` at `0a8cd327`; branch `develop/0.10.0`.
**Primary target:** WSL2 Ubuntu 26.04.1 / Rust 1.99.1 / Open MPI 5.0.10,
while Windows 11 MSVC + Microsoft MS-MPI remains a separate supported ABI and
native-MPI qualification environment.

This plan incorporates the independent positive **and critical** 0.9.0 review.
It accepts the criticism that excellent unit tests and cautious wording do
not substitute for comparative real-FEM time-to-solution. It does **not**
accept the inference that weaker numerical or ABI gates would produce better
performance evidence: false convergence makes a benchmark meaningless.

## What 0.10.0 must demonstrate

1. **Correctness is non-negotiable.** High-level solver reports independently
   certify original-CSR `||b-Ax||`, identify breakdown separately from an
   iteration cap, and do not infer SPD or forward accuracy from convergence.
   The first code checkpoint is the staged post-0.9.0 hardening patch.
2. **A win is defined by total cost.** On fixed matrix/RHS/tolerance, compare
   cold start (read + analyze + preparation + solve), prepared repeated RHS
   (amortized setup + solves), peak RSS, and original-CSR residual/forward
   error. Report failures in the same dataset as successes.
3. **A baseline must be competitive.** Include current Jacobi-PCG, the
   relevant HyBIT Hybrid/structural options, and independent native AMG
   baselines where installed: PETSc GAMG, PETSc PCHYPRE/BoomerAMG or hypre
   BoomerAMG. On appropriate small/medium systems, a sparse-direct reference
   such as MUMPS/SuperLU can inform accuracy and memory feasibility.
   Match operator, RHS, stopping rules, precision and thread/affinity policy.
4. **Autonomy is earned.** Observe where Jacobi, ILU(0), local Cholesky,
   region sizes, RCM and coarse correction win or lose; initially expose
   explicit suggestions/diagnostics. Do not turn any tuning result into a
   production default without held-out evidence and budget guards.
5. **ABTM must pay rent.** Evaluate metadata-driven region planning, masked
   symbolic work, compact layout and partition/Halo savings against a
   strong prepared-CSR reference. Retain CSR execution where it wins;
   numeric kernel microbenchmarks are not full-solve speedups.

## First real-FEM experiment: bounded local-direct region size

**Hypothesis:** Some ill-conditioned SPD FEM cases benefit from increasing
`max_local_region_size` from 128 to 256 or 512, provided total factor bytes
stay fixed. Other cases will lose from cubic local-factor work.

| Independent parameter | Values for pre-registered pilot |
| --- | --- |
| Matrices | `boneS01`, `cant`, `thermal1`, plus small `poisson5` smoke; add L-angle when available |
| Solver class | SPD **only after independent matrix-class checks**; do not infer from positive diagonal |
| RHS | A documented manufactured vector `b=A*x_ref` and one application RHS if available |
| Rel. stopping tolerance | `1e-8` on original CSR; keep max iterations identical |
| Local region limit | `64, 128, 256, 512` DOF |
| Local-factor budget | Fixed 64 MiB initially; also record actual factor bytes and skipped regions |
| Local region count | Fixed 8, with unchanged overlap/probe/selector/coarse policy |
| Preparation/reuse | Cold one-shot plus 5-RHS prepared reuse (same matrix coefficients) |
| Trial design | At least 3 warmups + 9 measured independent trials; rotate execution order, control CPU affinity/threads, report median and dispersion |
| Numerical gate | Solver status `Converged`, independently recomputed true residual `<=tol` (with explicit rounding margin), finite data, and manufactured-solution error |
| Memory gate | No factor-budget overrun; report peak process RSS in addition to internal estimates |

The existing `crates/hybit/examples/fem_bench.rs` already accepts
`--max-region`, `--max-regions`, `--factor-budget-mib`, `--tol`,
`--max-iters`, `--selector`, `--backend` and an explicit RHS.
**Reuse this CLI** for the pilot. C1-A adds the opt-in WSL2 Python runner
`benchmarks/scripts/fem_region_sweep.py` to collect randomized repeated
one-shot runs, raw logs, CSV/JSON, peak child RSS, true-residual outcomes and
zero-escalation diagnoses; see `docs/C1_FEM_REGION_SWEEP.md`. Prepared
five-RHS reuse, native AMG baselines and external class certification are
**not yet measured** by this runner.
The C1-A WSL `thermal1` pilot reported identical Jacobi-PCG behavior and zero
escalations across the four region limits. The subsequent **C1-B candidate**
adds an explicitly opt-in original-CSR residual monitor (`--monitor-iters`)
to test whether later poor progress can reveal local-direct opportunities.
Its default remains off; the monitor must earn adoption by **paired**
time-to-solution, RSS, and independently verified residual evidence.
The WSL C1-B pilot confirmed that the same 0.5 residual ratio at different
monitor intervals is not an equal per-iteration convergence criterion.
C1-C therefore separates observation from policy, reports geometric per-step
progress on the original CSR, and requires an explicit experimental
rate-limit before allowing post-probe monitored escalation. Local factor
ROI, AMG comparisons, and production auto-routing remain unproven.
The C1-D candidate adds a randomized **paired observational control** to
the existing WSL region sweep, requiring both arms to pass numerical gates
and an actual escalation before claiming any local-factor economic evidence.
It does not change the solver's default policy or prove a win.
Its default `b=A*1` is a manufactured solution **only**, not a physical
load or a certificate that the matrix is SPD.

Do not use `cant` to claim success from a low residual alone: existing
validation found a small residual alongside poor manufactured forward error.
An exploratory failure can remain in the table without being called PASS.

## Comparative and distributed qualification

- Measure native WSL Open MPI 1/2/4/8 ranks, then held-out real-FEM matrices
  with controlled rank/core placement; record **maximum rank time**, setup,
  partition, Halo, reductions and peak per-rank RSS. WSL MPI and native
  Windows MS-MPI remain separate gates.
- User-provided WSL2 **baseline-before-patch**: standard Debug and thin-LTO
  Release workspace tests 236 each PASS; MPI feature tests 100 PASS;
  synthetic `mpi_owned_pcg` 20 cases each at 1/2/4/8 ranks (**80/80 PASS**).
  This is correctness evidence, not a multicore/multi-node speedup.
- No native hypre/PETSc/MUMPS availability has been established on WSL2
  yet. Mark external-baseline columns `NOT RUN` until reproducible
  commands, versions, options, and actual results are captured.

## Milestone / go-no-go gates

| Milestone | Deliverable | Pass condition |
| --- | --- | --- |
| C0 — reliability | Original-CSR certifier, finite PCG, FFI classification, CI integrity | Debug/release tests, strict Clippy/rustfmt, source hashes, native ABI/MPI cross-check |
| C1 — reusable evidence | Machine-readable pilot runner with explicit config and failure rows | Repeated measurements reproducible; no silently accepted nonconvergence; all metadata captured |
| C2 — local region economics | Fixed-budget 64/128/256/512 study on validated SPD cases | Truth residual/forward error + cold/warm time + peak RSS + held-out loss cases |
| C3 — competitive comparison | AMG / optional sparse-direct reference on identical cases | Versioned baseline, matched tolerance, apples-to-apples setup/reuse/rank/threads |
| C4 — guarded autonomy | Explicit recommendation based on pilot and held-out data | Conservative budget cap, no regression gate failures, opt-in only until validated |

Keep long-term MINRES, pivoted LDLᵀ/LU, frictional contact, variable DOF and
global sparse-direct fallback on the architecture roadmap, but do not widen
the first 0.10.0 checkpoint to implement them without a concrete class-specific
numerical corpus. GPU remains later.

## Development discipline, simplified

- Each milestone needs a **short hypothesis, executable acceptance commands
  and one summary table**, not a proliferation of tiny numbered reports.
- A failing case stays visible; the goal is a reliable map of where HyBIT
  wins, ties or loses rather than cherry-picked speedups.
- Preserve the simple `hybit::solve` facade and existing C ABI. Current
  published `v0.9.0` stays immutable; this branch is unreleased.
- Update `docs/SESSION_HANDOFF.md` with exact HEAD/dirty files/PASS/FAIL
  after meaningful work. Commit/push only with the owner's authorization.

**Next single action after C0 gates:** implement a repeatable, machine-readable
real-FEM region-size sweep runner around the existing `fem_bench` CLI,
without modifying the production solver's defaults.
