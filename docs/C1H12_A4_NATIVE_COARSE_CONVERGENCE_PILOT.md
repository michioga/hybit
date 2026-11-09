# C1-H12-A4: HyBIT-native two-level convergence pilot

Status: **opt-in experiment to run on the owner's WSL2**, not accepted
production behavior. This pilot uses **no PETSc or hypre code, linkage or
runtime dependency**. PETSc GAMG and BoomerAMG remain external references.

## Goal and numerical contract

A2/A3 established that an ABTM representation can generate the exact
symbolic candidate for two-hop support and one-hot `P^T A P`, but cannot by
itself establish AMG convergence. The next question is whether the
**existing native** `TwoLevelBlockJacobiPreconditioner` reduces iteration
counts on the C1-G difficult SPD manufactured-solution FEM corpus.

The executable is the existing `fem_coarse_crosscheck` example, extended
with optional selection flags, an explicit Jacobi-PCG control and one
`C1H12A4|` log record. Existing defaults (`Graph`, `JacobiSmoothed`, F32,
direct coarse, tol=1e-8, 3000 iterations) are unchanged. Production
`hybit::solve`, selector, public C ABI and automatic preconditioner routing
are unchanged.

All arms call the **same** `hybit::pcg`, start at `x=0`, and solve with
`b=A*ones` on the same original CSR operator. Pilot overrides:

| Property | Controlled value |
|---|---|
| Matrices | island_shell, multi_patch, rotated_anisotropy, n=20,736 |
| PCG | relative tol=1e-10, max_iters=600, abs tol=0 |
| Coarse target | 512 scalar coarse DOF, 1 fine DOF per node |
| Transfer values | F64 (avoid mixed-precision confound) |
| Execution | RAYON_NUM_THREADS=1, OMP_NUM_THREADS=1 |
| Accuracy | solver-reported residual, independently recomputed original-CSR true relative residual, relative error to `ones` |
| Per-process costs | Matrix load, preconditioner setup, solve, GNU time wall/RSS |

Seven treatment arms per matrix:

1. Jacobi only (no coarse factor allocated): negative control.
2. Contiguous aggregation + piecewise-constant transfer.
3. Contiguous aggregation + Jacobi-smoothed transfer.
4. Graph aggregation + piecewise-constant transfer.
5. Graph aggregation + Jacobi-smoothed transfer.
6. StrongGraph aggregation + piecewise-constant transfer.
7. StrongGraph aggregation + Jacobi-smoothed transfer.

The current implementation uses a single coarse level and a native coarse
Cholesky/auto solve: this is **two-level algebraic coarse correction**, not
a standalone multilevel AMG hierarchy, not classical/interpolation AMG,
and not BoomerAMG. Neither the C1-H12-A3 symbolic code nor the ABTM
candidate is wired into this numerical preconditioner pilot.

## Owner WSL reproduction

Apply the delivered A4-only patch atop **both** already-applied A2 and A3
probes, on `develop/0.10.0` HEAD `4593c20`, with no staged changes:

```bash
cd ~/hybit
bash benchmarks/scripts/C1H12_A4_native_coarse_pilot.sh --dry-run
bash benchmarks/scripts/C1H12_A4_native_coarse_pilot.sh --run
```

Optional second repeat count (maximum 10):

```bash
bash benchmarks/scripts/C1H12_A4_native_coarse_pilot.sh --run \
    "$HOME/mtx/c1g_g2_hard_z32Jkd" 3
```

The runner first checks manifest hashes, rustfmt, Rust 1.99.1 Clippy/build
and the Python negative tests. Every one of the 21 process outcomes is
saved under `~/c1h11_evidence/c1h12_a4_native_XXXXXX/` with `.log`,
`.time`, git HEAD and input matrix hashes. The Python numerical gate
writes `native_coarse_summary.csv`, **retains failures** and fails if any
arm is missing, exits nonzero, contains invalid cost metadata, reports
false convergence, or claims MaxIterations while its independently
measured true residual already meets tolerance.

Correctly reported `MaxIterations` does **not** fail the evidence gate:
such a result is a valuable measurement of an ineffective preconditioner.
An honestly reported `Breakdown` with a finite above-tolerance residual is
also retained as a negative numerical outcome (not a convergence success).
Relative error to the known manufactured solution is recorded separately:
small residual alone does not imply a small solution error for an
ill-conditioned system.

## Decision after owner measurement

Report, per case, 600-iteration Jacobi status, each coarse arm's numerical
status, independently verified residual, iteration count, solution error,
setup/solve timing and coarse factor memory. Prioritize **numerical rescue
at the fixed budget**, then factor memory, then total setup + solve wall.
Do not infer a speedup from a single uncontrolled process run.

If none of these two-level variants reaches true-relative 1e-10 at 600
iterations, retain that negative evidence and investigate native multilevel
construction (strength graph, stable aggregation, smoothed transfer,
Galerkin numeric values, recursive V-cycle) with SPD/symmetry/energy and
raw-residual tests before touching default routing. If an arm rescues the
hard cases, isolate the successful **numerical mechanism**, then benchmark
and reduce its memory costs. **No automatic AMG promotion** is authorized
by the symbolic A2/A3 results or this pilot alone.

Related evidence: `docs/C1H12_A2_A3_REAL_FEM_EVIDENCE.md`.
