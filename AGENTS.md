# HyBIT: instructions for AI development sessions

> Read this file first, then `docs/PROJECT_CHARTER.md` and `docs/SESSION_HANDOFF.md`. The local repository and its test output take precedence over GitHub, older conversations and assumptions.

## Non-negotiable product goal

HyBIT (Autonomous Hybrid Sparse Solver) provides a **simple, stable API to sparse-solver users** while the implementation automatically seeks robust convergence and high performance:

1. Begin with memory-efficient iterative Krylov methods selected for the *declared mathematical class* of the matrix.
2. Monitor true residuals, progress, stalls, breakdown and resource budgets. Escalate preconditioning and/or solver strategy instead of letting an iteration fail silently. The long-term objective includes an appropriate sparse direct solve fallback when necessary and feasible.
3. Use **ABTM** for compact sparse topology and values where evidence supports it, bitmap Boolean operations, symbolic work, graph regions, and efficient distributed **Halo planning/exchange**.
4. Exploit **Rayon** intra-process parallelism and **MPI** inter-process parallelism for end-to-end improvements in time-to-solution, scalable memory use and robustness, not just isolated microbenchmark speed.
5. Keep implementation choices mostly internal; do not force ordinary users to tune ABTM, ranks, memory layout, preconditioner switches, or many solver knobs.

## Important implementation distinction

**Already implemented:** accessible Rust `hybit::solve` / `HybitSolver`, analyzed/prepared reusable solvers, SPD PCG and hybrid selective *local* Cholesky preconditioning, GeneralSquare FGMRES and explicit ILU(0), ABTM topology/Boolean operations, prepared CSR paths, Rayon kernels, MPI rank-local CSR, Halo exchange and overlapped distributed PCG, and G8 experiments. Check actual code and branch before changing anything.

**Not proven complete:** generic, automatic **whole-system sparse direct solver fallback** after a failed iterative solve; universal ABTM faster-than-CSR SpMV; production-ready ABTM-compressed numeric Halo payload. Do not conflate local Cholesky correction with a global direct solver or claim either is implemented without source and tests.

## Workflow for EVERY new session

1. Treat the user's **local `D:\Work\hybit\hybit` worktree** as canonical. If it is not available, ask for the current ZIP/log or use the exact current commit as a clearly marked fallback; never silently substitute an older remote branch.
2. Inspect `git status --short`, `git log -5 --oneline`, branch, `Cargo.lock`, relevant modules, tests and prior benchmarks **before** proposing a patch. Read this charter and handoff file.
3. First reuse and extend existing crates, public APIs, ABTM algorithms, `mpi_backend`, `mpi_overlap`, `mpi_pcg`, tests and scripts. Do not create parallel implementations of completed functionality.
4. State hypothesis, required metrics, correctness gate, rollback and expected touched files *before* implementation. Separate environment diagnosis from algorithm experiments and release checks.
5. Preserve true residual and mathematical-class contracts; explicitly label experimental/unverified SPD data. An MPI-enabled benchmark must actually execute under MS-MPI, not just emulate ranks.
6. Use Windows 11, `x86_64-pc-windows-msvc`, **Microsoft MPI/MS-MPI**, LLVM/libclang and native MSVC build tools as the supported local validation target. Do not switch to MinGW MPI, OpenMPI or a different target as an ad hoc workaround.
7. Keep commits small and frequent; **push validated checkpoints frequently** on the user's chosen branch. Do not impose a blanket commit/push freeze. Do not commit/push on the user's behalf without authorization. Report working tree and verified gates; do not call a failed/unrun gate PASS.
8. Protect `MANIFEST.txt` / `SOURCE_SHA256.txt` integrity. Both must agree with tracked files, including docs and sources. Prefer clean, reviewable changes over generated one-off patches.
9. A speedup claim always identifies baseline, matrix corpus, rank count, CPU, threads, warmups/repeats, measured scope, conversion/setup costs, memory budget and numerical checks. Compare whole distributed solve performance, not mask-only timings.
10. Commit/push and release are separate. **GitHub Releases and crates.io publication occur only after a major, validated milestone and the user's explicit authorization.** Check API compatibility, numerical/CI gates, source packaging and `cargo publish --dry-run` before publishing. Never tag, release, or publish as an incidental step of ordinary development.
11. Update `docs/SESSION_HANDOFF.md` after each meaningful checkpoint (or before ending a session): branch/commit, dirty files, what passed/failed, measured results, unresolved blockers and exactly one next prioritized action.

## Read before optimizing

- `README.ja.md` — intended public API and current hybrid solver.
- `docs/ABTM_TOPOLOGY.md` — ABTM G1–G7 evidence; bitmap metadata-first design; CSR remains a strong scalar SpMV backend.
- `docs/DEVELOPMENT_STATUS.md`, `docs/ROADMAP.md`, `docs/SOLVER_SELECTION.md` — actual/current progress and mathematical safeguards; verify their dates.
- `crates/hybit-auto`, `crates/hybit-matrix`, `crates/hybit-distributed` — implementation ground truth.

Avoid adding features merely because they are on a roadmap if a suitable implementation or measurement already exists.
