# C1-G3-A2: opt-in real local-Cholesky shadow trial

Status: **WSL Rust gates PASS and real-FEM evidence collected; shadow admission remains experimental and disabled by default**.
Source baseline: `develop/0.10.0` at `8da0913` (C1-G3-A1 accepted).

## Intent

On SPD systems, A2 tests the *existing* real-FEM region selector and packed
local-Cholesky `HybridPreconditioner`, instead of the synthetic diagonal
candidate from A1. A user must explicitly enable it. The default controller,
public `SolveReport` layout and existing local/coarse policies are unchanged.
This is an experimental **admission check**, not an autonomous coarse solver
or a guarantee of better eventual convergence or elapsed time.

## Experimental policy

`HybridOptions` adds:

| Option | Default | Meaning |
| --- | ---: | --- |
| `shadow_trial_iterations` | `0` | `0` disables A2. Otherwise max PCG steps in each branch. |
| `shadow_accept_ratio` | `0.8` | Candidate true-residual norm must be at most this fraction of the baseline. |
| `shadow_max_extra_bytes` | `16 MiB` | Upper bound for a copied solution and five PCG workspace vectors (`6*n*8` bytes). |

Opt-in requires one maximum local escalation, normal enabled SPD controller,
no algebraic coarse, weighted routing, monitor-only mode or PCG kernel
profiling. The existing factor-memory budget remains separate. No additional
`PcgWorkspace` scratch beyond the cloned vectors is counted; local-factor
setup temporaries and Rust allocator overhead are *not* in this clone cap.

At the existing escalation trigger, and **after** the standard selection and
local-factor preparation:

1. Obtain an independent original-CSR checkpoint residual.
2. Clone the live Jacobi-PCG solution and five-vector `PcgWorkspace`.
3. Start a new PCG recurrence with the actual local-Cholesky preconditioner
   on the clone, for up to `shadow_trial_iterations` steps.
4. Advance the untouched, live Jacobi recurrence by the same step allowance,
   without restarting it.
5. Independently compute both original-CSR residual norms. Adopt the
   candidate only if it is finite, did not break down or falsely converge,
   improved the checkpoint norm, and reached the configured ratio relative
   to the baseline. A baseline already satisfying the true target wins.
6. Rejected/error candidates are dropped; the live Jacobi session continues
   with its original recurrence. An accepted candidate's copied session,
   workspace, solution and local factor become the continuing solver state.

The accepted-branch `SolveReport.iterations` counts **only** selected-path
iterations; the comparison executes additional work. `trial_seconds`,
`baseline_iterations` and `candidate_iterations` expose the trial overhead.
The A2 stage telemetry covers the *trial portion* of the local stage; total
iterations include the subsequent continuation. The prepared context exposes
`shadow_trial_report()` for its most recent RHS; it is cleared each solve.

`HybridShadowDecision`: `Accepted`, `Rejected`, `CandidateFailed`, or
`SkippedCloneBudget`. `NotTriggered` in the example means the controller did
not build a local candidate, and is **not** a claim of acceptance/rejection.
If factorization fails before the trial, the existing fail-closed behavior
continues and no candidate decision is produced.

## WSL acceptance (first run)

Use Rust **1.99.1** with `--locked`. After installing the guarded patch and
running `cargo +1.99.1 fmt --all` and the installer's
`--refresh-format-hash`, run:

```bash
set -e
git diff --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit-auto
cargo +1.99.1 clippy --release --locked -p hybit-auto -- -D warnings
cargo +1.99.1 clippy --release --locked -p hybit --example fem_bench -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example fem_bench
```

Four additional focused unit tests check an actual full local-Cholesky
admission, exact Jacobi continuation after rejection, clone-budget skip,
and default/incompatible policy gates. They **must pass on WSL** before A2
is treated as accepted. Rust has not been built in the packaging environment.

## Real FEM pilot

Use the already SHA-checked C1-G2 n144 corpus:
`~/mtx/c1g_g2_hard_z32Jkd`, cases `island_shell`, `multi_patch`,
`rotated_anisotropy`. Keep `tol=1e-10`, `max-iters=600`, CSR, 1 CPU thread,
the same selector and region limits, and the existing Jacobi control run.

Example command (insert matching `--matrix`, `--rhs`, `--xref`):

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=10s 120s \
  target/release/examples/fem_bench \
    --matrix "$BASE.mtx" --rhs "$BASE.rhs.txt" --xref "$BASE.xref.txt" \
    --tol 1e-10 --max-iters 600 --backend csr \
    --probe-iters 12 --monitor-iters 64 --monitor-rate-limit 0.99 \
    --max-region 128 --max-regions 8 --overlap 1 \
    --factor-budget-mib 64 --max-escalations 1 \
    --shadow-trial-iters 32 --shadow-accept-ratio 0.8 \
    --shadow-max-mib 16 --skip-plain
```

Capture `C1G3A2_SHADOW|...` plus `status`, `iterations`, `verified residual`,
`relative x error`, `total wall`, elapsed time and peak RSS. Save full logs,
`git_head.txt`, `git_status.txt`, input SHA256, source SHA256 and `/usr/bin/time`
output in a new `~/c1h11_evidence/c1g3_a2_...` directory. A shell process
exit code of zero is **not** numerical convergence; enforce `status=Converged`
and independent true-residual tolerance in evidence intake.

In particular, the n144 `island_shell` legacy hybrid was worse than Jacobi
(600-step verified residual `1.131198e-6` vs `7.749079e-9`; 1200-step
convergence 898 vs 637). A2 must demonstrate an actual `Rejected` or
`Accepted` decision from matched branch data before concluding anything.
Even a successful short shadow can later regress; compare full-solve
iterations, error, runtime and memory against both original arms.

## Scope boundary

This patch does **not** enable graph coarse space automatically, change the
default local-region selector, introduce a new Krylov method, or remove A1.
Production code is modified only behind an explicit experimental option;
default behavior must remain byte-for-byte numerically compatible.

## WSL numerical acceptance and limitations

The 2026-11-06 Rust and numerical acceptance was completed.
All 19 n144 numerical runs exhausted 600 iterations, and short
horizon shadow admission regressed badly in island_shell.
See [C1-G3-A2 shadow horizon evidence](C1G3_A2_SHADOW_HORIZON_EVIDENCE.md)
for the verified results, source hashes and explicit non-promotion.
