# C1-G3-A1: isolated PCG shadow-checkpoint reference

Status: **experimental reference, not production controller behavior**.

## Objective and evidence boundary

The C1-G2 constructed SPD pilot identified a counterexample to unconditional
local-Cholesky promotion: for the n=144 `island_shell` matrix, Jacobi-PCG reached
the requested `1e-10` verified relative residual in 637 iterations, while the
locally promoted path required 898. At the fixed 600-iteration checkpoint, its
verified relative residual was `1.131198e-6` compared with `7.749079e-9`
for Jacobi, a roughly 146-fold deterioration. Local direct factors were not
proven numerically defective; promotion policy and convergence effectiveness
are distinct questions.

The standalone Rust example `crates/hybit-krylov/examples/c1g3_shadow_reference.rs`
probes only the **stateful PCG branch mechanics** needed by a future admission
controller. It does **not** reproduce HyBIT's region selection or the real FEM
counterexample. Its sample is a synthetic SPD *diagonal* operator with an exact
Jacobi candidate; it is not a Schwarz or local-Cholesky benchmark.

## Shadow trial procedure

1. Start the baseline PCG session and advance it to an unfinished checkpoint.
2. Clone the current solution, the `PcgWorkspace`, and the `PcgSession`
   (the session is already `Copy`). The candidate runs on these copies.
3. Start a **new** PCG recurrence for the candidate preconditioner and run
   at most `trial_iterations` iterations. The original baseline is untouched.
4. Continue the original baseline session for the same iteration allowance,
   without restarting or changing its search direction.
5. Recompute both true relative residuals independently using the operator;
   do not use recursive PCG residuals as the admission metric.
6. Admit a finite candidate only if it is sufficiently better than the
   baseline, the baseline has not already met its true target, the candidate
   did not break down, and its reported convergence is not false according
   to the independent true residual. Otherwise keep the live baseline state.

The reference uses an illustrative `candidate_accept_ratio=0.8`. This number
is **not validated for production** and must not become a default without
the full C1-G3 controller study. Candidate setup/factorization time is outside
this helper; the report counts both branch iteration totals to prevent
concealing speculative work. The iteration total excludes the candidate's
initialization SpMV and the independent residual-verification SpMVs; full
wall-time and SpMV accounting is necessary in A2. The caller is responsible
for restricting the trial allowance to the remaining solver budget.

## Memory and future integration

One candidate clone retains one `x` vector and the five `PcgWorkspace`
vectors: **approximately `6*n*sizeof(f64)` additional bytes**. For n=20,736
this is `995,328` bytes (`0.949 MiB`); for n=252,168 this is `12,104,064`
bytes (`11.543 MiB`). The actual peak includes candidate factorization,
matrix storage, diagnostic residual buffers and allocator overhead. The
existing `max_local_factor_bytes` limit does **not** cover these transients.

If the candidate wins, its session and workspace must be paired with its
preconditioner thereafter. If it loses, the base session remains valid because
the baseline never restarted. This is the key property tested here.

## Initial validation gates

After installation and `cargo +1.99.1 fmt --all`, refresh the source checksum
using the installer `--refresh-format-hash` mode. Then run:

```bash
git diff --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit-krylov --example c1g3_shadow_reference
cargo +1.99.1 clippy --release --locked -p hybit-krylov --example c1g3_shadow_reference -- -D warnings
cargo +1.99.1 run --release --locked -p hybit-krylov --example c1g3_shadow_reference
```

The example has five unit tests:

- A copied PCG checkpoint continues identically to the uninterrupted path.
- A clearly effective SPD candidate is admitted with an independently
  verified true residual.
- A restarted copy of the unchanged preconditioner is rejected without
  disturbing the exact original recurrence.
- A non-SPD candidate with nonpositive `r^T M^-1 r` is rejected fail-closed.
- An invalid trial budget is rejected without changing the solution.

Successful compilation/tests and a synthetic candidate acceptance are the
**A1 reference gate only**. They do not establish a real FEM rescue rate.

## Next checkpoint

For C1-G3-A2, integrate an explicitly opt-in candidate comparison with
HyBIT's actual local-region selection. Use the frozen C1-G2 matrices and
independent true residual/forward-error checks, preserve existing default
behavior, account for total speculative matvecs and factorization cost, and
verify memory caps. Only then compare coarse-space promotion against local
Cholesky. Do not silently assume that graph aggregation is ABTM-specific
or that a smaller residual implies a smaller forward error.
