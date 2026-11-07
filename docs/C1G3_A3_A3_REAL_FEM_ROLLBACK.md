# C1-G3-A3-A3 — Experimental real-FEM shadow rollback integration

Status: **WSL validation pending**. Based on `develop/0.10.0` commit `bde0583`.

## Intent and non-goals

The C1-G3-A2 local-Cholesky shadow trial can admit a branch that looks better
over 32–64 PCG iterations but deteriorates at the 600-iteration horizon.
The `island_shell` n144 FEM case demonstrated precisely that limitation:
`shadow=Accepted` at trigger 396 for a 32-step trial, while its final true
relative residual was `1.131198e-6` versus `7.749079e-9` for live Jacobi.

This patch adds a **second, independently verified comparison** after an
admitted candidate has run for a bounded watchdog window. The original Jacobi
PCG scalar recurrence, five work vectors, and solution are saved at the actual
fork. The original recurrence is replayed **without restarting PCG** to the
same selected-path logical horizon. If the candidate is worse, breaks down,
reports false convergence, or produces non-finite values, a valid baseline
checkpoint replaces the candidate state exactly.

The feature is experimental and **disabled by default**. It does not establish
a general, dynamically tuned autonomous admission policy, and does not change
the non-shadow PCG controller or FGMRES policies.

## New opt-in

- `HybridOptions::shadow_watchdog_iterations` defaults to `0` (A2 behavior).
- `fem_bench --shadow-watchdog-iters N` activates the second comparison only
  when `--shadow-trial-iters` is also positive.
- `shadow_max_extra_bytes` covers the **peak two cloned states** while the A2
  comparison and rollback checkpoint overlap: `2 * (workspace.bytes() + 8*n)`.
  Factor memory, ABTM metadata, and transient factorization allocations are
  separately accounted as before.
- A budget shortfall skips the candidate before local factorization. The
  original Jacobi PCG recurrence continues, without mutation or restart.
- Original-CSR norms are authoritative; a recursive `Converged` status with
  true residual above the target cannot be selected via rollback.
- `HybridShadowDecision::RolledBack` and watchdog-specific decision reasons
  distinguish initial admission from the final selected preconditioner.

**Horizon limitation:** the watchdog is one comparison window, not continuous
monitoring. Setting `--shadow-watchdog-iters` below the *remaining* iteration
budget only protects the immediate window. For the full-horizon experiment,
set it at least to `--max-iters` so it clamps to all remaining iterations.

## Accounting

The public `SolveReport.iterations` remains the number of iterations on the
**selected** trajectory: probe/monitor + chosen trial/watchdog or exact Jacobi
replay + any continuation. Speculative A2 branch work is in `trial_seconds`.
The new `HybridShadowTrialReport` fields expose watchdog candidate iterations,
additional baseline replay iterations, their independently checked true
residuals, peak clone bytes, and watchdog elapsed seconds. `watchdog_seconds`
is charged to `restart_seconds` (and hence `solve_seconds`), **including**
baseline replay; `local_factor_seconds` remains separate. No claim is made
that the rollback is faster than running Jacobi alone.

The `C1G3A2_SHADOW|...` log keys and order remain unchanged for A2 comparison
regression. The optional `C1G3A3_WATCHDOG|...` line holds the extra data.

## WSL acceptance gate

Run from `~/hybit` on the guarded `develop/0.10.0` source. Preserve the
unrelated `benchmarks/scripts/__pycache__/` directory.

```bash
(
  set -e
  cargo +1.99.1 fmt --all -- --check
  sha256sum --check --strict --quiet SOURCE_SHA256.txt
  cargo +1.99.1 test --release --locked -p hybit-auto
  cargo +1.99.1 clippy --release --locked -p hybit-auto -- -D warnings
  cargo +1.99.1 clippy --release --locked -p hybit --example fem_bench -- -D warnings
  cargo +1.99.1 build --release --locked -p hybit --example fem_bench
)
```

Then compare `observe`, A2 `shadow=32` and guarded
`shadow=32, watchdog=600` on `island_shell`, and a 64-step guarded trial on
`multi_patch`. Use the same corpus and parameters as the A2 horizon evidence:
`--tol 1e-10 --max-iters 600 --backend csr --probe-iters 12
--monitor-iters 64 --monitor-rate-limit 0.99 --max-region 128
--max-regions 8 --overlap 1 --factor-budget-mib 64 --max-escalations 1
--shadow-accept-ratio 0.8 --shadow-max-mib 16 --skip-plain`.

Example for the `island_shell` candidate:

```bash
BASE="$HOME/mtx/c1g_g2_hard_z32Jkd/c1g_p1_island_shell_n144"
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  target/release/examples/fem_bench \
    --matrix "$BASE.mtx" --rhs "$BASE.rhs.txt" --xref "$BASE.xref.txt" \
    --tol 1e-10 --max-iters 600 --backend csr --probe-iters 12 \
    --monitor-iters 64 --monitor-rate-limit 0.99 \
    --max-region 128 --max-regions 8 --overlap 1 \
    --factor-budget-mib 64 --max-escalations 1 \
    --shadow-trial-iters 32 --shadow-accept-ratio 0.8 \
    --shadow-watchdog-iters 600 --shadow-max-mib 16 --skip-plain
```

**Predeclared expected checks**, not measured A3-A3 outcomes:

1. A2's admission telemetry must match A2 within numeric/log rounding.
2. The `island_shell` full-horizon guard should report `RolledBack` and
   recover the Jacobi `7.749079e-9` true residual and `1.721567e-7` x error
   seen in A2's observe/reference arms; if not, inspect exact PCG fork state.
3. The `multi_patch` 64-step candidate may remain accepted if its final CSR
   residual is still below the full-horizon baseline residual.
4. Selected iterations never exceed 600; speculative baseline replay is
   reported separately. `--shadow-watchdog-iters 0` reproduces A2 results.
5. Run a zero/insufficient clone budget case and confirm no local factor is
   allocated, and the live Jacobi recurrence remains unchanged.

No production-speedup claim or default-policy promotion until independently
repeated FEM evidence, memory measurements, and a passing CI run.
