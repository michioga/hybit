# C1-F: Opt-in serial SPD PCG kernel timing (experimental)

## Scope

This diagnostic **does not change PCG arithmetic** or enable any new Hybrid
escalation policy. `--profile-kernels` instruments only the legacy serial
SPD PCG recurrence (`ExecutionPolicy::Auto` or `Cpu`); GeneralSquare FGMRES,
resident CPU/Rayon paths and structural solves are outside its scope.

It reports two phases: `base` (Jacobi or explicitly enabled algebraic coarse)
and `hybrid` (selective local Cholesky, including learned/prepared reuse).
Each phase reports `starts` (fresh PCG sessions), `spmv_calls` and seconds,
`preconditioner_calls` and seconds. The baseline startup SpMV and initial
preconditioning count as kernel calls. Successful convergence may skip the
last preconditioner application. Timing uses `std::time::Instant` and adds
small overhead only when enabled; OFF uses direct existing calls.

**Not separately measured**: vector dot/AXPY/norm costs, operator/preconditioner
internal subkernels, true residual monitoring, residual certification,
profile overhead, or lost conjugacy/restart-induced iteration inflation.
`solve_seconds - measured kernel seconds` is *unattributed*, not a vector
kernel metric. Other reported wall-time fields can overlap and must not be
summed indiscriminately. There is no SPD certification by the benchmark.

## Two-arm smoke test (single thread)

```bash
cd ~/hybit
cargo build --release --locked -p hybit --example fem_bench
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 target/release/examples/fem_bench \
  --matrix ~/mtx/thermal1.mtx --tol 1e-8 --max-iters 3000 \
  --probe-iters 12 --monitor-iters 64 --monitor-rate-limit 0.99 \
  --monitor-only --profile-kernels --max-region 128 \
  --max-escalations 1 --skip-plain
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 target/release/examples/fem_bench \
  --matrix ~/mtx/thermal1.mtx --tol 1e-8 --max-iters 3000 \
  --probe-iters 12 --monitor-iters 64 --monitor-rate-limit 0.99 \
  --profile-kernels --max-region 128 --max-escalations 1 --skip-plain
```

Both runs must independently certify the original CSR residual <= 1e-8.
Observe should show base PCG starts=1, hybrid starts=0, and 1286 iterations;
Hybrid should show at least one hybrid start/one escalation, and around 1305
iterations (the older, unprofiled evidence). This is a *sanity expectation*
for the same matrix and controls, not a correctness assertion for arbitrary
matrices. Compare corresponding `kernel ...` lines, timings, and residuals.
The profiler is observational, not a production policy. Run multiple
AB/BA pairs with separate process starts to measure its own overhead.

## Validation gate

```bash
python3 -B -m unittest discover -s benchmarks/scripts -p 'test_fem_*.py' -v
cargo fmt --all -- --check
cargo test --workspace --locked
cargo test --release --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
sha256sum --check --strict --quiet SOURCE_SHA256.txt
```

Do not commit or push until WSL2's actual Release build and numerical
comparison both pass. Avoid upgrading this experimental observation to an
unvalidated automatic cost selector.
