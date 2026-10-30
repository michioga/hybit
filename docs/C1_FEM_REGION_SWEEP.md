# C1-A — repeatable WSL2 FEM region-size evidence

This is an **experimental benchmark harness**, not a production auto-selector
or proof of an AMG speedup. It runs the existing `fem_bench` example in new
Linux processes, with fixed tolerance/factor budget, randomized region-size
order, and explicit warmups. It records failures and zero-escalation cases.

## Build and self-test on WSL2 Ubuntu

From the repository root on `develop/0.10.0`:

```bash
python3 -m unittest discover -s benchmarks/scripts -p 'test_fem_region_sweep.py' -v
cargo build --release --locked -p hybit --example fem_bench
```

Short smoke (creates a **new** directory; never overwrites an earlier run):

```bash
python3 benchmarks/scripts/fem_region_sweep.py \
  --matrix benchmarks/data/poisson5.mtx \
  --output-dir "$HOME/hybit-c1-poisson5-$(date +%Y%m%d-%H%M%S)" \
  --regions 64,128 --warmups 1 --repetitions 2
```

Real-FEM pilot (start with the **existing, experimentally used** `thermal1`;
symmetry and positive diagonal do **not** independently establish SPD):

```bash
python3 benchmarks/scripts/fem_region_sweep.py \
  --matrix "$HOME/mtx/thermal1.mtx" \
  --output-dir "$HOME/hybit-c1-thermal1-$(date +%Y%m%d-%H%M%S)" \
  --regions 64,128,256,512 \
  --factor-budget-mib 64 --max-regions 8 \
  --tol 1e-8 --max-iters 3000 --probe-iters 12 \
  --threads 1 --warmups 3 --repetitions 9 --seed 20261028
```

Optional `--cpus 0-3` pins the runner **and inherited children** to those
logical CPU IDs, *if permitted* by the WSL scheduler. Do not assume that
0–3 are physical cores without separately validating topology. By default,
affinity is inherited and recorded, not enforced. `--skip-plain` omits the
Jacobi reference where only the Hybrid pathway is of interest.

## Outputs and numerical gate

Each output directory contains:

- `metadata.json`: matrix/RHS/executable SHA-256, source HEAD and dirty
  indicator, compiler, kernel/CPU/affinity/threads, randomized seed, exact
  command template, warmups/trials, budget and measurement boundaries.
- `results.csv` / `results.jsonl`: every trial **including warmups, process
  errors, timeouts, non-finite values and non-convergence**; separate
  `hybit_pass`/`plain_pass` and overall `paired_pass`.
- `logs/*.stdout.log` and `logs/*.stderr.log`: raw per-process output.
- `summary.json`: valid measured-trial median internal HyBIT wall time,
  process wall time and child peak RSS; count of actually escalated trials.

The gate accepts each solver only when it reports `Converged`, its printed
independently verified CSR relative residual is finite and within the
requested tolerance (allowing the example's six-digit print rounding), and
the child exited successfully. HyBIT additionally must report finite factor
memory within its cap. Forward error is **recorded** for manufactured RHS but
is not an SPD proof and is not silently turned into a generic acceptance
threshold. An external physical RHS has no known manufactured solution.

`wait4` reports **per-child Linux peak RSS in KiB**; this is process RSS,
not only HyBIT's internal factor bytes. `process_wall_ms` includes process
creation and Matrix Market load (as well as both solvers unless
`--skip-plain`); OS disk cache may already be warm. `hybit_wall_ms` is the
example's internal elapsed solve call and excludes matrix load and the plain
baseline. Plain always executes first when included; paired internal timing
can therefore be sensitive to cache/order effects. Such differences require
an additional controlled crossover before performance claims.

If **all `escalated_trials` are zero**, changing `max_region` is not a
meaningful test of local Cholesky economics. Report the sweep as a valuable
`no-escalation` diagnosis, **not** a win or loss for region size. This applies
to the observed `thermal1` default configuration (1,286 Jacobi iterations,
zero escalations) until tested otherwise.

This first C1-A harness is **one RHS per fresh process**. It does *not* yet
measure prepared five-RHS amortization, true cold-disk I/O, native external
PETSc/hypre AMG, or multi-node MPI. Those remain subsequent C1/C2/C3 tasks. Do not
interpret a passing PCG solve as an independent SPD certificate; obtain a
matrix-class-specific validation for performance conclusions.

## Minimal validation and rollback

```bash
python3 -m unittest discover -s benchmarks/scripts -p 'test_fem_region_sweep.py' -v
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
sha256sum --check --strict --quiet SOURCE_SHA256.txt
```

The runner adds only stdlib Python scripts and documentation; it does not
modify the Rust solver, any ABI or MPI wire format, or policy defaults.
Rollback removes this isolated C1-A addition; preserve C0 at the existing
`747ffa2` commit and do not reset the user's checkout or published 0.9.0.

## C1-B historical prototype — original-CSR true-residual progress monitor

The WSL `thermal1` C1-A trial converged in 1,286 Jacobi-PCG iterations for
every region size, with **no** local-Cholesky escalation. C1-B tests an
additional diagnostic boundary without changing the production default:

- `HybridOptions::progress_monitor_interval = 0` by default. `0` disables the
  monitor entirely. `fem_bench --monitor-iters N` and the sweep runner's
  `--monitor-iters N` explicitly enable it for positive `N`.
- If the initial `--probe-iters` step already requests escalation, the
  existing direct-region controller runs unchanged. Otherwise, after the
  probe, PCG continues **without restarting** for blocks of `N` iterations.
- An original-CSR `||b-Ax||₂` norm is evaluated after the probe, then after
  each complete block while further iterations remain. Escalation is
  considered when `r_new/r_old > escalation_residual_ratio` (default 0.5).
  An actual Cholesky factor and PCG restart still require valid regions and
  the existing memory cap. The monitor does **not** guarantee selection.
- `progress_monitor_checks` counts original-CSR norm measurements **including
  the post-probe baseline**, and `progress_monitor_trigger_iteration` stores
  the absolute PCG iteration at the first later trigger, if any. The benchmark
  emits `monitor checks` and `monitor trigger`, which the sweep records as
  `hybit_monitor_checks` and `hybit_monitor_trigger_iteration`.

**C1-C supersedes the C1-B post-probe trigger above.** The same 0.5 ratio
compares different numbers of iterations: as a per-step rate it corresponds to
0.989228 for 64 steps, 0.994599 for 128, 0.997296 for 256, and 0.998647
for 512. This was not interval-invariant. Do not treat the C1-B threshold
as a validated automatic selector. Extra CSR SpMV and local-factor work
must be included in a time-to-solution decision.

Example two-stage WSL2 experiment (first a low-cost diagnostic, then a
repeated sweep only if monitoring actually triggers):

```bash
cargo run --release --locked -p hybit --example fem_bench -- \
  --matrix "$HOME/mtx/thermal1.mtx" --tol 1e-8 --max-iters 3000 \
  --probe-iters 12 --monitor-iters 64 --max-region 128 --skip-plain

python3 benchmarks/scripts/fem_region_sweep.py \
  --matrix "$HOME/mtx/thermal1.mtx" \
  --output-dir "$HOME/hybit-c1b-thermal1-$(date +%Y%m%d-%H%M%S)" \
  --regions 64,128,256,512 --monitor-iters 64 \
  --factor-budget-mib 64 --max-regions 8 --tol 1e-8 \
  --max-iters 3000 --probe-iters 12 --threads 1 \
  --warmups 3 --repetitions 9 --seed 20261028
```

Keep the already-pushed `4ceaa6a` C1-A output as the unmonitored control;
`--monitor-iters 0` must also remain a valid new-build control. Neither a
trigger nor a shorter iteration count by itself proves improvement: compare
verified residual, actual local factors, elapsed time, and child peak RSS.
This C1-B patch requires WSL2 validation **before** a checkpoint commit.

## C1-C candidate — comparable rate telemetry, observation-first

The user-run WSL2 `thermal1` C1-B observations (single runs, `max-region=128`):

| Interval / max escalations | Escalations | Iterations | Total wall |
| --- | ---: | ---: | ---: |
| off / 3 | 0 | 1286 | 934.642 ms |
| 64 / 3 | 3 | 1321 | 1356.150 ms |
| 128 / 3 | 3 | 1451 | 1431.641 ms |
| 64 / 1 | 1 | 1305 | 1097.294 ms |
| 256 / 1 | 0 | 1286 | 914.359 ms |
| 512 / 1 | 0 | 1286 | 928.794 ms |

All runs reported `Converged` and independently verified residual `<= 1e-8`.
These timings are **not** repeated randomized evidence; the two apparent
faster-than-baseline runs have no certified speedup. The evidence rejects
automatic adoption of the C1-B window-ratio trigger on this matrix.

C1-C introduces opt-in, interval-comparable diagnostics:

- `--monitor-iters N` selects the original-CSR diagnostic sampling interval.
- `--monitor-only` **disables all local-direct escalation**, including the
  initial probe's escalation, while keeping the same uninterrupted base PCG.
- Each sample reports its total PCG iteration, true `||b-Ax||₂`, and the
  geometric mean residual ratio per actual iteration since the last sample:
  `exp((ln(r_new)-ln(r_previous))/delta_iterations)`. Values near or above 1
  are slow/growing. This is a *diagnostic*, not evidence that local Cholesky
  will help. Samples are available as `SolveReport.progress_monitor_samples`.
- Without `--monitor-rate-limit`, post-probe monitoring **only observes**.
  To authorize experimental post-probe local escalation, opt in explicitly
  with `--monitor-rate-limit R` where `0<R<1`. If a checkpoint's per-step
  ratio is greater than `R`, it requests the existing budgeted escalation.
  The initial probe still uses its original policy unless `--monitor-only`
  is given. `--max-escalations 1` remains advisable in pilot comparisons.
- The sweep collects monitor rate summary columns and accepts
  `--max-escalations` / `--stage-iters`, preserving every raw sample in its
  per-invocation stdout log. No C ABI or default public routing changes.

**First WSL C1-C observational probe:**

```bash
cargo build --release --locked -p hybit --example fem_bench

OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  target/release/examples/fem_bench \
  --matrix "$HOME/mtx/thermal1.mtx" --tol 1e-8 --max-iters 3000 \
  --probe-iters 12 --monitor-iters 64 --monitor-only \
  --monitor-rate-limit 0.99 --max-region 128 --skip-plain

OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  target/release/examples/fem_bench \
  --matrix "$HOME/mtx/thermal1.mtx" --tol 1e-8 --max-iters 3000 \
  --probe-iters 12 --monitor-iters 256 --monitor-only \
  --monitor-rate-limit 0.99 --max-region 128 --skip-plain
```

The rate threshold `0.99` is a **hypothesis to inspect**, not a production
default. Do not adopt a new selector from `thermal1` alone.

### 2026-10-30 WSL2 C1-C observation results

The user applied C1-C to the still-uncommitted C1-B tree at `4ceaa6a`.
Standard WSL2 Debug/Release workspace tests (including thin-LTO Release),
strict Clippy, rustfmt, Python 9/9, full Source SHA-256 and the two real
`thermal1` observations all **PASS**. These observations used manufactured
`b=A*1`, `tol=1e-8`, one thread, `--monitor-only`, hypothetical
`--monitor-rate-limit 0.99`, and `max-region=128`:

| Interval | True-residual checks | Candidate samples | Escalations | PCG iterations | True relative residual | Diagnostics | Total wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 20 | 10 | 0 | 1286 | 9.840383e-9 | 11.731 ms | 903.841 ms |
| 256 | 5 | 1 | 0 | 1286 | 9.840383e-9 | 3.423 ms | 911.766 ms |

Exactly matching original-CSR norms at shared checkpoints (268, 524, 780,
1036 iterations) confirm that the unmodified PCG recurrence was preserved.
The early 64-step windows around 140–460 iterations were comparatively
slow, but the following window from 460 to 524 reduced the true norm from
`3.582280e-2` to `3.656073e-3` (roughly 9.8x). Another slow interval
around 780–972 iterations likewise did not prevent eventual convergence.
The longer 256-step window smooths over those transient slow phases: only
the observation at iteration 1036 exceeded `0.99` per iteration.

**Interpretation:** the normalized geometric rate makes measurements
comparable across intervals, but the *actual average rate* can differ with
window length on nonstationary PCG trajectories. A candidate flag is not
evidence of locally remediable difficulty. Timings are single-run diagnostics;
no certified speedup, AMG comparison, or validated autonomous policy.
No branch/tag/release/publish action is implied. The next feature candidate
must explicitly measure local-factor cost and check for a net benefit before
adopting a stronger preconditioner.

## C1-D candidate — randomized paired control for local-direct economics

**Hypothesis:** an observed slow residual window does not imply that the
local-Cholesky escalation will repay its analysis, factor and application
costs. Compare it against **uninterrupted Jacobi-PCG on the same input**.
This is evidence collection, **not** an autonomous production decision.

The existing sweep now accepts `--paired-observe-control`. It requires
`--monitor-iters N`, `--monitor-rate-limit R` and `--skip-plain`, and forbids
an explicit `--monitor-only` (the runner assigns that flag by arm). In every
warmup and measured region/round, two **separate processes** run in randomized
order:

1. `observe`: `--monitor-only` plus the specified rate threshold; records
   the candidate flags but never builds a local factor.
2. `escalate`: otherwise identical arguments, without `--monitor-only`;
   allows the *experimental* budgeted local direct path.

Both arms use the same matrix, RHS, tolerance, iteration cap, backend, factor
budget and run environment. Their respective raw logs and SHA-identified
inputs are preserved. A comparison is **informative** only when both arms
pass original-CSR true-residual and budget checks, observation has zero
escalations, and treatment actually builds at least one local correction.
An early breakdown, numerical failure or zero-escalation pair is retained
but cannot count as evidence of local-factor effectiveness.

Run a quick WSL2 pilot first:

```bash
python3 -m unittest discover -s benchmarks/scripts -p 'test_fem_region_sweep.py' -v
cargo build --release --locked -p hybit --example fem_bench

python3 benchmarks/scripts/fem_region_sweep.py \
  --matrix "$HOME/mtx/thermal1.mtx" \
  --output-dir "$HOME/hybit-c1d-pilot-$(date +%Y%m%d-%H%M%S)" \
  --regions 128 --warmups 1 --repetitions 2 \
  --monitor-iters 64 --monitor-rate-limit 0.99 \
  --max-escalations 1 --factor-budget-mib 64 \
  --tol 1e-8 --max-iters 3000 --threads 1 \
  --paired-observe-control --skip-plain
```

If the pilot shows informative escalations, repeat on the held-out
**independently validated SPD** FEM corpus with `--regions 64,128,256,512`,
`--warmups 3 --repetitions 9`, fixed CPU affinity (when available) and
other documented settings. Select a new output directory for each sweep.
Do **not** choose a favored rate threshold or matrix based only on the
same training corpus and then claim held-out performance.

The v2 `metadata.json` records the paired-arm rule. Each CSV/JSONL row
includes `arm` and `pair_id`. For each region, `summary.json` adds:

- `complete_pairs`, `numerically_valid_pairs` and
  `informative_escalation_pairs` (the denominator for any speed claim);
- `treatment_faster_pairs`, paired median signed HyBIT-wall savings
  (observe minus escalate), and relative percentage gains;
- paired median peak child-RSS and iteration changes (escalate minus
  observe). These are *paired descriptive medians*, not confidence bounds.

**Evaluation:** time savings must exceed noise and monitor/factor costs,
while true residual and peak memory remain acceptable. Negative savings
are regressions and stay in the result. The child `process_wall_ms` includes
process overhead; HyBIT internal wall excludes Matrix Market load. Neither
includes amortized prepared multi-RHS costs. No causal benefit can be
attributed to the region-size setting if all informative counts are zero.
The observation arm itself pays true-residual monitoring overhead: even a
paired treatment win is **not** proof of a win over the unmonitored
production Jacobi policy (`--monitor-iters 0`). Measure that separately
before recommending any production escalation or monitor default.
The manufactured RHS does not independently certify `thermal1` as SPD.

**Rollback:** do not enable a new runtime policy; remove only this opt-in
runner mode and its matching tests/docs if the comparison is unreliable.
No Rust solver, C ABI, default threshold or published release changes.
