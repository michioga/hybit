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
PETSc/hypre AMG, or multi-node MPI. Those remain C1-B/C2/C3 tasks. Do not
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
