# C1-D — Paired offline cost assessment for optional local Hybrid correction

**Not a production automatic policy.** This experimental WSL2 runner repeats
`fem_bench` under **two policies that use identical monitor parameters**:

* `observe`: `--monitor-only` (Jacobi-PCG continues unchanged, even when the
  monitor flags a candidate);
* `hybrid`: same run without `--monitor-only` (optional strengthening can occur).

The script alternates these in fresh processes in AB/BA pairs. It compares
`total wall` **inside HyBIT** (includes setup/solve but not Matrix Market load)
for each pair, and records process wall time, peak RSS, independently verified
original-CSR residual, iterations, escalation count, and local-factor bytes and
timings. Both arms receive the same matrix, RHS, executable, threads, tolerance,
monitor interval and threshold; the `CSR` and `jacobi-byte` selections are
explicit. `metadata.json` hashes the matrix, RHS (if given), and binary.

## Example: thermal1, monitor interval 64, at most one escalation

```bash
cd ~/hybit
python3 -B -m unittest discover -s benchmarks/scripts \
  -p 'test_fem_hybrid_crossover.py' -v
cargo build --release --locked -p hybit --example fem_bench
OUT="$HOME/hybit-c1d-thermal1-$(date +%Y%m%d-%H%M%S)"
python3 -B benchmarks/scripts/fem_hybrid_crossover.py \
  --matrix "$HOME/mtx/thermal1.mtx" --output-dir "$OUT" \
  --monitor-iters 64 --monitor-rate-limit 0.99 \
  --max-region 128 --max-escalations 1 \
  --tol 1e-8 --max-iters 3000 --threads 1 \
  --warmup-pairs 2 --pairs 9 --seed 20261030
python3 -m json.tool "$OUT/summary.json"
```

The runner keeps *all* cases including warmups/failures in CSV, strict JSONL,
and individual raw logs; it refuses to overwrite an existing output directory.
It exits nonzero for failed measured numerical gates. Result classification:

* `measured_regression_in_this_matrix`: complete numerical evidence, actual
  escalation in every measured pair, and the bootstrap 95% interval for paired
  *median* relative cost lies entirely **above +5%**;
* `potential_benefit_in_this_matrix`: the same checks but interval entirely
  **below -5%**;
* `mixed_or_no_activation`: at least one candidate never escalated;
* `insufficient_evidence`: missing/failed data or fewer than five pairs;
* `inconclusive`: otherwise.

These are **exploratory empirical descriptors**, not calibrated coverage or a
safe automatic solver switch. Residual behavior can be nonmonotone and timing
can be affected by OS cache, WSL scheduling and thermal drift. `b=A*1` is a
manufactured RHS, not a physical FEM load or proof that the matrix is SPD.
The experiment does not estimate what Jacobi would have done *after the same
trigger on the same execution*; it measures two fresh-run policies. It also
does not establish prepared five-RHS amortization, multi-node MPI scaling or
competitive AMG performance. Do not change `HybridOptions::default()` based on
one matrix.

## Safeguards

Run Python unit tests, Rust workspace tests, `cargo fmt`, Clippy, and the
`SOURCE_SHA256.txt` gate before commit. The script uses only the Python
standard library plus the existing `fem_region_sweep.py` module. It does
**not** modify Rust code, APIs, ABI, default policy, or MPI behavior.
