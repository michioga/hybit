# HyBIT 0.9 H1-C3C — conservative cost-aware partition selector

**Checkpoint status:** H1-C3C explicit Rust API validated locally on Windows (39 distributed tests, Clippy, source integrity, four real-MTX policy probes). **No H1-C3C native MPI consensus, end-to-end solver, or field-mixed real FEM validation** has been performed. No solver default, MPI transport or ABI changes.

## Why

The H1-C3B real-MTX tests (commit `5d0e919`) verified numerical equivalence, but optimizing ghost/cut counts did not consistently reduce SpMV time. `thermal1` (8 ranks, confirmation) Field 0.17155 ms vs Graph 0.20305 ms despite reduced ghosts; `nd3k` 4/8 rejected graph proposals after 124/196 ms symbolic preparation. Graph was the quickest of Equal/Field/Graph in only one of eight confirmed 4/8-rank cases. Thus the decision must be guarded by measured calibration and explicitly pay for preparation.

## API contract

`cost_aware_contiguous_partition(matrix, layout, ranks, options)` in `hybit-distributed` is an **explicit reference preparation API** that returns contiguous rank-owned DOFs with diagnostics. It is not attached to `hybit::solve`, PCG, FGMRES or MPI-owned production ingest.

1. Compute H1-C2 Field baseline and, **only if no field block is split**, Equal baseline. Use exact CSR HaloPlan telemetry to prefer Equal if it does not regress ghosts, cut nnz, peer count, maximum local nnz or maximum owned DOFs relative to Field. Otherwise use Field.
2. Skip ABTM/Graph entirely for 1 rank, zero expected SpMV calls, search radius zero, or absent calibration. Reject **before ABTM construction** if externally calibrated Graph SpMV is no faster than baseline or its predicted saving does not exceed the Graph preparation budget plus requested minimum net gain.
3. In explicitly profitable candidate cases only, reuse H1-C3 `field_graph_contiguous_partition` (ABTM dual symbolic topology). Recheck the exact Halo telemetry *against the selected Equal or Field baseline* and enforce absolute maximum owned-DOF and local-nnz imbalance caps (defaults 1.15, meaning 115% of mean). Reject changes outside the caps. Selected graph ownership is still contiguous and field-block aligned.
4. The reported local wall-clock Graph preparation time is diagnostic **only**. It is not an input to selection, to avoid different MPI ranks independently choosing different partitions due to timing noise.

The estimated cost model is `net_ns = expected_spmv_calls * (calibrated_baseline_spmv_ns - calibrated_graph_spmv_ns) - calibrated_graph_prepare_budget_ns`. All operands are nonnegative integers with `u128` arithmetic. It is a **model**, not a claim of execution-time improvement. The caller is responsible for calibration measured in representative, independent runs. For native distributed use, every MPI rank must receive identical calibration/options (or receive the same selected offsets via broadcast). Do **not** use each rank's private elapsed time for the decision.

The symbolic H1-C3 heuristic itself is unchanged: local search is not a global optimum and requires a replicated global CSR here. The initial fast path still scans exact CSR Halo telemetry but does not allocate/build ABTM's dual topology. Not a scalable MPI distributed repartitioner. General/nonsymmetric CSR is permitted; no SPD assumption is made for SpMV.

## Windows verification

Package applied to clean HEAD `5d0e919` and the following checks passed on Windows (user-reported logs):

```powershell
cargo fmt --all -- --check
cargo test --release --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi
cargo clippy --locked -p hybit-distributed -p hybit-matrix -p hybit -p hybit-ffi --all-targets -- -D warnings
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
git diff --cached --check
```

### Real-MTX probe evidence (2026-10-15)

All four probes below completed without errors. `source-integrity-gate.ps1`: **280 manifest entries / 278 verified hashes, PASS**. `cargo fmt --check`, `cargo test --release --locked` (`hybit-distributed` **39/39**, `hybit-matrix` **52/52**, `hybit-ffi` **5/5**, existing integration tests), `cargo clippy --all-targets -- -D warnings`, and staged diff check also PASS.

| Matrix | Ranks | Search radius | Expected calls | Decision | ABTM Graph attempted | Selected | Ghosts | Cut nnz | Notes |
|---|---:|---:|---:|---|---|---|---:|---:|---|
| `Goodwin_010` | 4 | 5 | 1,000 | `UnamortizedPreparation` | no | Field | 542 | 5,091 | Early exit before ABTM |
| `Goodwin_010` | 4 | 5 | 10,000 | `GraphSelected` | yes | Graph | 513 | 4,930 | `nnz_imbalance=1.1090`, cap 1.15 PASS; local graph prep 0.9573 ms |
| `nd3k` | 4 | 5 | 10,000 | `NoCalibratedSpeedup` | no | Equal | 24,147 | 1,786,056 | Early exit before ABTM |
| `Goodwin_010` | 4 | 32 | 10,000 | `LoadImbalanceLimit` | yes | Field | 542 | 5,091 | Candidate rejected by load cap; local graph prep 1.0385 ms |

In the radius=5 `Goodwin_010` case, the selected graph's partition offsets were `[0, 296, 590, 888, 1182]`, with ghosts reduced **542 → 513** and cut nnz **5091 → 4930**. The radius=32 case reverted to Field offsets `[0, 295, 591, 886, 1182]`; `attempted_graph=true` is intentional because the load cap is checked after the graph proposal has been built. Both are **local/non-MPI policy probes**: no native inter-rank agreement or MPI SpMV performance claim follows from these runs.

Example commands and expected key decisions:

```powershell
cargo run --release --locked -p hybit-distributed --example cost_field_partition_probe -- D:\Work\Goodwin_010.mtx 4 1000 8000 7600 1578200 5 1.15 1.15
# selected=Field reason=UnamortizedPreparation attempted_graph=false
cargo run --release --locked -p hybit-distributed --example cost_field_partition_probe -- D:\Work\Goodwin_010.mtx 4 10000 8000 7600 1578200 5 1.15 1.15
# selected=Graph reason=GraphSelected attempted_graph=true
cargo run --release --locked -p hybit-distributed --example cost_field_partition_probe -- D:\Work\Goodwin_010.mtx 4 10000 8000 7600 1578200 32 1.15 1.15
# selected=Field reason=LoadImbalanceLimit attempted_graph=true
cargo run --release --locked -p hybit-distributed --example cost_field_partition_probe -- D:\Work\nd3k.mtx 4 10000 663200 741200 124391600 5 1.15 1.20
# selected=Equal reason=NoCalibratedSpeedup attempted_graph=false
```

### Calibration limitations discovered during verification

The supplied Goodwin example uses **illustrative** `baseline_spmv_ns=8000`, `graph_spmv_ns=7600`, `graph_prepare_budget_ns=1578200` from the earlier H1-C3B **radius=32** run. But the H1-C3B confirmation measured **Equal=8000 ns and Field=7900 ns**, while the H1-C3C selector reports its baseline as **Field**, so the 8000 ns calibration is **not for its selected baseline**. Moreover radius=5 changes the graph partition and has **no independently measured MPI SpMV time**. The positive modeled net savings (`2,421,800 ns` at 10,000 calls) therefore represents only a hypothetical demonstration, **not a valid prospective speedup estimate for the selected partition**.

This API currently does not enforce calibration identity (matrix structure/version, rank count, selected baseline kind, search radius, platform, and measurement protocol). **Do not enable automatic routing or treat `GraphSelected` as performance validation** until calibration matching and native max-rank timing/consensus are added. On a real distributed run, all ranks must receive the same selected ownership offsets, not individually make decisions from rank-local timing.

## Deferred

- Multi-process consensus or root broadcast of selected offsets, and reliable global maximum-rank elapsed timings.
- Calibration provenance/identity validation, including chosen baseline, graph search radius, rank count, matrix fingerprint, build/hardware, and measurement protocol; mismatch must conservatively disable Graph.
- End-to-end PCG/FGMRES benchmark and independent randomized calibration/repetitions.
- Actual FEM field metadata from mesh and mixed nodes/constraints, not scalar synthetic blocks.
- Memory-budget-aware ABTM preparation or symbolic reuse, and sparse MPI-owned input without all-rank global CSR.
- A default public API policy only after end-to-end improvement is established.
