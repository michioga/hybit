# H1-C3K2-B — ABTM as an explicit partition benchmark competitor

Base: `develop/0.9.0`, `03915d4` (the real FEM Equal/ScalarGraph MPI-PCG probe).

## Purpose and scope

Do not optimize **for** ABTM. Benchmark its three existing ownership algorithms alongside the existing strong baselines using exactly the same CSR matrix and number of ranks:

| Label | Implementation | Ownership constraint | Default load constraint |
| --- | --- | --- | --- |
| `equal` | `ContiguousPartition::balanced` | contiguous | exactly equal +/- 1 DOF |
| `scalar_graph_contiguous` | `field_graph_contiguous_partition` with synthetic one-component blocks | contiguous | at most worse of Equal max or 115% of mean DOF count |
| `abtm_region_growth` | `abtm_region_grow_partition` | potentially noncontiguous | exactly equal +/- 1 DOF |
| `abtm_multisource` | `abtm_balanced_multisource_partition` | potentially noncontiguous | exactly equal +/- 1 DOF |
| `abtm_multilevel` | `abtm_multilevel_partition` with its version-controlled defaults | potentially noncontiguous | 3% fine DOF imbalance budget (default) |

External competitors such as METIS, SCOTCH or KaMinPar should join the **same quality schema** after an independently verified native tool interface exists. Never invent their benchmark numbers. Compare load-constrained families explicitly: a 3% relaxed noncontiguous partition is **not** a strictly equal partition.

This new example is a **single-process partition quality/construction benchmark**, not a native MPI-PCG benchmark. Only `equal` and `scalar_graph_contiguous` currently have rank-zero distribution plus real MPI-PCG results. `ABTM region/multisource/multilevel` **do not yet** have live MPI end-to-end timings here. No new automatic routing is enabled.

`read_matrix_market` loads CSR once, outside per-candidate timer. `partition_ms` includes all algorithm-specific ABTM or synthetic field-layout preparation and owner-vector conversion, but excludes parser and **separately charged** `telemetry_ms`. `communication_volume` is exact deduplicated remote DOF per receiver under the owner labels, *not measured network bytes/time*. `cut_nnz` counts directed off-rank CSR references. No process peak RSS measurement is claimed. `abtm_metadata_bytes` is the available dual-topology diagnostic, not total memory; multilevel does not expose this field and is blank. `configured_imbalance_per_mille` records the algorithms' constraints; other algorithms may also be limited by implementation-internal caps.

All repeated candidate ownership vectors must match bit-for-bit across trials, or the benchmark fails. Every rank must own at least one DOF. Source Matrix Market coefficients are not modified, but this probe does not calculate a solution or prove global SPD.

## Native Windows experiment

```powershell
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1
cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --all-targets -- -D warnings
cargo test --release --locked -p hybit-distributed --features mpi --lib
cargo run --release --locked -p hybit-distributed --example k2_partition_candidates -- D:\Work\thermal1.mtx D:\Work\k2_abtm_candidates_01.csv 3 2,4,8 16
```

The output CSV must not exist before invoking the example. A typical run has 3 repetitions x 3 rank counts x 5 methods = **45 candidate evaluations**. Real run duration may be substantial, especially for multilevel. Start with `1 4 16` instead of `3 2,4,8 16` if runtime is uncertain.

## Promotion criteria

1. Record per-rank cut, exact unique-ghost volume, peer counts and imbalance for each partition candidate (CSV quality telemetry is per-rank aggregate metrics, not a network trace).
2. Introduce a reversible symmetric permutation `P A P^T` to pack each noncontiguous owner set into a contiguous rank-owned CSR range. Preserve inverse DOF mapping for manufactured RHS, true residual and returned solution.
3. Then run **actual** MS-MPI PCG for Equal, ScalarGraph and each ABTM candidate, with matched tolerance, RHS, preconditioner and rank count. Time permutation/build, distribution, halo, PCG prep, solver and wall. Only this stage supports numerical/runtime comparisons.
4. Compare external partition algorithms using the same tolerance constraints and metric definitions; choose fastest **whole solver**, not a favored internal data structure.

Validation gates for the experiment are lighter than production promotion, but no candidate is promoted if it changes the mathematics or fails true residual and manufactured-solution checks.
