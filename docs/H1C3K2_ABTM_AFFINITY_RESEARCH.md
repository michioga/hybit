# H1-C3K2 research: ABTM affinity and constrained multilevel partitioning

## Source and decision register (2026-10-18)

Base: `develop/0.9.0` HEAD `f7c0c4b`; Windows MSVC/LLVM23.1.3/MS-MPI environment is canonical at `tools/env/Enter-HybitMsMpi.ps1`.

| External prior work / source | Concrete implementation insight | HyBIT adoption gate |
|---|---|---|
| KaMinPar, Gottesbueren et al., Deep Multilevel Graph Partitioning (ESA 2021), MIT repository https://github.com/KaHIP/KaMinPar | Size-constrained label propagation for coarsening / uncoarsening, strict balance, optional FM/flow refinement | Compare constrained aggregate sizes, actual cut/ghost/imbalance and prep-cost; external reference first, not vendored |
| KaHIP https://github.com/KaHIP/KaHIP | Multilevel framework, refinement and partition-quality oracle | Compare cut and balance for representative real FEM graphs; note unconstrained arbitrary owner IDs are *not* MPI `ContiguousPartition` |
| Meyerhenke, Sanders, Schulz, size-constrained graph clustering, DOI 10.1007/s10732-016-9315-8 | Cluster-size cap avoids oversized coarse vertices; pair-only matching may under-coarsen irregular graphs | Opt-in only after existing G8-A6 pairing has baseline |
| Karypis/Kumar, multilevel k-way partitioning DOI 10.1006/jpdc.1997.1404 | Balance-aware matching and inexpensive refinement matter more than local similarity score alone | End-to-end partition quality, not only affinity microbench |
| Ron/Safro/Brandt, DOI 10.1137/100791142; code https://github.com/safrolab/AMG-Graph-Hierarchy | Algebraic distance from numerical relaxation test vectors | Deferred until spectral/near-nullspace and numeric setup costs can be measured; not pure ABTM bitwise |
| Advanced Coarsening Schemes for Graph Partitioning https://www.eecis.udel.edu/~isafro/papers/advcoarseningpart.pdf | Weight-normalized ratings can beat unnormalized common-neighbor scores | Compare deterministic normalized/Jaccard variants on fixed topology |

## What existing HyBIT does, and what it does NOT

`crates/hybit-distributed/src/multilevel.rs::coarsen_once_cached` uses cached, deduplicated adjacency lists, a per-node `neighbor_stamps` array and exact common-neighbor counts. `AbtmDualTopology` constructs the directed union adjacency, but G8-A6 matching does **not** currently evaluate every candidate using ABTM mask AND+popcount. Coarse aggregate **weights and balance** are already present; multi-level uncoarsening and cut refinement already exist. Replacing those implementations blindly is unnecessary.

`field_graph_contiguous_partition` returns contiguous ownership offsets. Existing MPI ingestion and Halo require contiguous owned rows. Unconstrained k-way FM moves and external partition IDs cannot be substituted without a new owner-based MPI distribution and halo contract.

## New experiment (EXAMPLE ONLY; zero solver behavior changes)

`crates/hybit-distributed/examples/abtm_affinity_ab.rs` loads a real Matrix Market matrix via the existing `hybit_matrix` loader. It constructs the exact `A union A^T` deduplicated, diagonal-free adjacency corresponding to G8-A6, then builds sparse 64-bit word masks once. It compares **every directed adjacency candidate's** common-neighbor score between the current stamp array and word-wise AND+POPCNT, and compares the *entire deterministic pair-coarsening assignment*; any mismatch is fatal. Alternating paired runs report median elapsed time to reduce order bias. The ABTM row word prepack time is reported separately to avoid free-prep bias. CPU results must come from the actual Windows machine.

```
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1
cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --all-targets -- -D warnings
cargo run --release --locked -p hybit-distributed --example abtm_affinity_ab -- D:\Work\thermal1.mtx 7
```

**Gate 1:** exact score equality on every tested pair and identical matching; no unexpected allocations/overflows. **Gate 2:** measured pair-selection median and extra word-packing cost; run against `thermal1`, plus irregular cfd/bone matrices when applicable. **Gate 3:** if competitive, integrate an explicit, default-off affinity mode in `coarsen_once_cached` and compare full coarse-level setup, partition quality and later real MPI-PCG. Otherwise retain the existing stamp method. **Gate 4:** assess size-constrained label propagation, then balance-safe contiguous boundary refinements. Do not change MPI-PCG default route or claim solver speedup based on this standalone probe.

For the actual K2 integration, preserve K1 `thermal1.mtx` experimental SPD-unverified warning, relative tolerance `1e-11`, manufactured truth/RHS, Jacobi and 1/2/4/8 MS-MPI launches. Validate true residual and max absolute error, Ghost DOFs, cut arcs, load and total acquisition/prep/solve time. No physical field DOF metadata can be inferred from Matrix Market alone.

## 2026-10-18 H1-C3K2 A/B native results and MSRV fix

Native Windows x64/MSVC release diagnostic, `thermal1.mtx`, 7 alternating paired repetitions, 82,654 DOFs, 574,458 CSR nnz, input entries 328,556:

| Measurement | Observed |
|---|---:|
| Matrix read | 68.1820 ms |
| Dual undirected adjacency preparation | 37.7780 ms |
| Sparse bitmap preparation | 6.6234 ms |
| Directed neighbor-pair scores checked | 491,804 |
| Resulting coarse vertices, one pass | 44,293 |
| Exact scores / matching | PASS / PASS |
| Existing stamped pair matching median | 4.0511 ms |
| Bitmap AND+POPCNT pair matching median | 7.4006 ms |
| Bitmap pack + pair selection | 14.0240 ms |
| Bitmap pair speed relative to stamp | 0.5474x |

**Decision:** keep the existing stamped affinity method as the default. The bitmap pair-selection time is about 1.83x the stamped time on this input, or 3.46x when adding bitmap preparation. No whole-partitioner / MPI-PCG speedup is established. `bitmap_payload_est_bytes=4126688` excludes `Vec` headers/capacities, existing adjacency, and runtime peak RSS; it is NOT measured total memory.

The standalone example compiled and ran, but Clippy with `-D warnings` failed because `Option::is_none_or` is only stable in Rust 1.82 while the workspace declares MSRV 1.73. Replace it with equivalent MSRV-compatible `match`, preserving comparison and exact results; re-run fmt, Clippy, and release diagnostic before commit. This is a source compatibility correction, not an algorithm change.

Further candidates: retain stamped matching; on separate real-matrix A/B trials investigate degree-aware *hybrid* score evaluation only if the mask packing can be reused, constrained multi-vertex aggregation (size-constrained label propagation) and balance-safe contiguous boundary refinement. Compare full coarse graph build, cut/ghost load, and wall time; isolate packed-mask cost vs preexisting adjacency. Never make ABTM Graph production default from this diagnostic alone.
