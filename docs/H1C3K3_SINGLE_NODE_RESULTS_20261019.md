# H1-C3K3 one-node native MS-MPI validation (2026-10-19)

**Status: validated experimental checkpoint; not a production distributed solver or multi-node result.**

- Repository: `michioga/hybit`, branch `develop/0.9.0`; base commit before K3: `e26e081` (K2 benchmark committed/pushed).
- Evidence: user-supplied Windows PowerShell console output on 2026-10-19 for K3 FIX2 Apply, and native **1/2/4/8-rank** invocations. Source CSV files exist on the user's Windows machine, but the K3 raw CSV contents were **not independently inspected** when authoring this report. Do not treat five 15-sample medians as five independent run-level replications.
- Host topology: **one physical node** running MS-MPI, Windows 11, x64 MSVC/LLVM/MS-MPI. All MPI ranks run on that same node; **no inter-node networking was tested**. Results must not be described as distributed-across-nodes scaling.
- Matrix: `D:\Work\thermal1.mtx`; global CSR `n=82,654`, expanded `nnz=574,458`. Matrix Market field labels are unavailable; scalar graph candidate uses synthetic 1-DOF blocks.
- Program: `mpi_k3_assignment_spmv.exe`; graph search radius `16`, warmups `3`, timed samples `15` per policy and rank count. Five policies: `equal`, `graph_contiguous`, `abtm_region_growth`, `abtm_multisource`, `abtm_multilevel`.
- Native test builds use `tools/env/Enter-HybitMsMpi.ps1` (MSVC, clang/libclang, MS-MPI); `-Apply` K3 FIX2 completed formatting, Clippy with `--features mpi --all-targets -- -D warnings`, 85/85 release library tests, example release build, source integrity 311 manifest entries / 309 hashes, and staged diff check.

## Native single-node measurements

Times are milliseconds. `partition_ms` includes rank-0 proposal and ownership-map broadcast; `setup_ms` measures MPI Halo setup. `spmv_ms` is the **median of each timed trial's maximum elapsed SpMV time across ranks**. `exchange_ms` similarly summarizes max-rank numeric Halo exchange (packing and MPI); it does not represent network transfer alone. Never add medians of independently reduced stages to reconstruct total time.

| MPI ranks | Policy | Partition ms | Halo setup ms | SpMV ms | Exchange ms | Cut nnz | Ghost DOFs | Directed peer relations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | equal | 0.175 | 22.800 | 0.5780 | 0.0127 | 0 | 0 | 0 |
| 1 | graph_contiguous | 31.469 | 22.795 | 0.5745 | 0.0130 | 0 | 0 | 0 |
| 1 | abtm_region_growth | 32.772 | 22.464 | 0.5785 | 0.0119 | 0 | 0 | 0 |
| 1 | abtm_multisource | 35.244 | 22.439 | 0.5751 | 0.0121 | 0 | 0 | 0 |
| 1 | abtm_multilevel | 179.758 | 22.685 | 0.5765 | 0.0119 | 0 | 0 | 0 |
| 2 | equal | 0.220 | 12.690 | 0.3008 | 0.0152 | 16,268 | 8,002 | 2 |
| 2 | graph_contiguous | 35.505 | 12.818 | 0.3036 | 0.0167 | 16,234 | 7,989 | 2 |
| 2 | abtm_region_growth | 31.055 | 11.428 | 0.3002 | 0.0072 | 600 | 300 | 2 |
| 2 | abtm_multisource | 42.164 | 11.502 | 0.2961 | 0.0076 | 1,196 | 600 | 2 |
| 2 | abtm_multilevel | 190.382 | 11.789 | 0.2955 | 0.0095 | 1,954 | 981 | 2 |
| 4 | equal | 0.333 | 7.409 | 0.1850 | 0.0256 | 28,372 | 14,035 | 12 |
| 4 | graph_contiguous | 39.572 | 6.970 | 0.2156 | 0.0225 | 28,296 | 14,008 | 12 |
| 4 | abtm_region_growth | 31.669 | 5.865 | **0.1711** | 0.0090 | 3,610 | 1,813 | 10 |
| 4 | abtm_multisource | 52.308 | 6.649 | 0.1791 | 0.0130 | 5,818 | 2,922 | 12 |
| 4 | abtm_multilevel | 189.672 | 6.023 | 0.1898 | 0.0115 | 4,094 | 2,053 | 8 |
| 8 | equal | 0.384 | 4.990 | 0.1584 | 0.0394 | 48,258 | 23,085 | 46 |
| 8 | graph_contiguous | 42.371 | 4.297 | 0.1554 | 0.0414 | 48,004 | 22,987 | 46 |
| 8 | abtm_region_growth | 31.659 | 3.951 | 0.1516 | 0.0334 | 6,422 | 3,227 | 32 |
| 8 | abtm_multisource | 72.136 | 3.908 | **0.1427** | 0.0246 | 11,602 | 5,841 | 50 |
| 8 | abtm_multilevel | 201.029 | 3.832 | 0.1634 | 0.0329 | 7,318 | 3,687 | 36 |

All 20 method/rank combinations reported `K3 NATIVE PASS`, with `rel_err=0.000e0` **at logged precision**, and matching sent/received f64 counts. The benchmark compares each local output and ghost against a serial CSR reference and rejects relative errors greater than `1e-11`. This is numerical validation, not a proof of exact bitwise identity for all inputs.

## Interpretation of these specific measurements

- **4 ranks:** region growth reduced Ghost DOFs from 14,035 to 1,813 (**87.1%**), exchange time from 0.0256 to 0.0090 ms (**64.8% lower**) and SpMV from 0.1850 to 0.1711 ms (**7.5% lower**) relative to Equal.
- **8 ranks:** multisource was the fastest timed SpMV (0.1427 vs Equal's 0.1584 ms, **9.9% lower**) while region growth had fewer Ghost DOFs (3,227 vs multisource's 5,841). Ghost count or peer count **alone** is therefore not a reliable latency predictor on this host.
- **2 ranks:** observed speed differences are 0.0006–0.0053 ms; do not infer a robust policy winner from one execution of 15 timed samples.
- **1 rank:** all policies have zero cut, Ghost, and peer relationships. The reported exchange time (~0.012 ms) is implementation/timing overhead; **it is not network traffic**. ABTM's nonzero partition work has no useful communication benefit at one rank.
- **Observed 1-to-8 rank strong scaling** on this one node is about 3.65x for Equal and 4.03x for multisource; this is *intra-node* parallelism, **not multi-node scalability**. CPU core/cache/memory-bandwidth contention and scheduling may materially change the result.
- The following crude amortization estimates use reported `partition_ms + setup_ms` differences and `spmv_ms` medians, not full application wall times: 4-rank region ~2,143 SpMVs; 8-rank region ~4,446 SpMVs; 8-rank multisource ~4,501 SpMVs. They are **illustrative, not validated break-even measurements**.

## Measurement boundaries / caveats

1. The experimental executable loads/replicates the full input CSR on **every rank**, so the measurements do not establish scalable distributed input or rank-local total memory usage.
2. The timed SpMV measurement excludes construction of `x_owned` from the global input vector immediately before timing. Halo setup timing also does not encompass all global CSR-to-owned-row extraction work. Full input read, conversion, replication, and full application startup are not in the cited amortization estimate.
3. K3 measures **one host only**, with MS-MPI and no network transfer between physical machines. Multi-node latency/bandwidth, NUMA, process placement/affinity, and remote-memory effects were not investigated.
4. Median/max-rank summaries omit tails and run-to-run variation. Confirm the 4/8-rank differences using independent, alternating-order repeats and raw CSV samples before creating a performance selector.
5. K3 is a separate opt-in MPI-SpMV path. It is **not wired into MPI-PCG, preconditioners, automatic policy selection, production distribution or ABI/FFI**. Do not claim solver-level acceleration or multi-node readiness.
6. Graph comparisons are on a scalar surrogate layout: no physical FEM field/element metadata has been validated here.

## Reproduce on the same single Windows node

```powershell
cd D:\Work\hybit\hybit
. .\tools\env\Enter-HybitMsMpi.ps1
$mpi = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
$exe = 'D:\Work\hybit\hybit\target\release\examples\mpi_k3_assignment_spmv.exe'
foreach ($r in @(1, 2, 4, 8)) {
    $csv = "D:\Work\k3_thermal1_r${r}_rerun.csv"  # must not exist
    & $mpi -n $r $exe $r D:\Work\thermal1.mtx 16 3 15 $csv
    if ($LASTEXITCODE -ne 0) { throw "K3 native MPI failed at ranks=$r" }
}
```

## Next session: H1-C3K4 proposed gates

1. Preserve this K3 checkpoint and the existing contiguous production MPI-PCG path. Do **not** merge the noncontiguous assignment into PCG without explicit global/local-index, owned-vector, reduction, and preconditioner correctness contracts.
2. Repeat interleaved 4/8-rank runs (multiple independent processes) and review **raw CSV**, including variability, ranks' max/mean time, and communication asymmetry. Test additional real matrix structures, not only `thermal1.mtx`.
3. Measure owner-map broadcast, partition creation, *noncontiguous row extraction*, `x_owned` Gather, Halo setup, numeric exchange, SpMV, and resident memory **separately**. Collect end-to-end amortized metrics for repeated SpMV / multiple RHS.
4. Prepare noncontiguous MPI-PCG as an independent experimental path; verify real residual, iteration counts, numerical equivalence, and failure propagation at 1/2/4/8 ranks before any solver routing change.
5. For genuine distributed scaling claims, acquire **at least two physical nodes** and benchmark network-dependent behavior. Until then, label every speed comparison **"single-node MPI"**.

**Checkpoint recommendation:** commit/push K3 as an **experimental single-node validation**, retaining all production routes unchanged. Nothing here implies a confirmed multi-node or PCG speedup.
