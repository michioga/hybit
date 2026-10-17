# HyBIT 0.9 H1-C3J — scoped ABTM Graph proposal reuse

Status: **Windows MS-MPI correctness/reuse checks PASS; interleaved synthetic A/B recorded; production speedup not established**.
Baseline: `develop/0.9.0`, clean/pushed commit `a129ee7` (H1-C3I).
No production auto selection, MPI transport, public HyBIT solver routing, or C ABI changes.

## Problem and hypothesis

The root-only H1-C3I benchmark computes an ABTM field-aware Graph candidate
once for timed calibration and `cost_aware_contiguous_partition_scoped` can
compute it **again** after robust admission. The second search repeats topology
preparation and graph refinement without improving correctness.

**Hypothesis:** a typed, scoped, immutable prepared candidate can be replayed
only after matching H1-C3D calibration and structural/imbalance checks,
reducing second-selector wall time without changing the partition selected.

## API and safety

- `prepare_cost_graph_candidate_scoped`: root-only opt-in proposal; holds exact
  proposal partition, `FieldGraphPartitionComparison`, original search time,
  and `PartitionCalibrationIdentity` (CSR structure **and numeric values**,
  physical fields, ranks, radius, selected Equal/Field baseline, environment).
- `cost_aware_contiguous_partition_scoped_prepared`: recomputes *current*
  baseline and identity and verifies both supplied timing evidence and cached
  proposal identity. Any mismatch fails closed to Equal/Field (`ScopeMismatch`).
  Missing timing evidence returns `Missing` and the reference baseline.
- The old `cost_aware_contiguous_partition_scoped` and
  `select_and_broadcast_cost_partition_mpi` remain unchanged at their public
  interface. Both cached and non-cached selection run the **same** economic,
  structural dominance, imbalance, and report logic; only candidate source
  changes. `local_graph_prepare_ns` reports the *previously paid* proposal
  preparation time, not time spent again in the final selector.
- `broadcast_root_cost_selection_mpi` continues to propagate root success/error
  collectively and check ownership/field layout before native Halo use.
  Only root constructs ABTM/global CSR; workers continue using owned rows.

This is scope fingerprinting against accidental reuse, not a cryptographic
attestation of hardware or timing provenance. The caller must not mutate the
matrix/storage between proposal preparation and calibration. The existing
H1-C3I robust gate is NOT weakened by this experiment.

## Comparison / acceptance gates

1. Rust tests: prepared-vs-legacy partition, selected policy, exact CSR telemetry,
   and model net savings agree. Mutation of matrix values, environment and
   radius must reject prepared evidence; missing evidence and zero expected
   calls must keep baseline. No new default automatic routing.
2. Windows: rustfmt, strict MPI-feature Clippy, Release Rust tests, Source
   Integrity and cached-diff check all PASS.
3. Native MS-MPI: `mpi_root_graph_reuse_bench` at 1/2/4/8 ranks and
   N=10,000; 2/4/8 ranks and N=1; 2/4/8 ranks at N=1,000,000 to exercise
   positive admission where observed. For identical captured calibration evidence, require prepared-vs-legacy
   selected offsets and guard to match; separately verify owned rows, exact
   CSR numeric values and reciprocal Halo traffic in native MPI.
   Independent launches are necessary before performance claims.
4. Compare `second_selector_wall_ms` against the *same binary* H1-C3I
   baseline on same machine/settings; record per-rank max, median of launches,
   original ABTM proposal time, probe cost, modeled total cost and memory.
   Timing from separate sessions and small synthetic matrices is not a stable
   quantitative speedup guarantee.
5. No measured full-N iterative solution here. Real FEM Matrix Market,
   heterogeneous DOF layouts, MPI-PCG convergence, true residual and peak
   preparation memory remain subsequent release gates.

## Windows invocation (after Preflight and Apply)

```powershell
$env:CLANG_PATH='C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH='C:\Program Files\LLVM\bin'
$env:MSMPI_INC='C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64='C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'
cargo fmt --all -- --check
cargo clippy --locked -p hybit-distributed --features mpi --all-targets -- -D warnings
cargo test --release --locked -p hybit-distributed --features mpi --lib
cargo test --release --locked -p hybit-distributed --test graph_candidate_reuse
cargo build --release --locked -p hybit-distributed --features mpi --example mpi_root_graph_reuse_bench
$exe=(Resolve-Path .\target\release\examples\mpi_root_graph_reuse_bench.exe).Path
foreach ($r in @(1,2,4,8)) { & mpiexec -n $r $exe $r 10000; if($LASTEXITCODE -ne 0){throw "reuse $r FAIL"} }
foreach ($r in @(2,4,8)) { & mpiexec -n $r $exe $r 1; if($LASTEXITCODE -ne 0){throw "fallback $r FAIL"} }
foreach ($r in @(2,4,8)) { & mpiexec -n $r $exe $r 1000000; if($LASTEXITCODE -ne 0){throw "long horizon $r FAIL"} }
```

Observed Source Integrity after installation: **299 manifest entries / 297 SHA-256 PASS**.
Do not commit before seeing Windows test and MPI logs. Do not tag, release,
or publish crates.io from this checkpoint.

## 2026-10-17 — Windows/MS-MPI native validation and interleaved A/B

**Evidence:** developer-supplied Windows 11/MSVC/MS-MPI logs from 2026-10-17.
The H1-C3J experiment remains opt-in; no default solver routing, C ABI or
production MPI transport was changed. This is a **synthetic 168-DOF,
3,552-nnz** matrix with 84 two-component field blocks, not a physical FEM
problem and **not** a measured 10,000-iteration solve. The reported
`modeled_*` totals extrapolate held-out per-SpMV medians.

### Correctness and build gates

- Original patch `-PreflightOnly` and `-Apply` PASS on `develop/0.9.0`
  baseline `a129ee7`; final integrity after install **299 manifest / 297
  verified SHA-256 PASS** (this corrects an erroneous 298/296 estimate
  printed by the original documentation).
- Rustfmt `--check`, MPI feature `cargo clippy --all-targets -- -D warnings`,
  `cargo test --release --locked -p hybit-distributed --features mpi --lib`
  **81/81**, and prepared reuse integration **5/5** PASS.
- `mpi_root_graph_reuse_bench`: ranks 1/2/4/8 for N=10,000 (4/4);
  ranks 2/4/8 for N=1 (3/3); ranks 2/4/8 for N=1,000,000 (3/3):
  **10/10 native MPI invocations PASS**. Both positive Graph selection
  and conservative Equal/Field fallback occurred. Every reported numeric
  CSR reference error was zero, `workers_global_csr=0`, and Halo traffic
  reconciliation passed.
- Independent interleaved A/B across **2/4/8 ranks x five repeats x two
  executables = 30/30 PASS**, alternating C3I/C3J launch order per repeat.
  All 30 records report zero reference error and `workers_global_csr=0`.
- Final non-MPI Release regression: `hybit-distributed` **54/54**,
  `hybit-matrix` **52/52**, `hybit-ffi` **5/5**, public API
  tests **3+27+4+13**, reuse integration **5/5**, hybit doctest **1/1**;
  remaining reported doctest suites PASS. `git diff --cached --check`
  and Source Integrity **299/297** PASS. Ten expected files staged; no
  commit or push during validation.

### Interleaved A/B — N=10,000 modeled calls, five independent launches each

| Ranks | C3I Graph selections | C3J Graph selections | C3I median probe wall ms | C3J median probe wall ms | C3I median initial Graph preparation ms | C3J median initial Graph preparation ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0/5 | 0/5 | 1.5529 | 1.7577 | 0.0769 | 0.1448 |
| 4 | 0/5 | 0/5 | 1.6390 | 1.7219 | 0.1204 | 0.2085 |
| 8 | 3/5 | 5/5 | 2.6120 | 2.7766 | 0.1374 | 0.1992 |

At 2 and 4 ranks the robust gate rejected Graph in every launch, so no
second ABTM Graph exploration was needed by either selector. H1-C3J does
**not** demonstrate a runtime speedup for these ranks. Some held-out Graph
per-SpMV timings were faster even on rejected runs; the four calibration
subepoch gates intentionally require consistent gains. A rejection does not
imply a numerical failure.

The stronger *like-for-like* comparison is **8 ranks, repeats 2/4/5**,
where both independent programs actually admitted Graph:

| Repeat | C3I second selector ms | C3J second selector ms | Observed reduction ms |
|---:|---:|---:|---:|
| 2 | 0.3718 | 0.2113 | 0.1605 |
| 4 | 0.4680 | 0.2976 | 0.1704 |
| 5 | 0.4406 | 0.3001 | 0.1405 |

**Median absolute reduction: 0.1605 ms** for those three matched
Graph-admission cases. This supports avoiding the duplicate search; it
is not a full-N solve time improvement. The difference includes all
other final-selection steps and run-to-run noise; these are not isolated
ABTM kernel timings. C3J `graph_searches=1` is a fixed program diagnostic,
not an independently instrumented search counter. Code path and numerical
parity provide the additional evidence that no second search is issued.

At 8 ranks the **C3J modeled total including probing** was lower than its
**own per-launch** baseline-only modeled cost **5/5** times (C3I **3/5**).
C3J 8-rank fully accounted modeled Graph break-even, over five launches,
was **2,213–2,463 calls; median 2,243 calls**. This calculation uses
held-out medians after selection, not an actually timed multi-thousand
iteration Krylov solve. C3J Graph admission 5/5 vs C3I 3/5 in independently
noisy trials does **not** prove that the prepared cache increased the
true admission probability, as calibration conditions differed.

### Performance qualification and next work

- Initial Graph preparation and total calibration probe were **longer in
  C3J's median launches at every rank count** (table above). A complete
  end-to-end improvement over C3I is **not** established. Compare the
  prepared fingerprint/validation overhead separately and avoid double
  charging the initial Graph preparation that is already in `probe_wall_ms`.
- No paired per-launch controlled *identical calibration samples* comparison
  exists; interleaving launch order reduces order bias but does not remove
  clock/OS scheduling or MPI jitter. No statistical confidence interval is
  justified by five launches.
- Matrix Market real FEM, physical variable-field DOF, rank-local distributed
  ingestion, MPI-PCG true residual, solve-many amortization, and peak memory
  are **out of scope**, and remain prerequisites for production automatic
  adoption or a global HyBIT performance claim.
- **Checkpoint recommendation:** accept H1-C3J as opt-in functional
  candidate reuse after final documentation/source integrity confirmation;
  retain existing default path, document mixed performance, and commit/push
  only upon explicit user action. Next test stage should instrument identity
  digest/reuse overhead and integrate representative FEM/MPI-PCG measurement.
