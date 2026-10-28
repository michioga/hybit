# HyBIT session handoff / 引継ぎノート

> **2026-10-12時点の共有ZIPに基づくスナップショット。** 次セッション開始時は必ず実際のローカル `git status` / `git log` / テストログで更新する。ここにあるコミットID・dirty状態を永続的に最新と誤認しない。

## 固定の長期目標

詳細は `docs/PROJECT_CHARTER.md`。要約：**簡単なユーザーAPI**、低メモリ反復法を開始点とし、停滞時にはより強い前処理・必要な場合の直接法に切替可能な自律ハイブリッドソルバ、**ABTM**による圧縮・ビット論理演算・高効率Halo、**RayonとMPI**による大規模FEM/HPCのend-to-end高速化。既存機能を極力再利用。検証節目ごとにcommit/push。**大きな開発計画の検証が成功した時点で、ユーザーの判断によりGitHub Releaseとcrates.ioへ正式公開する。**

## 2026-10-12 に受領した `hybit.zip` の状態

- ローカル予定パス：`D:\Work\hybit\hybit`。ブランチ：`develop/0.9.0`、ZIPのHEAD：`e5c90be`。
- ZIP内で未コミット変更を確認した6ファイル：`SOURCE_SHA256.txt`、`crates/hybit-distributed/src/mpi_block_jacobi.rs`、`mpi_pcg.rs`、`mpi_two_level.rs`、`crates/hybit/examples/mpi_two_level_experiment.rs`、`docs/SOLVER_SELECTION.md`。
- G8-C12d-6C4A：8-rank MS-MPI実験（smooth/mixed/localized RHS）でpooled replay memory-budget equivalence gate等がPASS。自動セレクタの有効性を示したものではない。
- C4A masked bit microbench：bool基準比6.74×（旧）、既存 `[u32;8]` との比較では *prepacked* `[u64;3]` のXOR+POPCNTが約1.64×速いが、毎回の変換を含めると約92.8×遅い（当該microbench）。ABTM疎行列SpMVやMPI総性能とは無関係。
- G8-C12d-6C5：MPI-owned CSR vs ABTM probeのpreflight PASS。**数値比較は未実行**。Rust `mpi-sys 0.2.4` → `bindgen 0.72.1` で `libclang error`（Fix1でも再現）。C5の一時Rust exampleは撤去され既存6ファイルは保持された。
- 環境：Windows 11、Rust `x86_64-pc-windows-msvc`、MS-MPI 10.1.12498.52、MS-MPI SDK、LLVM/libclang 23.1.3（検出確認）。MinGW系MPIは使用しない。
- ビルドログ中 `VCINSTALLDIR`/`INCLUDE`/`LIB` がnoneだが、これだけでエラー原因は確定しない。MS-MPIリンク先は見つかっている。`libclang.dll`の存在確認だけではヘッダ解析が成功したことにならない。

## 既存の主な技術要素

- 公開APIとhybrid：`hybit::solve`、`HybitSolver`、prepared solve-many、局所Cholesky昇格。
- ABTM：`docs/ABTM_TOPOLOGY.md` と `hybit-matrix` の実装。G6で普通のscalar ABTM SpMVの優位性は確認されず、prepared CSRを継続利用する設計。
- MPI：`mpi_backend.rs`、`mpi_input.rs`、`mpi_overlap.rs`、`mpi_pcg.rs`、`mpi_two_level.rs`、既存MS-MPI実機テスト。
- Rayon：並列prepared演算、CSR SpMV、ベクトル/前処理（細部は現行実装を読む）。

## 開発／公開の運用区分

- **開発checkpoint：** 検証済みの小さな変更ごとにcommit/pushし、最新ソースとhandoffをGitHubで参照可能にする。
- **正式リリース：** 大きなマイルストーンの合格後に限り、バージョン・互換性・数値精度・性能・パッケージを確認し、ユーザー承認後にGitHub Releaseとcrates.ioを公開する。
- ユーザーから指示がない限り、タグ付け、Release作成、`cargo publish`は行わない。現時点のREADMEにはHyBIT 0.8.0の公開実績がある。0.9.0は開発途中であり正式公開完了とは扱わない。

## 次のセッションでの一手

1. **まず最新のローカルZIP/ブランチ状態を照合。** このメモより新しいcommit/pushがあればそちらを採用。
2. 未コミットだったC4Aの変更について、既存の成功gateを確認した上で適切な粒度でcommit/pushする（ユーザー判断）。
3. C5の実験を繰り返す前に、MSVC Developer環境、MS-MPI Cビルド、`clang`ヘッダ解析、`mpi-sys`クリーンビルドを独立に検証し、特定した原因に対する最小修正を行う。
4. ビルド環境が安定してから、既存Halo/PCG instrumentationsを使って分散ABTM構造演算の価値を実測する。

## セッション終了時の更新用ひな型

```text
日付:
Branch / HEAD / push済みcommit:
GitHub Release / crates.io の公開状況:
Dirty files:
今回変更したファイル:
実機環境 / MPI ranks / threads / 行列:
PASS gates:
FAIL gates とログへのリンク/パス:
主な実測値と比較ベースライン:
未解決の技術課題:
次の1作業:
```

## 2026-10-13 update — supersedes older environment/C5 notes above
<!-- HYBIT_2026_10_13_ARCHITECTURE_HANDOFF -->
**Authoritative design:** `docs/HYBIT_LONG_TERM_ARCHITECTURE.md` and `docs/HYBIT_DECISION_LOG.md`. **Resume workflow:** `docs/SESSION_RESUME_PROTOCOL.md`. **Current user-supplied test evidence:** `docs/HYBIT_VALIDATION_BASELINE_2026-10-13.md`.

- Windows MS-MPI / MSVC native build and MPI 2/8-rank smoke PASS with explicit `CLANG_PATH=C:\Program Files\LLVM\bin\clang.exe` and `LIBCLANG_PATH=C:\Program Files\LLVM\bin`. Earlier C5 `mpi-sys`/`bindgen` failure is **not an active blocker** in that tested environment.
- MPI-owned C2 8-rank PCG: **20/20 numerical PASS**. B3 4-rank overlap SpMV: **4/4 PASS**, errors zero; the tiny microbenchmark does not guarantee FEM scaling.
- MPI C5 `nd3k.mtx` native 1/2/4/8 ranks: **numerical PASS** (CSR-vs-ABTM rel_error about `3e-14`–`9e-14`). 8 ranks: CSR SpMV `0.301870 ms`, ABTM SpMV `0.523540 ms`; storage CSR `39,392,312` bytes vs ABTM `31,390,616` bytes; ghosts `44,213` (sum of receives). Separate ABTM conversion max `19.731600 ms`; ordinary ABTM numeric SpMV is **slower**. No claim of full-PCG ABTM speedup.
- At end of latest user-run C5 the reported `git status --short` was **empty**, and the temporary example was removed. Current HEAD/branch/dirty state **must be rechecked**, not inferred from 2026-10-12 snapshot.
- New nonnegotiables: iterative-first plus local-direct reinforcement, simple public API, metadata-first ABTM, reduced MPI/Rayon data movement, mixed-field variable DOFs, long-term frictional contact and nonsymmetric/indefinite systems. Do **not** claim these new requirements are implemented.
- **Next one action:** inspect current local code/API and specify the smallest testable `FieldAwareDofLayout` + constraint/topology/versioning contract, keeping CSR execution unchanged. Define exact reference cross-checks and rollback before any Rust code changes.

## 2026-10-14 H1-B — Field-aware ABTM symbolic region (validation pending)

- Starting point: **H1 `81f653d`** pushed to `develop/0.9.0`, worktree clean; H1 Rust tests 46 matrix / 4 FFI passed, independent C/C++/Fortran DLL ABI smoke passed after WinLibs-first PATH precedence.
- Added checkpoint proposal: `field_region.rs` block-closed undirected ABTM growth, reuse of G3 `A[R,R]` symbolic extraction, ownership continuity validation, optional C/C++/Fortran packed-mask API. **Not yet adopted as an automatic solver/Schwarz/MPI path.**
- H1-B gate status: **PENDING** native Windows cargo/Clippy/source-integrity/C/C++/Fortran runtimes. Run and record actual PASS/FAIL before commit/push.
- Next *one* priority: run `APPLY_HYBIT_H1B.ps1 -PreflightOnly`, then `-Apply` and the H1-B acceptance gate; send logs. Preserve simple `hybit::solve` and C ABI compatibility.

## 2026-10-14 H1-B accepted and pushed; H1-C1 preparation proposed

- H1-B validated on Windows 11 native MSVC Rust toolchain: `hybit-matrix` 50/50 PASS, `hybit-ffi` 5/5 PASS, old `hybit` public tests PASS; Clippy `-D warnings` PASS after `field_region.rs` iterator fix, source integrity 267 manifest / 265 verified PASS, C/C++/Fortran executable ABI smoke PASS (`0x77` / `119` / `119`).
- H1-B `develop/0.9.0` HEAD `10b6eec`, pushed to `origin/develop/0.9.0`, clean worktree per user-provided log.
- H1-C1 proposal: cached ABTM dual topology and field-aware region + G3 local numeric-address plans, reuse with unchanged CSR structure, Rust-only opt-in; no MPI exchange or numerical solver routing modifications. C/C++/Fortran existing ABI preserved; new prepared ABI is future work.
- **H1-C1 gates PENDING** Windows Rust test/Clippy/source-integrity. Do not mark passed based on patch installation.
- Next one action: validate H1-C1 package on exact `10b6eec`, then separately design real MS-MPI owner/ghost/Halo integration H1-C2.

## 2026-10-14 H1-C2 — block-aligned owned/ghost + native MPI smoke (candidate)

- Canonical developer-side baseline: **`bbfafc2`**, pushed to `develop/0.9.0`, clean working tree; H1-C1 52 hybit-matrix unit tests and 5 FFI tests PASS, strict Clippy and source integrity PASS. Existing C/C++/Fortran ABI unchanged.
- Patch adds `field_halo.rs`: deterministic whole-block-aligned contiguous rank partition, existing arbitrary `PartitionAssignment` field-ownership validation, and adapter to already-implemented unique-ghost reciprocal HaloPlan / rank-local CSR / telemetry. No new MPI transport; no new automatic repartitioning policy.
- Real MS-MPI smoke `mpi_field_aware_halo` compares the actual rank-owned CSR/Halo to global-reference plans and executes native MPI SpMV at 1/2/4/8 ranks. Global fixture replication is testing-only. Does not establish performance speedups.
- **Validation pending**: developer-machine Preflight/Apply, Rust + strict Clippy, Source Integrity, native MS-MPI 1/2/4/8 rank execution; do not mark PASS until logs arrive.
- Next prioritized action: execute H1-C2 validation gates on Windows, correct any concrete failing gate, then checkpoint commit/push with the new results.

## 2026-10-14 H1-C3 graph-aware field partition (patch pending validation)

- Baseline `bbe5cef` on `develop/0.9.0`; H1-C2 Windows Rust 29 distributed /52 matrix /5 FFI and native MS-MPI 1/2/4/8 ranks all PASS. H1-C2 push confirmed and worktree clean.
- H1-C3 proposed: `field_graph_contiguous_partition` traverses ABTM deduplicated symbolic edges, proposes nearby *whole-field-block* contiguous rank cuts, then accepts only if the existing HaloPlan CSR reference reports non-regression for ghost count, cut nnz, and peer relations. Also adds `PreparedFieldGraphHalo`, reference unit tests and a native 1/2/4/8-rank MS-MPI smoke. No solver/ABI change; no arbitrary noncontiguous ownership support.
- H1-C3 **Rust, Clippy and native MS-MPI verification pending on developer Windows machine**. No performance or speedup claim; no commit/push until gate review.
- NEXT: H1-C3 preflight/apply and Rust correctness gates, then native MS-MPI smoke; review comparisons and any regressions before checkpoint commit.

## 2026-10-14 H1-C3B — Matrix Market partition benchmark (candidate, tests pending)

- Developer-provided canonical baseline: `224b129` (`develop/0.9.0`, pushed, clean); H1-C3 Windows Rust/Clippy and native MS-MPI 1/2/4/8 synthetic fixture PASS. H1-C3 @ 8 ranks: Ghost 22 -> 20, cut 242 -> 188, peer 12 -> 8, max DOF imbalance 1.160; no proven end-to-end speedup.
- Proposed H1-C3B adds **one opt-in example**, `mpi_field_mtx_bench`, to compare equal/field/graph contiguous partitions using real `.mtx`, actual MS-MPI halo SpMV, max-rank time medians, exact traffic and CSR numerical cross-checks.
- Matrix Market lacks physical field metadata. The benchmark uses synthetic scalar one-DOF field blocks ONLY, not a genuine heterogeneous FEM DOF layout. Global CSR is replicated per rank ONLY within conservative corpus limits.
- All Windows Rust/MPI and actual `.mtx` tests **PENDING**. No solver routing or published C ABI changes.
- Next one action: apply only after preflight, run native 1/2/4/8-rank MS-MPI benchmarks starting with Goodwin_010 and sherman5; review numerical/traffic/time/imbalance logs before committing.

## 2026-10-14 H1-C3B validation completed — supersedes the H1-C3B candidate/pending entry above

- **Baseline:** `224b129`, branch `develop/0.9.0`. H1-C3B patch was staged but not committed at the time of these tests. The earlier “tests PENDING” note is historical and is superseded by this entry.
- **Gates PASS:** H1-C3B Preflight/Apply; `cargo fmt --all -- --check`; Rust release tests (`hybit-distributed` 32, `hybit-matrix` 52, `hybit-ffi` 5, existing public suites and doc-test); non-MPI and MPI-example Clippy `-D warnings`; MS-MPI release build; Source Integrity 277 / 275. Initial `libclang error` cleared by explicitly setting `CLANG_PATH`, `LIBCLANG_PATH`, `MSMPI_INC`, `MSMPI_LIB64` (LLVM 23.1.3); no HyBIT source correction needed.
- **Native MS-MPI numerical PASS:** Goodwin_010 (1/2/4/8 ranks), sherman5 / nd3k / thermal1 (2/4/8 ranks); for the confirmation, four matrices at 4 and 8 ranks × three policies, 5 warmups + 20 measured calls per policy; all reported `rel_err=0`.
- **Confirmed median of max-rank SpMV (Equal / Field / Graph, ms):** Goodwin_010 4r 0.00800/0.00790/0.00760, 8r 0.00940/0.01045/0.00955; sherman5 4r 0.00955/0.00920/0.00980, 8r 0.01875/0.01955/0.01895; nd3k 4r 0.71745/0.66320/0.74120, 8r 0.78075/0.77120/0.78110; thermal1 4r 0.18760/0.17525/0.18350, 8r 0.18760/0.17155/0.20305.
- **Graph selected as fastest in only 1/8 conditions** (Goodwin_010 4 ranks). Examples of structural improvements: Goodwin_010 8r Ghost 1,244 -> 1,163 (nnz max imbalance 1.126 -> 1.300); sherman5 8r Ghost 2,505 -> 2,456 and peers 36 -> 32; thermal1 8r Ghost 23,083 -> 22,905 but Field had lower measured SpMV time.
- **Graph rejected by its structural selector on nd3k 4/8r**; selected partition was identical to Field but graph preparation still cost 124.3916 / 195.9729 ms versus Field 0.0504 / 0.1968 ms. Differences in observed SpMV timing on those rejected cases must not be interpreted as a partition-policy effect.
- **Limits:** `.mtx` lacks real field-block semantics; 1 scalar DOF per synthetic block. Global CSR replicated on all ranks solely for bounded benchmarks. No full PCG/FGMRES speedup, physical-field correctness, production memory advantage or automatic solver adoption is claimed. Exact 8-case tables and discussion: `docs/H1C3B_REAL_MTX_BENCH.md`.
- **Next single task:** checkpoint/commit H1-C3B after doc/SHA gates; then design **H1-C3C cost-aware early reject and selection**, retaining Equal/Field fallback and no production auto-route until measured performance justifies it. Do not create release/tag or publish crates.io without user approval.

## 2026-10-15 H1-C3C — conservative cost-aware partition candidate (not yet validated)

- Starting checkpoint: `5d0e919` pushed to `develop/0.9.0`, clean worktree per user log. H1-C3B MTX benchmark and docs accepted; single-machine `Goodwin_010`/`sherman5`/`nd3k`/`thermal1`, 4/8 ranks, 20 repeated SpMVs, accurate within reported error zero. Graph fastest in just 1/8 cases; `nd3k` 4/8 graph proposal was rejected but incurred symbolic preparation. Do NOT claim Graph is generally faster.
- New candidate H1-C3C explicit `cost_aware_contiguous_partition`: field-safe Equal/Field baseline, *pre-ABTM* economic rejection without representative SpMV timing evidence, opt-in Graph with exact Halo and load caps. Shared calibration only; local measured preparation is diagnostic and never controls rank-dependent MPI decisions. No auto routing, no ABI change, no transport change.
- **Validation pending**: Rust unit tests (7 new), `cargo fmt`, Clippy `-D warnings`, Source Integrity, and example probes on `Goodwin_010` and `nd3k`. Native MS-MPI example is deliberately unchanged; this new probe is non-MPI. Do not commit until Windows logs confirm gates.
- Next: apply H1-C3C package on exact `5d0e919`; inspect test/probe logs. Correct any defects, document actual results, then commit/push checkpoint only. Then design MPI agreement and real mixed FEM field layouts.

## 2026-10-15 H1-C3C native Windows Rust and MTX policy probes — validated, commit pending

- Replaces the earlier H1-C3C "validation pending" note above, **for the explicit non-MPI selector only**. Starting HEAD `5d0e919`, staged modifications not yet committed/pushed.
- Windows `cargo fmt --check`, `cargo test --release --locked` (39 distributed, 52 matrix, 5 ffi, existing public tests), strict Clippy, source integrity (280 manifest / 278 SHA), staged diff gate: **PASS**.
- `Goodwin_010` ranks=4 radius=5: expected 1,000 calls -> `UnamortizedPreparation`, no ABTM; 10,000 calls -> `GraphSelected`, ghosts 542→513, cut nnz 5091→4930, local nnz imbalance 1.1090. `nd3k` ranks=4 radius=5 -> `NoCalibratedSpeedup`, no ABTM. `Goodwin_010` radius=32 -> `LoadImbalanceLimit`, Graph attempted, Field retained, 1.0385 ms local graph prep wasted.
- **Calibration mismatch:** Goodwin's supplied baseline 8000 ns is the former **Equal** timing, but selector chose **Field** (measured 7900 ns in H1-C3B). Radius=5 Graph time is unmeasured, while supplied Graph 7600 ns was from radius=32. Current `GraphSelected` is illustrative model behavior only, **not validated MPI speedup**. Do not deploy automatic selection without matching calibration provenance.
- No native MPI consensus, MPI PCG/FGMRES, mixed physical field FEM, or end-to-end performance validation of H1-C3C. Existing MPI transport and solver defaults unchanged.
- Next: commit/push H1-C3C after documentation+Source Integrity gate; then design calibration-identity guard and root-coordinated MPI selection. No release/tag/publish.

## 2026-10-15 H1-C3D candidate — scoped calibration identity and MPI collective guard

- Baseline `c711b99` clean/pushed `develop/0.9.0`: H1-C3C Rust 39 distributed / 52 matrix / 5 FFI, strict Clippy, 280 manifest / 278 SHA and real MTX non-MPI probes passed; docs and SHA Fix2 accepted. H1-C3C remains opt-in and not end-to-end MPI auto policy.
- H1-C3D proposal adds deterministic noncryptographic identity (full CSR pattern and numeric values, physical field blocks, ranks, search radius, selected Equal/Field baseline, explicit environment tag) to caller-supplied calibration. Missing/mismatched identity fails closed to Equal/Field without ABTM. Caller is responsible for actual matching performance evidence; **not** an automatic benchmark or proof of speedup.
- MPI feature adds a collective scope/options/decision agreement and exact offset Allgather before Halo; included native 1/2/4/8-rank numerical smoke and 2/4/8-rank deliberately mismatched-environment rejection runs. No change to production MPI selection, routing or C ABI; root-coordinated selection and field-real FEM tests are still future work.
- **All H1-C3D Windows/native MPI tests PENDING.** Run patch Preflight/Apply, 45 expected hybit-distributed tests (39+6), release full regression, strict Clippy (MPI and non-MPI), Source Integrity and native MS-MPI positive/negative runs. Do not commit or push before logs are reviewed.

## 2026-10-15 H1-C3D native Windows/MS-MPI validation — supersedes H1-C3D PENDING note

- Reference baseline: `c711b99` on `develop/0.9.0`, patch **staged, not yet committed/pushed** at time of the tests. Preflight/Apply, formatting, release regressions (`hybit-distributed` **45/45**, `hybit-matrix` **52/52**, `hybit-ffi` **5/5**, HyBIT public tests), ordinary and MPI-enabled strict Clippy, MS-MPI release build, Source Integrity **284/282**, and staged diff check: **PASS**.
- Native MS-MPI positive smoke: 1/2/4/8 ranks **PASS**, matching calibration scope and identical selected Equal partitions; `ghosts=0/4/8/16`, send=receive=ghost count, `abs_error=0`. Decision: `SingleRank` at 1 rank, `UnamortizedPreparation` at 2/4/8. This is **not** a demonstration of a Graph-selected MPI execution.
- Native negative smoke: `--mismatch-env` 2/4/8 ranks **PASS**, deliberate rank-1 environment-tag disagreement **rejected collectively** before Halo traffic. Corrupted offsets and mismatched options were not separately fault-injected in these runs.
- Identity is deterministic **noncryptographic FNV-1a** over the metadata selected by the guard; performance evidence and environment tags are caller-provided. Tests do **not** establish end-to-end PCG/FGMRES speedup, real heterogeneous FEM field metadata, production auto-selection, or full fault tolerance.
- **Next single action:** refresh these docs/SOURCE_SHA256, verify integrity/staged diff, then commit/push H1-C3D checkpoint. Later work: independently test mismatched offsets/options, root-owned calibration and broadcast, real mixed FEM layouts, full-solver performance. No release/tag/publish.

## 2026-10-15 H1-C3E candidate — rank-0 owned policy and offset MPI broadcast

- **Clean pushed base:** `75fe1b1` on `develop/0.9.0` (H1-C3D). Native H1-C3D Rust/MPI 1/2/4/8 positive and 2/4/8 environment-mismatch negative PASS, Source Integrity 284/282.
- **Proposed H1-C3E:** root rank alone constructs the global CSR and runs H1-C3D scoped selection; all ranks collectively compare metadata/options, broadcast root-success/header/full offsets, validate field-safe contiguous ownership before reusing existing `mpi_input::prepare_owned_rows` / `spmv_local`. Non-root ranks only need global field-layout metadata and rank-owned CSR rows. No production routing changes, MPI-PCG/FGMRES modifications or C ABI change.
- **Safety smoke proposed:** native MS-MPI 1/2/4/8 rank owned-row SpMV numerical PASS and 2/4/8 rank collective rejections for changed environment/options/layout, injected root error, split-field offset. Calibration is synthetic, Graph speedup/Graph-selected MPI not demonstrated.
- **All H1-C3E native Windows acceptance gates PENDING.** Preflight+Apply, cargo release tests, strict Clippy normal/MPI, Source Integrity expected 287/285, MPI positive/negative tests. Do not claim PASS, commit, push or release before user-supplied logs.
- **Next action:** validate patch against exact `75fe1b1`, then review any Rust/bindgen/MPI runtime errors and correct with a small follow-up patch; document actual results before checkpoint commit/push.
## 2026-10-15 H1-C3E Windows/MS-MPI validation — supersedes H1-C3E PENDING entry

- **Reference base:** `75fe1b1`, `develop/0.9.0`; H1-C3E patch staged, not committed/pushed at test time. Preflight/Apply, `cargo fmt --check`, release tests (`hybit-distributed` 45/45 non-MPI, `hybit-matrix` 52/52, `hybit-ffi` 5/5, existing public API), strict Clippy normal and MPI-enabled, MPI example release build, Source Integrity **287/285**, staged diff check: **PASS**.
- **Native MPI normal:** 1/2/4/8 ranks all PASS; Equal policy, `SingleRank` at 1 rank / `UnamortizedPreparation` otherwise; global owned DOFs 160, ghosts 0/4/8/16, send=receive, zero absolute SpMV error and `workers_global_csr=0`. Only root holds full synthetic CSR; all ranks still hold complete field layout.
- **Native MPI negatives:** 2/4/8 ranks x five injections = **15 PASS**, collectively rejecting mismatched environment/options/layout, root-side selection error, and split-field broadcast offset *before* incompatible Halo exchange.
- **MPI-feature unit tests:** `cargo test --release --locked -p hybit-distributed --features mpi --lib` **72/72 PASS**, including the 2 new H1-C3E digest tests. The initial `mpi-sys`/`bindgen` `libclang` error on a new PowerShell environment was resolved using explicit `CLANG_PATH`, `LIBCLANG_PATH`, `MSMPI_INC`, and `MSMPI_LIB64`; no source modifications were required.
- **Still not exercised:** Graph-selected MPI path, production distributed matrix ingest, real mixed-field FEM files, full MPI-PCG/FGMRES timings, memory scaling and arbitrary MPI failure handling. No speedup claim.
- **Next action:** apply the corrected H1-C3E validation-document patch and refresh SHA-256; verify Source Integrity/staged diff; then commit/push the development checkpoint on `develop/0.9.0` (no tag/release/crates.io publish). Future H1-C3F: test Graph-selected root broadcast with matching measured calibration and perform true owned-row FEM input and whole-solver comparisons.

## 2026-10-15 H1-C3F — Graph-selected root MPI comparison (candidate, native validation pending)

- Baseline `8601e93`, pushed to `develop/0.9.0`, clean worktree according to user's actual commit/push output; H1-C3E 45 non-MPI distributed / 72 MPI-enabled distributed Rust tests plus 1/2/4/8-rank native positive and 15 injected-negative MPI cases PASS, with H1-C3E root broadcast selecting Equal.
- New **example only** `mpi_root_graph_selection_bench` on synthetic 84 two-component field blocks/168 DOFs in eight dense regions with a directed inter-region connection. Expected `GraphSelected` at 2/4/8 ranks, SingleRank at rank 1; root alone holds global CSR and runs ABTM refinement, nonroots generate only owned-row CSR. Existing Halo/SpMV numerical equivalence and reciprocal traffic are checked.
- Compare baseline and Graph with 5 warmups + 20 medians of maximum-rank native MPI SpMV; collect Rank 0 selector and native setup time. Cost calibration values are **invented fixture values solely to exercise the selected-Graph code path**, not evidence of a speedup, accurate ROI, or comparable live production workload. No production auto-route or solver change.
- **PENDING:** Windows package Preflight/Apply, `cargo fmt`, MPI example Clippy/build, MS-MPI 1/2/4/8 PASS, source integrity and staged diff. Do not commit/push H1-C3F until actual user logs are reviewed. No release/tag/publish.

## 2026-10-15 H1-C3F actual Windows/MS-MPI validation — supersedes PENDING

- H1-C3F based on clean/pushed `8601e93` (develop/0.9.0); staged, not committed
  at this validation checkpoint. Preflight/Apply and FIX2 MSRV correction
  passed Source Integrity **289 / 287**. Initial strict Clippy failed on two
  `is_multiple_of` calls requiring Rust 1.87 > project MSRV 1.73; FIX2 used
  `% 7 == 0` / `% 2 == 0`. FIX1 SHA mismatch was a packaging guard error,
  not a compiler bug. A transient libclang/bindgen failure was resolved by
  setting CLANG_PATH/LIBCLANG_PATH/MSMPI_INC/MSMPI_LIB64 in the active shell.
- Final strict **MPI-enabled Clippy PASS**, release example build **PASS**,
  native MS-MPI 1/2/4/8 ranks **PASS**; numeric CSR reference error **0** for
  both baseline and chosen policy; `workers_global_csr=0`.
- Ranks 1: Equal/SingleRank, 0 ghosts, max-rank SpMV 1.900/1.900 us.
  Ranks 2: GraphSelected, ghosts **24->2**, SpMV **5.400->5.250 us** (2.8%).
  Ranks 4: GraphSelected, ghosts **68->4**, SpMV **4.900->4.250 us** (13.3%).
  Ranks 8: GraphSelected, ghosts **156->8**, SpMV **7.900->6.300 us** (20.3%).
  All times = 20-sample median of maximum-rank native MPI SpMV, 5 warmups;
  rank-0 selector took 0.2372/0.2914/0.3336 ms at 2/4/8 ranks.
- These are **tiny synthetic benchmark results**, not an end-to-end solver
  speedup or proof of general scalability. The calibration numbers supplied
  to force Graph selection are synthetic; the selector guard only verifies
  scoped identity, not calibration provenance. No real FEM, solver convergence,
  production matrix distribution, or repeated independent calibration tested.
- **Next:** append these validated observations to H1-C3F docs, update
  SOURCE_SHA256, run final integrity/diff and (preferably) full release Rust
  regression tests, then commit/push only once all checks pass. No tag,
  release, crates.io publish, or automatic solver policy change.

## 2026-10-15 H1-C3G candidate — native measured calibration, economic gate (pending)

- Exact source checkpoint **`a118565`**, branch `develop/0.9.0`, H1-C3F has been committed/pushed and the user confirmed `HEAD == origin/develop/0.9.0`, clean worktree.
- New **example only** `mpi_root_measured_calibration_bench.rs`: rank-0-only Graph proposal, native MPI baseline/candidate calibration with 5 warmups + 40 samples of maximum-rank SpMV, scoped capture of **measured** timing evidence (not H1-C3F illustrative numbers), existing root-broadcast selector, and 5 warmups + 20 held-out samples. Numeric and traffic checks include no full CSR on workers.
- Modeled break-even accounts for root Graph preparation + positive native setup delta; calibration **probe cost excluded**, repeated independent runs not yet done. This is still a tiny synthetic matrix, not real FEM nor production solver routing. No promise of GraphSelected: measured slow / unamortized candidates should fail closed to baseline.
- **PENDING** Windows Preflight/Apply, rustfmt, strict MPI Clippy, release build, native 1/2/4/8 MPI at N=10000 and 2/4/8 at N=1, release regression and Source Integrity. No commit/push/release/tag/publication until user shares PASS logs.
## 2026-10-15 H1-C3G Windows/MS-MPI measured-calibration validation — supersedes pending

- Base `a118565`, `develop/0.9.0`; H1-C3G staged, **not committed or pushed** at validation time. Preflight/Apply, rustfmt, MPI-enabled strict Clippy, release example, complete non-MPI release regressions (distributed 45/45, matrix 52/52, FFI 5/5, HyBIT public APIs), Source Integrity **291/289**, staged diff: **PASS**.
- Native MS-MPI **7/7 PASS**: N=10000 at 1/2/4/8 ranks and N=1 at 2/4/8 ranks; all numeric reference errors zero, guard `Matched`, `workers_global_csr=0`.
- At N=10000, rank 1 `SingleRank/Equal`; 2/4/8 `GraphSelected`. Baseline/Graph calibrated max-rank SpMV medians (us): 2r **4.10/4.00**, 4r **12.20/11.40**, 8r **10.15/8.80**. Root proposal preparation (ms) **0.0887/0.0878/0.1706**, modeled break-even **887/110/127** calls; ghosts **24->2 / 68->4 / 156->8**.
- At N=1, 2/4/8 ranks return `UnamortizedPreparation` and choose Equal/Equal/Field. The printed **modeled** total can nevertheless favor Graph on 4/8 ranks because the conservative selection gate does not credit native Halo setup savings. `modeled_*_total_ms` is a cost calculation, not full-N measured wall time.
- IMPORTANT: independent holdout selected SpMV medians (2/4/8 ranks, us) **2.25/2.80/5.20** differ substantially from Graph calibration medians **4.00/11.40/8.80**; repeat paired trials and examine jitter/variance. Holdout does not separately benchmark unselected baseline. The example excludes 40-sample calibration/probe overhead and a second Graph preparation performed by the guarded final selector. Thus printed break-even numbers likely understate full end-to-end break-even.
- **Next action:** apply H1-C3G native validation documentation update, run Source Integrity **291/289**, `git diff --cached --check`, staged stat/status; commit/push checkpoint only after Windows PASS. H1-C3H should quantify measured total setup + calibration cost, independent-run confidence intervals and real mixed-field FEM/PCG. No release/tag/publish or production automatic routing.

## 2026-10-16 H1-C3H candidate — complete MPI probe/decision cost ledger (PENDING)

- Baseline: `7ee08a4` on clean `develop/0.9.0`; H1-C3G measured-calibration checkpoint committed/pushed. New **example-only** `mpi_root_cost_ledger_bench`; no library/solver changes.
- Record whole-MPI calibration probe (initial baseline Bcast, root graph exploration, both native setups, 40 paired samples), final scoped selection including second graph search/Bcast, separate selected native setup, and three **paired independent held-out epochs** of baseline/Graph/selected.
- Compare modeled baseline-only versus calibration-inclusive selected total; graph selected only after measured guarded selection; N=1 fallback is valid. Neither modeled full-N total nor the inferred break-even is an actual N-iteration solve; no real FEM input, PCG, or statistical generality.
- **PENDING**: Windows package Preflight/Apply, `cargo fmt`, `cargo clippy --features mpi --all-targets -- -D warnings`, release example, native MS-MPI 1/2/4/8 N=10000 and 2/4/8 N=1; independent launch repeats, release regressions, source integrity **293/291**, staged diff. No commit/push until logs reviewed; no release/tag/publish/production auto-route.

## 2026-10-16 H1-C3H actual Windows/MS-MPI validation — supersedes H1-C3H PENDING

- Based on committed/pushed `7ee08a4` (`develop/0.9.0`); H1-C3H example/doc changes staged, not committed at validation time. Native `mpi_root_cost_ledger_bench` only, with no production solver/routing changes. Initial Windows Preflight/Apply, rustfmt, strict MPI Clippy, Release MPI build, native 1/2/4/8 N=10000 and 2/4/8 N=1 **7/7 PASS**. N=1 on multi-rank correctly falls back; 1 rank stays Equal.
- Fresh MPI launches **15/15 PASS** (2/4/8 ranks x 5 at N=10000); all numeric CSR/Halo reference errors zero and `workers_global_csr=0`. Graph selection 2r **5/5**, 4r **4/5** (one `NoCalibratedSpeedup`/Equal), 8r **5/5**. Three-epoch all-Graph-faster count 5/5, 4/5, 5/5. Fully accounted *modeled* break-even median: **8,509 / 3,365 / 2,234** calls (2/4/8; 4-rank median among selected runs only). Ranges: **6,848–10,795 / 2,619–3,948 / 1,939–3,573**. At expected N=10000, probe-inclusive model favored selected path only **4/5 / 4/5 / 5/5** launches. Two-rank one GraphSelected launch exceeded payback (break-even 10,795); four-rank one failed calibration while later holdout favored Graph median, highlighting instability.
- Calibration/holdout timing ratios remain noisy: median **2.294x / 2.120x / 1.400x** at 2/4/8 ranks. These are short synthetic SpMV timings, not measured full-N runs nor real FEM/PCG. Initial probe, second selector, selected native setup are separately timed to avoid double charging, but production ingestion, solver, MPI startup and matrix redistribution are excluded. No production automatic selector or proven end-to-end speedup.
- Full non-MPI Release regression (`hybit-distributed` 45, `hybit-matrix` 52, `hybit-ffi` 5, HyBIT public API & doctest) PASS; `hybit-distributed --features mpi --lib` **72/72 PASS**; Source Integrity **293/291 PASS**; staged diff check PASS. Next action: append results to docs, rehash docs/handoff, rerun gates, then commit/push ONLY checkpoint. Next H1-C3I noise-aware calibration and representative FEM + MPI-PCG. No release/tag/publish.

## 2026-10-17 H1-C3I candidate — paired-epoch robust admission (PENDING)

- Baseline `dd1c9ad` pushed/clean on `develop/0.9.0`; H1-C3H `15/15` fresh native launches and MPI-feature 72/72 release unit tests PASS, Source Integrity 293/291 before H1-C3I.
- Propose `robust_cost.rs` as opt-in, MPI-independent, fail-closed economic preflight: >=3 paired epoch median max-rank timings, require positive gain in every epoch, compare minimum epoch gain * expected calls against full probe+selector-budget+native-setup delta and 25% preparation contingency.
- The native `mpi_robust_partition_bench` retains the H1-C3H synthetic MPI numerical/Halo cross-check and independent holdout, splits calibration into 4 epochs, and withholds scoped evidence on veto. It never routes production solver calls automatically. Caller budget is a **forecast**, not an upper bound; compare actual second selector against budget in logs.
- **Windows/MS-MPI compilation and execution PENDING:** expected Source Integrity **296/294**; nine new unit tests; strict Clippy, release regressions, 7 native MPI invocations, independent repeats required before commit/push.
- Next action: apply H1-C3I package Preflight then Apply on clean exact `dd1c9ad`; run tests and send logs for numeric/cost analysis. Do not tag/release/publish.

## 2026-10-17 H1-C3I native MS-MPI results — supersedes H1-C3I PENDING

- Baseline `dd1c9ad` on `develop/0.9.0`, H1-C3I opt-in robust veto/example staged but **uncommitted** during verification; no solver default, MPI transport or ABI change. Package Preflight/Apply, Rustfmt, MPI-feature strict Clippy, Release example and MPI-feature unit tests **81/81 PASS**; Source Integrity **296/294 PASS** before results-doc update.
- Native Windows MS-MPI initial **7/7 PASS** (N=10,000 1/2/4/8 ranks and N=1 2/4/8 ranks), independent fresh N=10,000 launches **15/15 PASS** (2/4/8 ranks × five), and additional modeled N=1,000,000 **3/3 PASS**; combined **25/25** native launches. All reported numerical CSR/Halo errors zero, no worker global CSR, selector budget never exceeded.
- N=10,000 robust Graph admitted/selected: **0/5 2r**, **0/5 4r**, **4/5 8r**. In admitted 8-rank cases, all four selected Graph and had lower probe-inclusive **modeled** total than within-run baseline. Rank-8 fifth run conservatively vetoed at required 12,109 calls > 10,000; modeled fallback cost was worse due to probing. All 4-rank held-out epochs favored Graph in **5/5** launches despite gate veto: high potential for false-negative/over-conservative selection.
- N=1,000,000 is **only an expected-call extrapolation**, not full-N MPI SpMV or PCG execution. 2/4 ranks still fail a paired epoch; 8 ranks selects Graph with guard `Matched` and numerical PASS.
- IMPORTANT remaining gate before commit/push: full non-MPI Release regression for distributed/matrix/hybit/ffi, final Source Integrity **296/294** and staged diff. Do not promote production auto-routing, tag, release or publish. Next priority H1-C3J **reuse ABTM Graph candidate to eliminate double exploration**, then real mixed-field FEM MPI-PCG and memory/time-to-solution comparison. Details in `docs/H1C3I_ROBUST_CALIBRATION.md`.

## 2026-10-17 H1-C3J proposal reuse candidate (Windows validation PENDING)

- **Required baseline:** clean `develop/0.9.0` HEAD `a129ee7` (H1-C3I push verified); Source Integrity 296/294.
- Candidate H1-C3J adds opt-in typed `PreparedCostGraphCandidate` with matching CSR numerical/structural, field-layout, rank/radius, baseline-kind, and environment identity. The guarded selector has an optional already-prepared graph input; it otherwise retains its previous implementation and auto selection remains unchanged.
- Root-only native MPI comparison example `mpi_root_graph_reuse_bench` reuses one ABTM Graph proposal across calibration and final guarded selection; unit tests cover legacy equality and fail-closed reuse.
- **Windows/MS-MPI gates NOT YET RUN.** Verify graph/legacy partition equality, 1/2/4/8-rank MPI numeric/traffic correctness and second selector wall time before commit/push or performance claims. Expected Source Integrity 298/296.
- Next one action: run `APPLY_HYBIT_H1C3J.ps1 -PreflightOnly` and `-Apply`, then strict Windows build/tests and native MPI runs; report logs for review.
## 2026-10-17 H1-C3J Windows/MS-MPI validation — supersedes PENDING

- Baseline `a129ee7` on `develop/0.9.0`; H1-C3J staged changes were
  validated but were **not** committed at the time of the supplied logs.
- Preflight/Apply, `cargo fmt`, strict MPI-feature Clippy, MPI lib unit tests
  **81/81**, prepared reusable-Graph integration **5/5**, initial native
  MPI **10/10**, interleaved C3I/C3J native A/B **30/30**, and non-MPI
  release regressions (`hybit-distributed` 54, matrix 52, FFI 5, public
  API and doctests) PASS. All native reported CSR-reference errors zero;
  all workers had no replicated global CSR; Source Integrity **299/297**,
  staged diff gate PASS. Ten expected files staged, no commit/push yet.
- N=10000 A/B Graph selected C3I vs C3J: 2r **0/5 vs 0/5**;
  4r **0/5 vs 0/5**; 8r **3/5 vs 5/5**. For 8r runs 2/4/5 where both
  selected Graph, second-selector wall reductions **0.1605/0.1704/0.1405
  ms**, median **0.1605 ms**. Reuse code-path equivalence PASS; fixed
  `graph_searches=1` diagnostic is not independent instrumentation.
- Initial Graph preparation medians 2/4/8: C3I **0.0769/0.1204/0.1374 ms**,
  C3J **0.1448/0.2085/0.1992 ms**: caching does **not** establish faster
  overall wall time. Synthetic 168-DOF data, modeled N only; no real-FEM
  end-to-end speedup, full MPI-PCG solve, memory peak, or default adoption.
- **Next:** review this validation documentation, rehash, run source gate and
  check stage; then user-approved commit/push of H1-C3J. Continue with real
  FEM/MPI-PCG + ABTM reuse overhead investigation. No release/tag/publish.

## 2026-10-17 H1-C3K1 candidate — real FEM MPI-PCG baseline (validation pending)

- Starting checkpoint: `b28dd0c` on clean `develop/0.9.0`, pushed and
  user-confirmed; Source Integrity 299/297.
- K1 reuses `mpi_mtx_profile` to avoid duplicating MPI-PCG logic, changes
  only its measured-run numerical gate to cover ALL measured runs and keep
  worst true residual / solution max-error, preserves CSV schema.
- New `benchmarks/scripts/bench-h1c3k1-real-fem-mpi.ps1` validates .mtx
  dimensions/NNZ cap, runs 24-DOF certified identity control 1/2/4/8 ranks,
  then a user-selected real FEM Matrix Market matrix with independent MPI
  launches 1/2/4/8 and rank-local CSR/halo PCG. Results in CSV/log.
- Explicit `ExperimentalUnverified` plus `-PermitUnverifiedSpd` is *not* SPD
  certification. No graph/field partition, physical field metadata, ABTM
  speedup or peak memory claims at K1.
- **Windows native Rust, Clippy, real FEM MPI runs PENDING**. Do not commit
  until results are reviewed and numerical gates PASS. Expected 301/299
  Source Integrity. Next: gate K1, then K2 Equal/Field/Graph ownership bridge.

## 2026-10-17 H1-C3K1-F2 precision diagnostic (uncommitted)

- Base HEAD remains `b28dd0c`; original H1-C3K1 staged, Source Integrity 301/299.
- Native `thermal1` 82,654 DOF, 574,458 nnz, PCG status Converged at 1304 iterations with true rel residual about 1e-9, but max manufactured error about 7.9e-7 FAIL versus unchanged 2e-7 cutoff in every 1/2/4/8 x3 run. Certified 24-DOF smoke 4/4 PASS.
- K1-F2 adds opt-in PCG relative tolerance to the experiment and launcher; default remains 1e-9. Diagnostic label clarifies residual-only success. Numerical acceptance still requires both correct true residual and original manufactured absolute error, now across all measured repetitions. `ExperimentalUnverified` never certifies SPD.
- **Pending Windows validation**: F2 preflight/apply, fmt/clippy, MPI tests, actual `thermal1` at rtol 1e-11, status/errors and steady-state 1/2/4/8 timing. Do not commit or push until reviewed.

## 2026-10-17 H1-C3K1-F2 strict real-FEM baseline — numerical validation completed

- Supersedes earlier H1-C3K1-F2 **pending** status. Still uncommitted on clean
  baseline HEAD `b28dd0c` plus exactly six staged K1 files. Source Integrity
  **301/299 PASS**. MSVC+MS-MPI / Clang 23.1.3 environment restored,
  strict Clippy PASS; earlier non-MPI regressions and 81/81 MPI tests PASS.
- `thermal1.mtx`: 82,654 DOF, expanded nnz 574,458, Equal-owned CSR,
  Jacobi MPI-PCG, experimental SPD **unverified**. At rtol=1e-9,
  12/12 real-matrix runs status Converged but max error about 7.9e-7
  exceeded unmodified 2e-7 gate: valid FAIL.
- K1-F2 rtol=1e-11, max_iters=5000: 1/2/4/8 ×3 independent MPI runs,
  warmups=3, measured=5 per launch: **12/12 actual FEM NUMERICAL_PASS**,
  plus certified control 4/4 PASS. Max manufactured error about 5.7e-9,
  true relative residual about 9.8e-12. No SPD proof implied.
- Median of independent launch solve medians (ms), 1/2/4/8:
  **1485.4827 / 834.7466 / 448.6422 / 297.4493**, giving 8r speedup
  **4.9941x**, 62.4% parallel efficiency. PCG iterations 1564/1564/1563/1563.
  These are solver-only timings; peak RSS not measured. Raw results:
  `D:\Work\h1c3k1_results\k1_summary_20261017T100505439Z.csv` and
  `k1_runs_20261017T100505439Z.log`.
- **Next:** H1-C3K1 final source/test/stage gates and user-controlled
  commit/push. Then H1-C3K2 real-matrix Equal/Field/ABTM Graph comparisons,
  preserving exact RHS, rtol, Jacobi, MPI and total-cost accounting.
  No tag/release/crates.io publish; no production auto selection.

## 2026-10-17 H1-C3K1 committed / Windows MSVC + Clang + MS-MPI invariant

- H1-C3K1 **committed and pushed**, `develop/0.9.0` HEAD **`cbbcdfa`**,
  clean local/remote, Source Integrity **301/299 PASS** at K1 checkpoint.
  Rustfmt, Clippy, non-MPI Release regression, 81/81 MPI-feature lib tests,
  and real `thermal1.mtx` strict numerical gate (12 FEM + 4 certified
  control) PASS. No release/tag/crates.io publication.
- **Before ALL future MPI Cargo build/test commands**, in **each new
  PowerShell session** run:
  `. .\tools\env\Enter-HybitMsMpi.ps1` from `D:\Work\hybit\hybit`.
  Reference: `docs/WINDOWS_MSVC_MSMPI_ENV.md`.
- **Fixed:** Visual Studio 2026 Community Developer PowerShell 18,
  `cl.exe` MSVC x64, LLVM/Clang 23.1.3 `C:\Program Files\LLVM\bin`,
  `CLANG_PATH=C:\Program Files\LLVM\bin\clang.exe`,
  `LIBCLANG_PATH=C:\Program Files\LLVM\bin`; `LIBCLANG_PATH` is not
  `PATH`. MS-MPI executable **`C:\Program Files\Microsoft MPI\Bin\mpiexec.exe`**,
  SDK `MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include\`,
  `MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\`.
  **Do not substitute MinGW or Open MPI.** An uninitialized MSVC shell
  causes `mpi-sys`/`bindgen` `libclang error` and missing `VCINSTALLDIR`,
  `INCLUDE`, `LIB`; this was an environment failure, not a source regression.
- Next H1-C3K2: compare real-FEM Equal baseline vs ABTM Graph; physical
  Field-aware DOF metadata must be supplied separately (Matrix Market does
  not encode it). Do not enable production auto Graph selection by default.

## 2026-10-25 H1-C3K8-D9-C8A — rankwise MS-MPI PCG cause profile (native verified)

- **Repository/ownership:** `develop/0.9.0`; validated against clean pre-experiment HEAD `f898be5fd397ed1dba9d01d1830f9509ffe80b72`. During validation, precisely two **unstaged** tracked files differed: `crates/hybit-distributed/examples/mpi_k8d7_prepared_p2p.rs` and `SOURCE_SHA256.txt`. The final D9-C8A Rust example SHA-256 after Rustfmt/Clippy FIX1 is `bd9fed27c89968f232d3ba8e5c86890c2e67d5e62e17fdefa411c8342eb0d9a1`. Commit/push have **not** yet been demonstrated by the supplied logs; replace this checkpoint state with actual Git status/HEAD after committing.
- **Scope:** read-only rankwise phase diagnostics added to the existing D7 native-MPI experiment. Original CSR matrix coefficients, prepared Halo, Jacobi-PCG, true-residual/reference verification, Alltoallv/P2P transports and the production solver are unchanged. Rankwise numeric, pack/unpack, preconditioner, dot, vector-update and elapsed telemetry is gathered outside timed solve; the additional DOF/NNZ workload vectors are gathered at setup.
- **Environment:** user's Windows 11 x64, Ryzen 7 7800X3D, Rust 1.99.1 `x86_64-pc-windows-msvc`, LLVM/Clang 23.1.4, Visual Studio 2026 developer environment for MSVC toolchain, **Microsoft MS-MPI only**. FFI use of separate GCC/G++/GFortran is unrelated. No MinGW/OpenMPI backend was used.
- **Gates:** source-integrity **374 manifest / 372 hashes PASS**; Rustfmt PASS; strict workspace/all-targets/all-features Clippy PASS; 16/16 D7/D8 release example tests PASS; release build PASS. Original D9-C8A payload had two Rustfmt diffs and two Clippy lints; FIX1 resolved them. Original native runner's hardcoded pre-FIX1 SHA rejected the new source; runner FIX1 corrected that check. Native 4-rank Equal smoke **4/4 PASS**, 8-rank Multilevel smoke **4/4 PASS**, full paired native MS-MPI ABBA **48/48 PASS**. Full post-change workspace regression and post-push CI are **not** claimed here.
- **Matrix/math:** `D:\Work\thermal1.mtx`, 82,654 DOF, original CSR nnz=574,458. Its SPD screen is **experimental / not a mathematical SPD certificate**. Jacobi PCG, `tol=1e-11`, max-iters=2,000, 4/8 ranks, Equal/ABTM Multilevel, 3 rounds per rank/policy of `Alltoallv/P2P/P2P/Alltoallv`, independent native launches; warmups=2, measured solves=5 per launch. Per transport 6 independent launches per rank/policy. All 48: iterations=1,586; original/distributed true relative residual `4.959126313934e-12`; maximum manufactured-solution error `2.173250024384e-9`; D7 bitwise prepared-Halo/original-CSR and exact-root provenance PASS. Rank DOF/NNZ and received ghost counts reconcile across all ranks.

| Ranks | Owner partition | Alltoallv median solve (ms) | P2P median solve (ms) | A/P speed ratio |
|---:|---|---:|---:|---:|
| 4 | Equal | 392.61705 | 381.16820 | 1.03004 |
| 4 | ABTM Multilevel | 358.82350 | 367.86675 | 0.97542 |
| 8 | Equal | 274.03790 | 263.55570 | 1.03977 |
| 8 | ABTM Multilevel | 267.99825 | **234.84810** | **1.14116** |

- **Primary D9-C8A observation:** at 8 ranks + ABTM Multilevel, median rank-max SpMV Halo-exchange phase: Alltoallv `41.3109 ms` vs P2P `8.6037 ms` (difference `32.7072 ms` over 1,637 SpMV calls). PCG medians differ `33.15015 ms` and all three within-round A/B pairs favor P2P (savings `28.760/36.118/33.517 ms`). **Inference:** frequent small-MPI-call transport/synchronization overhead is likely material here; phase timers include posting/waiting, not pure wire time, and independent rank maxima are not additive.
- **ABTM topology effect:** 4 ranks total received ghosts Equal `14,035` vs Multilevel `2,053` (−85.4%); 8 ranks `23,085` vs `3,687` (−84.0%). 8-rank P2P Equal-to-Multilevel median solve saving `28.7076 ms`, but Multilevel median partition cost approximately `224.88 ms` vs Equal approximately `0.04 ms`. Ignoring other setup deltas, partition amortization requires roughly eight prepared solves; **end-to-end/peak RSS and true reused solve-many crossover remain unverified**. 4-rank Multilevel P2P is slower than Alltoallv by `9.04325 ms`; do **not** set universal P2P or Multilevel production defaults.
- **Evidence outside Git (not in manifest):** `D:\Work\k8d8a_thermal1_raw.csv` (48 independent-launch records with rankwise vectors), `D:\Work\k8d8a_thermal1_comparison.csv` (four summary conditions) and individual `.log`/`.csv` files. Source-of-truth for performance is these user-provided files and the native Windows log, not this summary alone. Only **solver-only** times were compared; no global end-to-end production-speedup, SPD certification, cross-matrix generalization, memory-peak, or multinode claim.
- **D9-C8B design constraint:** the pre-existing `mpi_k5b_assignment_solve_many.rs` demonstrates repeated RHS orchestration but explicitly uses a **synthetic SPD surrogate**, not original Matrix Market coefficients. Reuse its experiment structure where helpful, but extend the validated D7 original-CSR prepared-MPI path for genuine `thermal1` solve-many measurement and retain original-CSR/true-residual checks. Measure N={1,4,8,16} independent RHS with setup amortization, 4/8 ranks, Equal/Multilevel, P2P/Alltoallv, persistent bytes and peak RSS if feasible; report time-to-solution including partition/halo/preconditioner preparation separately from solver-only medians.
- **Next one action:** validate this handoff addition and its Source Integrity hash with the existing two D9-C8A tracked changes, then obtain user-approved small checkpoint commit/push before implementing D9-C8B. No tag, release or crates.io publish.

## 2026-10-25 H1-C3K8-D9-C8B: original CSR prepared solve-many / MS-MPI checkpoint

### Scope and provenance

- Branch: `develop/0.9.0`; base HEAD before this checkpoint: `e6e717039ecf843be5fcc87c15068b95df952195`.
- The only Rust source changed for D9-C8B is `crates/hybit-distributed/examples/mpi_k8d7_prepared_p2p.rs`, SHA-256 `aa2a6f232f63536c6b14c04461eb7f4e8ef3ee44aa2641ccea5d59f839500297`.
- Opt-in `--solve-many 1|4|8|16` uses the original MatrixMarket CSR coefficients, independently constructed deterministic manufactured solutions, and bitwise cross-transport RHS verification. Existing D7 CLI remains usable. This is a benchmark/example checkpoint, **not** a production solver selection change.
- Native Windows MS-MPI, LLVM/libclang and Rust 1.99.1; no Open MPI or MinGW MPI. The `tools/env/Enter-HybitMsMpi.ps1` environment is used in the same PowerShell process. Visual Studio Developer Shell initialization in that environment file is a build-environment implementation detail, not a request to change MPI backend or FFI policy.
- User-observed gates: source integrity 374 manifest / 372 verified PASS, Rustfmt PASS, Clippy `--workspace --all-targets --all-features -- -D warnings` PASS, and 18/18 release example tests PASS.
- Matrix: `D:\Work\thermal1.mtx`, n=82,654, nnz=574,458; eight MPI ranks; Jacobi-PCG, relative tolerance 1e-11; `--experimental-large-spd` uses a bounded experimental curvature screen, **not** a universal SPD certificate.

### Native measured evidence

- Earlier `k8d8b_smoke8_compare` (8 ranks, RHS 1/4) and `k8d8b_rhs8_16` (8 ranks, RHS 8/16) ABBA experiments all passed; the latter initially suggested an ABTM Multilevel + P2P prepared-time gain at RHS 8/16.
- The independent 48-launch `k8d8b_crosscheck` (Equal first, then ABTM Multilevel; RHS 4/8/16; 2 rounds) was PASS in all cases. P2P estimated prepared totals (Equal vs ABTM): RHS 4 = 1,078.47 vs 1,198.95 ms; RHS 8 = 1,957.23 vs 2,266.45 ms; RHS 16 = 3,835.30 vs 4,117.40 ms. Thus it **did not reproduce** the earlier multilevel absolute-time advantage.
- The 32-launch `k8d8b_reverse_policy` (ABTM Multilevel first, then Equal; RHS 8/16; 2 rounds) was PASS in all cases. P2P estimated prepared totals (Equal vs ABTM): RHS 8 = 2,386.46 vs 2,283.16 ms; RHS 16 = 4,586.95 vs 4,317.09 ms. Reversing policy measurement order **reversed the winner**.
- Across the Equal-first and ABTM-first campaigns, P2P profiled batch times changed: at RHS 8, Equal 1,830.30 -> 2,251.14 ms (+23.0%) while ABTM 1,905.43 -> 1,938.68 ms (+1.7%); at RHS 16, Equal 3,708.46 -> 4,457.41 ms (+20.2%) while ABTM 3,782.33 -> 3,981.49 ms (+5.3%). The actual cause of this order-related variation remains unproven.
- Original-coefficient verification and cross-transport bitwise flags were true for every 48+32 launch; maximum observed root true relative residual approximately 5.0e-12, maximum root solution error approximately 4.84e-9. Equal halo ghosts=23,085; ABTM Multilevel halo ghosts=3,687 (84.0% fewer).
- P2P remained faster than Alltoallv *within either partition policy* in both campaigns. This communication-mode result is promising; **do not** promote an ABTM absolute-speedup or production performance claim yet.
- `estimated_prepared_time_to_solution_ms` combines measured setup and profiled batch, whose latter includes post-solve diagnostic collectives. It is **not** the measured whole-process wall time; RHS manufacturing and root verification are tracked separately. Do not treat these as interchangeable metrics.
- Raw CSVs, transport summaries and policy summaries reside in `D:\Work` under the above experiment prefixes; they are not part of the Git source checkpoint.

### Hardware and scheduling hypotheses for the next session (not established causes)

- Local CPU: AMD Ryzen 7 7800X3D, 8 physical cores / 16 threads, 96 MB shared L3 cache. An eight-rank native MS-MPI run fills the eight physical cores if each rank occupies one physical core. Cache reuse/eviction and rank migration may materially change original-CSR SpMV performance.
- Current `mpiexec` experiments **did not explicitly set or record processor affinity**. MS-MPI documents `mpiexec` affinity controls, but no affinity mode or rank-to-core placement has been verified for this machine. Avoid claiming the default placement or choosing logical-CPU masks without first checking `mpiexec -help` and observed placement. First instrument, then compare unpinned with a supported pinned configuration.
- Communication/computation overlap has **not** been implemented or demonstrated in the tested halo + SpMV path (per current development understanding). Examine the actual operator: a potentially beneficial implementation is initiate nonblocking ghost exchange, compute strictly local/interior rows, wait or progress communication, then compute halo-dependent/boundary rows. This requires a correctness-preserving row split and halo-workspace lifetime discipline.
- Nonblocking MPI initiation by itself does not guarantee useful overlap. Check MS-MPI communication progress, per-rank imbalance, fraction of interior SpMV work, MPI wait time, and net solve-only *and* prepared total performance before claiming a benefit.
- Cache behavior, core affinity, MPI progress, and row-work imbalance are hypotheses only. No cache misses, clock counters, rank placement, or overlap traces were measured in D9-C8B.

### Next session: D9-C8C plan and stop conditions

1. Do **not** rerun the same grouped-policy ABBA benchmark again as proof of policy superiority. Build a balanced cross-over launch schedule, interleaving Equal and ABTM and rotating Alltoallv/P2P positions; save schedule and per-launch order in CSV.
2. Establish rank-to-logical-CPU mapping and measured affinity, and record process/thread count. Benchmark unpinned first, then only validated MS-MPI affinity modes. Keep the 8-core/96 MB L3 constraint explicit.
3. Use the existing D9-C8A per-rank timing counters to correlate slowest rank, numeric skew, exchange skew, owned DOFs/nnz and halo sizes with observed runtime variability. Avoid new solver changes until this cause profile is reviewed.
4. Only then evaluate communication/computation overlap with baseline and opt-in A/B, using exact original CSR verification, identical RHS and iterations, rankwise timing, and whole-process wall-time alongside solver-only timing.
5. Maintain full source SHA-256 integrity, branch/HEAD/clean staging guards, Rustfmt, Clippy all-features, 18 example tests and native MS-MPI tests for any follow-up patch. This checkpoint is **not a release**; do not change production defaults.

### Reference sources used for the hypotheses

- AMD official CPU specifications: https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-7-7800X3d.html
- Microsoft mpiexec command and affinity options: https://learn.microsoft.com/en-us/powershell/high-performance-computing/mpiexec?view=hpc19-ps
- MPI Forum nonblocking communication and progress semantics: https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/node77.htm

## 2026-10-26 H1-C3K8-D9-C8G: native MS-MPI PCG certification and iteration-limit checkpoint

### Source, validation, and intended scope

- Branch `develop/0.9.0`, base HEAD `bd72afa5a1c17556f63095c3ab0f0ce9283dffc4` before this checkpoint. The modified D7 example is `crates/hybit-distributed/examples/mpi_k8d7_prepared_p2p.rs`, SHA-256 `328797c3a2765885a0afba6e7349126176e552dc76b153c359617f5a7872422b`. The accompanying `SOURCE_SHA256.txt` entry is updated.
- D9-C8D introduced per-rank solve-many phase reporting (rank numeric/exchange skew). On `thermal1.mtx`, the four native P2P cases (Equal/Multilevel x RHS8/16) passed; profiled batches in milliseconds were Equal 2160.992/4441.752 and Multilevel 1923.392/3936.087. Profiling alone does not establish physical-core placement, overlap, or a cross-launch production speedup.
- D9-C8E added failure-only post-PCG certification metrics. The two original-CSR candidates `cant.mtx` and `boneS01.mtx`, with Jacobi-PCG 8 ranks, `tol=1e-8`, `max-iters=3000`, had acceptable true residual but failed the independent absolute manufactured-solution error gate (`1e-5`): `cant` iteration 1871, true relative residual `4.725116546447e-9`, max absolute error `9.643888811329e-3`; `boneS01` iteration 1324, `4.934922631861e-9`, `1.255478137705e-4`.
- D9-C8F diagnostic: `boneS01` **passed both gates** at `tol=1e-10`, `max-iters=3000`, Jacobi, after 2147 iterations (true relative residual `4.886327334525e-11`, max error `9.123679092782e-8`). The initial `max-iters=6000` trial was rejected by an existing K8-B guard and is **not** a convergence result. `cant` at `tol=1e-10` did not converge within 3000 iterations.
- D9-C8G adds a **failure-only** global recursive/independent true residual and manufactured-solution error census at the iteration cap. The successful solver path, algorithm, target tolerance, acceptance policy, original CSR and selected preconditioner implementations remain unchanged. Extra MPI reductions on the failure path make that diagnostic path invalid for speed benchmarking.
- Native Windows Rust 1.99.1/MSVC/MS-MPI source integrity `374 manifest / 372 verified` PASS, Rustfmt PASS, strict workspace all-target/all-feature Clippy PASS, and D7 example `20/20` release tests PASS after D9-C8G. The three native MPI diagnostic launches were completed with their expected failure classification `ITERATION_LIMIT_MEASURED`. They are **not** successful certified solves.

### D9-C8G original CSR numerical evidence — `cant.mtx`

Common conditions: Windows native Microsoft MS-MPI (not MinGW or Open MPI), 8 ranks, Equal partition, P2P halo, original MatrixMarket CSR, one manufactured RHS (`rhs_id=0`), `--experimental-large-spd` opt-in, max iterations 3000, target relative tolerance `1e-10`.

| Preconditioner | Iterations | Recursive relative residual | Independent true relative residual | Maximum absolute error | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Jacobi | 3000 | 6.996658372331e-11 | 6.996658368670e-11 | 1.179879512210e-3 | Iteration limit; certification not met |
| Rank-local IC(0), block 4 | 3000 | 3.401287201384e-9 | 3.401287217356e-9 | 8.913498652824e-3 | Iteration limit; certification not met |
| Rank-local dense block Cholesky, block 32 | 3000 | 2.386277485730e-9 | 2.386277501762e-9 | 3.181249016524e-3 | Iteration limit; certification not met |

Evidence: `D:\Work\k8d8g_cant_iter_limit_summary.csv`, exact observed SHA-256 `d201a7ca5bf00923da2b574a6d497307b79ef64aed5d824af9d9f1d011d1854e`. Native three-rank-group logs are retained in `D:\Work` with the prefix `k8d8g_cant_iter_limit`; the CSV/logs are not included as source-controlled files.

Interpretation: the recursive and independently computed true residuals agree within a few parts in `1e9`; there is no evidence of material *residual drift* in these three experiments. Jacobi has the smallest residual and smallest maximum error of the tested arms, but its residual `6.997e-11` still exceeds the implementation's internal stopping threshold `0.5 * tol = 5e-11`, and its max error exceeds `1e-5` by about 118 times. IC(0) block 32 previously failed on a nonpositive local pivot; block Cholesky 32 prepared successfully but failed to converge. Larger local blocks must not be assumed automatically better or production-ready. This is matrix/configuration-specific evidence, not a proof of global SPD or non-SPD.

The `--experimental-large-spd` curvature screen is not an unrestricted SPD certificate. Do **not** make the current `cant` run pass by relaxing the forward-error gate, replacing the original matrix, silently raising the iteration limit beyond the guarded maximum, or conflating residual and solution error.

### CPU topology and performance caveats still open

- Host AMD Ryzen 7 7800X3D: 8 physical cores / 16 hardware threads, 96 MB L3. D9-C8C observed 8 MPI worker PIDs and OS process-affinity masks of `0x000000000000FFFF`, including with requested `seq:P` and `spr:P:L` layouts, but **did not** prove rank-to-PID mapping, actual CPU residency, or successful physical-core pinning. Do not cite the affinity probes as speedups; they intentionally sampled during computation.
- D9-C8B original-CSR ABBA showed P2P advantages over Alltoallv in multiple eight-rank/RHS configurations, but Equal versus ABTM Multilevel prepared-time *winner changed* with order of policy campaigns. Cross-launch absolute superiority is not established.
- Communication/calculation overlap has not been validated in the D7 halo+SpMV path. Nonblocking MPI alone does not demonstrate overlap. Optimize after correctness gates and controlled rankwise timing, not as an unmeasured release claim.
- Time labels remain distinct: `profiled_batch` includes diagnostic collectives; `estimated_prepared_total` is setup plus median profiled batch, not whole-process wall. Native C8G failure-only diagnostics cannot be compared as solver speed benchmarks.

### 0.9.0 release-gate priorities (not yet a release PASS)

1. **P0 — source and regression quality:** source integrity, Rustfmt, strict Clippy, release workspace tests/doc tests, MPI-enabled D7 20 tests; negative SPD/parameter/preconditioner failure tests; check Rust-2024 FFI unsafe/ABI contracts and package/version/CI consistency. Re-run gates at the release candidate SHA.
2. **P0 — numerical matrix contract:** maintain independent root/original-CSR and distributed true residual + manufactured-solution gates. Classify `thermal1`, synthetic FEM, and `boneS01` with their exact passing matrix/rank/tolerance/iteration conditions. Keep `cant` as a documented **known nonconvergent configuration**, not a production PASS or blanket unsupported matrix classification. Test both P2P and Alltoallv numerically.
3. **P1 — measured absolute MPI performance:** baseline 1-rank against 2/4/8 ranks (and applicable CPU-thread baseline) on several eligible original-CSR matrices, then compare Equal, ABTM and external labels fairly. Report whole-process wall, setup, solver-only, profiled batch, memory, accuracy, iteration count, per-rank skew, and policy launch order. No claim based only on halo cut/ghost reduction.
4. **P1 — CPU placement control:** map rank -> PID -> allowed logical CPU -> actual scheduling, validate a supported MS-MPI affinity policy, and repeat balanced randomized/crossover launches in the controlled configuration. AMD L3/cache and Windows scheduling remain possible confounders until measured.
5. **P2 — optional optimization after release correctness:** assess interior/boundary row split, actual communication progress and numerical overlap; consider global coarse correction / AMG for slow low-frequency components on harder matrices. Preserve current production defaults until independent correctness and absolute-time improvements justify promotion.
6. **P0 — publication gate:** verify public README/API, features, license, changelog, mdBook, source-manifest, clean Git tree, CI, crate dry-run, and release notes; preserve 0.8.0 stability and avoid announcing a 0.9.0 production speedup without the above evidence. Final tagging/publishing is a separate explicit step.

Checkpoint action: append this evidence to the existing `docs/SESSION_HANDOFF.md`, update its SHA-256 entry, verify exactly the three modified files (handoff, SHA manifest, D7 Rust example), then run native-enabled source/test gates and make a focused commit/push. The checkpoint is not a 0.9.0 release approval.

## 2026-10-26 HyBIT 0.9.0 release gates P1–P3C (MS-MPI)

Checkpoint base: `develop/0.9.0` at
`b51d84c68653b41f729b1080d4ba6f5fc14b183f`. The tested D7 example
(`crates/hybit-distributed/examples/mpi_k8d7_prepared_p2p.rs`) has SHA-256
`328797c3a2765885a0afba6e7349126176e552dc76b153c359617f5a7872422b`.
No solver algorithm or source files were changed for P1–P3C release gates.
Windows toolchain was MSVC + LLVM + native MS-MPI; no MinGW/Open MPI.

### Release P1: static quality

All four gates passed: Rustfmt, workspace-wide all-features Clippy with
`-D warnings`, release workspace tests, and release MPI D7 example tests.
The source integrity gate reported 374 manifest files, 372 SHA-256 checks,
and PASS.

### Release P2: original CSR numerical validation

All 12 MS-MPI cases passed independent root/distributed numerical gates:
`k8d1_fem64_shift002` at tolerance `1e-8`, `thermal1` at `1e-11`, and
`boneS01` at `1e-10`; 8 ranks, 1 RHS, Jacobi PCG; for both `equal` and
`multilevel` partition policies, and both P2P and Alltoallv transports.
The specified tests required the original coefficients of each supplied
matrix (without adding a benchmark-time surrogate shift), and verified true
residual and manufactured-solution error.

### Release P3-A: single-node rank scaling, native-default layout

`thermal1.mtx`, SHA-256
`260f4a65c7dd22ce6e87b1035169c14d53fbe44639f0005e46207e057fce70ee`,
8 manufactured RHS, P2P, 1/2/4/8 MS-MPI ranks, four observations per
policy/rank configuration, all 32 numerical/profiling cases passed.
The median **instrumented prepared-batch** times (ms) were:

| Policy | 1 rank | 2 ranks | 4 ranks | 8 ranks | 1-to-8 rank batch ratio |
|---|---:|---:|---:|---:|---:|
| equal | 10422.12 | 7091.69 | 3889.61 | 2156.48 | 4.83 |
| multilevel | 10597.51 | 6898.31 | 3770.44 | 1939.96 | 5.46 |

These ratios are rank-scaling diagnostics with instrumentation, not proof of
production speedup; CPU affinity was unverified.

### Release P3-B: 8-rank CPU sampling, NOT solver-thread residency

Each MPI launch captured unique PMI rank-to-PID mappings. All three layouts
reported broad process masks `0x000000000000FFFF` and zero singleton process
masks. However, the **PowerShell sampler main thread** visited 14–15 logical
CPUs per rank with `NativeDefault`, versus exactly one distinct logical CPU
per rank for all 300 samples under either `seq:P:L` or `spr:P:L`.
For ranks 0–7, observed CPU numbers were:

- `seq:P:L`: 14, 12, 10, 8, 6, 4, 2, 0.
- `spr:P:L`: 14, 6, 10, 2, 12, 4, 8, 0.

The observed sampler placement is meaningful but is **not** a hardware
physical-core mapping, an enforced process-mask proof, or proof of where
HyBIT's Rust solver threads ran. P3-B completed diagnostically, with
`RANK_PID_CAPTURED_AFFINITY_UNVERIFIED` classifications.

### Release P3-C: 8-rank MS-MPI placement crossover

24/24 MS-MPI runs passed `PASS_PROFILED_NUMERICAL` (four observations for
each of six configurations), with the same `thermal1.mtx` input, Jacobi PCG,
P2P, 8 ranks, 8 RHS, 1 warmup and 3 repeats per launch. All runs had
MPI exit code 0 and identical iteration counts by RHS. Root true relative
residuals were in `[4.97445e-12, 4.99651e-12]`, and original solution
errors in `[4.83478e-9, 4.83898e-9]`.

| Partition | MS-MPI layout | Median profiled batch (ms) | Ratio vs same-policy native default | Median setup + one batch (ms) | Median full instrumented launch (ms) |
|---|---|---:|---:|---:|---:|
| equal | native-default | 2145.87 | 1.000 | 2275.04 | 8812.75 |
| equal | `seq:P:L` | 1979.40 | 1.084 | 2107.64 | 8151.84 |
| equal | `spr:P:L` | 1647.20 | 1.303 | 1775.83 | 7134.66 |
| multilevel | native-default | 1908.48 | 1.000 | 2242.29 | 7977.11 |
| multilevel | `seq:P:L` | 1417.41 | 1.346 | 1736.38 | 6333.09 |
| multilevel | `spr:P:L` | 1441.82 | 1.324 | 1751.84 | 6146.75 |

`equal`/`seq:P:L` had substantial batch variance (1602.80–2638.13 ms,
22.7% sample CV). `equal`/`spr:P:L` was much steadier (1623.07–1682.80 ms,
1.8% sample CV). Under `multilevel`, the 1.7% median batch difference between
`seq:P:L` and `spr:P:L` is not a robust win given four samples per option.
`spr:P:L` was the lowest median full-launch time for the multilevel policy.

**Scope limits:** P3-C compares instrumented prepared-batch timing and full
launch timing within a controlled, single-node, pinned-code experiment.
Individual HyBIT solver thread CPU residency was not captured; neither hard
physical-core binding, actual communication/computation overlap, a portable
best layout nor production speedup was established. These results support
`spr:P:L` as an opt-in candidate for benchmarking this 8C/16T host, not a
portable default policy. Do not extrapolate the P3-A rank-scaling ratios to
P3-C layouts without matched baseline measurements.

### Evidence files (external to the repository)

The files were verified by SHA-256 before this handoff was appended. They
remain outside Git in `D:\Work` and are not part of the crates.io package.

| File | SHA-256 |
|---|---|
| `hybit_0_9_release_p1_20261026_summary.csv` | `7faa4f88045b39459de88759c07866e9a61a9d1680ac9b7df2c5f3c6d13b2d60` |
| `hybit_0_9_release_p2_matrix_20261026_summary.csv` | `710b6720f22828b2be5a4751539bb037160051e9f6c547a3ce5017808ef3ca33` |
| `hybit_0_9_release_p3a_thermal1_20261026_scaling_comparison.csv` | `b274fb8fe0ced2617104408a2ed15dcada3e929f4b2f14235c239ab91ffd1192` |
| `hybit_0_9_release_p3b_affinity_20261026_summary.csv` | `e328f5f903736298d6fa614e1b516553b107014401e92178c00e187afb7f79ab` |
| `hybit_0_9_release_p3b_affinity_20261026_rank_samples.csv` | `6adea5002bb1239beb81a5af6768e943c6c7fdb58460c3b134714419d6e1d879` |
| `hybit_0_9_release_p3c_affinity_20261026_comparison.csv` | `d2bd131245123b9f27f852fcf0ad0a4d158a26de42eb9a739736c90edf610b2c` |
| `hybit_0_9_release_p3c_affinity_20261026_raw.csv` | `6ecb02057fd48b4f12d641c9000677846c693176e93eb50dca3f7e521b2fb001` |

### Remaining 0.9.0 release gates

P4: README, CHANGELOG, documented support boundaries, feature/API surface,
mdBook build and navigation; Cargo package metadata and file inclusion;
license and FFI checks; local package/dry-run validation; then another
source-integrity gate. GitHub tagging and any crates.io publication must
remain separate explicit release actions. No 0.9.0 release tag was made by
P1–P3-C. This checkpoint records test evidence only.

## 2026-10-26 P4-A publication readiness audit and P4-A-FIX1 documentation/CI candidate

- Canonical audited branch/HEAD: `develop/0.9.0` at
  `a59387b8841d2a45745e124e878b13dfc429b898`, clean source tree.
  P1–P3C numerical/affinity evidence remains unchanged by P4-A-FIX1.
- P4-A source integrity: **374 manifest entries / 372 SHA-256 files PASS**.
  Cargo metadata and seven package file lists PASS, mdBook 0.5.4 PASS,
  and `hybit-core` `cargo publish --dry-run` PASS without upload.
- P4-A summary: **30 PASS / 4 BLOCKER / 7 WARN / 2 PENDING**.
  Exactly four blockers: no 0.9.0 release notes; no 0.9.0 CHANGELOG;
  0.9.0 publication instructions absent; CI package loop excludes
  `hybit-distributed`.
- All seven warnings concern absent `LICENSE` in package file lists.
  **Do not** treat the warnings as resolved solely because Cargo metadata
  declares MIT; release qualification must decide and verify the packaging
  policy separately.
- PENDING: actual Windows MSVC C/C++/Fortran ABI runtime (P4-B native
  follow-up), plus dependent-crate dry-runs after 0.9.0 registry
  prerequisites exist. Neither has passed yet.
- User-side evidence CSV: `D:\Work\hybit_0_9_release_p4a_20261026_summary.csv`;
  SHA-256 `faa39d5d563636760eceaeacf3f1fee1e5029fe00aa929b78e5de5e58b1119b4`.
- Proposed P4-A-FIX1 change scope is **documentation, CI package inventory and
  source manifests only**, with no Rust or numerical algorithm changes.
  Re-test after application and checkpoint separately before proceeding.

**Next single prioritized action:** apply/review the guarded P4-A-FIX1
documentation/CI patch on exactly the audited commit, then run integrity,
format/Clippy/test and mdBook/Cargo package gates. Do not tag/publish.

## 2026-10-28 — post-0.9.0 reliability review (Linux fallback, local changes)

- **Source:** exact published `v0.9.0` commit `0a8cd32784a4a8ba5e3fe5840b526963bb7b1eb4` on `develop/0.9.0`, checked out into local branch `review/0.9.0-true-residual`. The user's canonical `D:\Work\hybit\hybit` Windows worktree was not available; do not assume it matches this clone.
- **Implementation:** high-level prepared and structural solves independently certify final `||b-A*x||` from original CSR using a scaled finite-data norm, classifying false recursive convergence as `Breakdown`; low-level serial/parallel/resident PCG rejects `Inf <= Inf` zero-iteration convergence and non-finite update scalars. C ABI return codes now distinguish numerical failure (3) from exhausted iterations (4); signatures, struct layouts and defaults preserved. Linux CI checks tracked manifest and complete SHA-256 inventory. Entry READMEs and post-release notes corrected for the 0.9.0 release. See `docs/REVIEW_2026-10-28.md`.
- **Verified here:** Rust 1.99.1 Linux x86_64; pre-change and after-change `cargo test --workspace --locked` PASS; after-change `cargo clippy --workspace --all-targets --locked -- -D warnings`, `cargo fmt --all -- --check`, Linux source-integrity and staged diff gates PASS. Additional targeted tests cover actual high-level `1e-200` RHS false-success prevention, simulated recurrence drift, failed-status non-promotion, FFI error classification and non-finite PCG norms across three execution modes. Standard Linux Release with thin LTO **failed at rust-lld link** (unresolved core/alloc symbols); diagnostic `CARGO_PROFILE_RELEASE_LTO=false cargo test --release --workspace --locked` PASS. The same override allowed `fem_bench` `poisson5.mtx` 5-DOF smoke to PASS (3 iterations; true residual and manufactured solution error zero). The LTO-off result does **not** qualify the unchanged Release profile.
- **Not verified here:** Windows MSVC + Microsoft MS-MPI, C/C++/Fortran ABI smoke, real-FEM time-to-solution/peak RSS, arbitrary non-SPD matrices, post-push GitHub CI or any new release. No performance speedup or universal SPD claim.
- **Diff/rollback:** local uncommitted review branch; no push, tag, registry upload or Release. Revert only this branch's scoped changes if Windows numerical/ABI gates fail. Final source hashes, tracked manifest inventory and staged diff were rechecked before handoff.
- **Next single action:** transplant/compare this patch with the owner's canonical Windows worktree at its actual HEAD and run the native Windows release-test/ABI/MS-MPI correctness gates prior to any merge.

## 2026-10-28 — 0.10.0 first development checkpoint (supersedes prior next action)

- **Branch/base:** `develop/0.10.0` exists on GitHub at published
  `v0.9.0` commit `0a8cd32784a4a8ba5e3fe5840b526963bb7b1eb4`.
  The user fetched/tracked it in WSL2 at `/home/michioga/hybit`.
  The assistant's separate Linux working copy uses the same branch/base,
  with **uncommitted staged post-0.9.0 reliability changes** plus this
  checkpoint's new version/documentation edits. They are **not yet on GitHub**
  or in the user's WSL2 checkout.
- **User WSL2 baseline (before 0.10 edits):** Ubuntu 26.04.1, Rust/Cargo
  1.99.1, GCC/gfortran 15.2, Open MPI 5.0.10/PRRTE 3.0.13, WSL 3.0.3 and
  Linux kernel 6.18.54.1. `perf` user-mode cycles/instructions available.
  Debug + standard thin-LTO Release workspace tests **236/236 each PASS**,
  strict Clippy and rustfmt PASS, `hybit-distributed --features mpi`
  **100/100 PASS**, native Release owned-CSR/PCG across 1/2/4/8 MPI
  ranks **80/80 PASS** on manufactured 4k/16k-DOF SPD cases.
  This is not post-patch validation or proof of multi-node speedup.
- **Implementation in assistant copy:** carry forward the 16-file
  numerical-hardening patch; align workspace + eight crate versions and
  internal path-dependency requirements to `0.10.0` in Cargo manifests/
  lockfile; add `develop/0.10.0` to GitHub Actions push triggers;
  document the independent review's critical performance/autonomy concerns
  in `docs/DEVELOPMENT_PLAN_0.10.0.md`; correct development-status/roadmap
  headings while retaining published 0.9.0 history. No automatic solver/
  MPI/ABTM policy or public API signature change.
- **Correctness hypothesis:** original-CSR final certification and finite
  PCG guards make false success observable without changing valid SPD
  results or ABI layout. **Performance hypothesis to test next:** with
  fixed local factor memory, some verified SPD FEM problems benefit from
  256/512 vs 128 local DOFs while others lose. Require matched residual,
  manufactured error, cold/setup/reuse time and peak RSS, plus PETSc/hypre
  AMG comparison where available. See the new plan. No such speedup has
  been established yet.
- **Assistant-side gates after version edit:** Rust 1.99.1 offline
  `cargo metadata --locked`, default-feature `cargo test --workspace
  --locked`, strict workspace Clippy, rustfmt and diagnostic LTO-off Release
  tests **PASS**. Standard thin-LTO Release previously failed only in this
  isolated Linux runner; the user's WSL2 standard Release passed on the
  **pre-patch** baseline. WSL post-patch native MPI/FFI and real-FEM gates
  remain pending.
- **Rollback:** preserve the `v0.9.0` base, user's WSL state and the
  saved `HyBIT_0.10.0_post_release_hardening.patch` in ChatGPT Library.
  Do not discard user changes, commit, push or tag without approval.
- **Next one action:** integrate this candidate with the owner's clean
  WSL2 `develop/0.10.0` and rerun native Linux release + MPI +
  source-integrity/ABI gates before the first authorized checkpoint;
  then build the real-FEM repeated-trial runner.
