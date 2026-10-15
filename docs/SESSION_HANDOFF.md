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
