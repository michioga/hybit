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
