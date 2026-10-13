# HyBIT 0.9 — 2026-10-13 検証基準 / Evidence Snapshot

**性質:** ユーザーから提供されたWindows PowerShell実行ログに基づく実測の記録。**ソースまたはこのファイルの存在だけで再現テスト済みとはみなさない**。その後のローカルログが優先。行列のSPD性をこの資料のみで認定しない。

## 環境・再現条件

- ローカルrepo: `D:\Work\hybit\hybit`、ユーザー実行時のブランチ `develop/0.9.0`。最終C5実行後 `git status --short` は**空（clean）**。HEAD commitの最新値は未確認。
- OS: Windows 11 / Rust target `x86_64-pc-windows-msvc` / Microsoft MPI + MS-MPI SDK / LLVM 23.1.3 (`C:\Program Files\LLVM\bin`)。
- `mpi-sys 0.2.4` / `mpi 0.8.0` / `bindgen 0.72.1`。`CLANG_PATH`を明示すると、クリーンビルドとMPI 2/8-rank smokeがPASS。正確な低レベル原因を断定しない。
- 安定したPowerShell環境変数:

```powershell
$env:CLANG_PATH = 'C:\Program Files\LLVM\bin\clang.exe'
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
$env:MSMPI_INC = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Include\'
$env:MSMPI_LIB64 = 'C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\'
```

## C2 — MPI owned CSR / PCG (8 ranks)

- `mpi_owned_pcg.exe` 実機8ランク、Chain/Banded/Heterogeneous/Poisson2d/Disconnected × 4096/16384 DOF × RHS 0/1 = **20/20 PASS**、exit 0。
- 数値真相対残差（実験表示値）の最大 `9.889e-10`、製造解との誤差最大 `4.540e-8`。**これはテスト行列のゲートであり、任意のFEM行列の収束保証ではない**。
- 出力例 `C:\Users\michi\AppData\Local\Temp\hybit-owned-pcg8-20261013-194719.csv`（ユーザー環境内、可用性は要確認）。

## B3 — MPI Halo / 非同期CSR SpMV (4 ranks)

- `mpi_overlap_spmv.exe` は現行例では1/2/4ランクをサポート（8-rank対象ではない）。4ケース全て `error=0.000e0`, exit 0。

| case | n | Blocking SpMV ms | overlap SpMV ms | recv ghosts | 備考 |
|---|---:|---:|---:|---:|---|
| chain | 131 | 0.015935 | 0.003315 | 6 | 短時間の局所測定 |
| banded | 128 | 0.004400 | 0.001580 | 18 | 同上 |
| directed | 120 | 0.001755 | 0.001030 | 3 | 同上 |
| disconnected | 128 | 0.000170 | 0.000400 | 0 | 通信不要ならoverlap overhead |

マイクロ秒の短時間テストであり、実FEMの高速化倍率を意味しない。

## C5 — MPI-owned `nd3k.mtx` CSR vs ABTM numerical SpMV

- G8-C12d-6C5 probe: source integrity gate **253 manifest / 251 SHA verified PASS**。Preflight PASS → MS-MPI実機1/2/4/8ランクのnumerical gate全PASS。Temporary probe exampleは撤去。`git status --short` は空。
- 問題: `nd3k.mtx`, n=9000, nnz=3,279,690。測定対象は**同一blocking Halo後のローカル数値カーネル**。非同期ABTMや全PCGの性能比較ではない。

| ranks | CSR kernel ms | ABTM kernel ms | CSR/ABTM | rel_error | total ghosts | CSR MB | ABTM MB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.848130 | 2.825060 | 0.6542 | 3.649e-14 | 0 | 39.392284 | 31.338476 |
| 2 | 1.063190 | 1.610220 | 0.6603 | 3.649e-14 | 8778 | 39.392288 | 31.325696 |
| 4 | 0.565540 | 0.834450 | 0.6777 | 3.353e-14 | 24147 | 39.392296 | 31.337800 |
| 8 | 0.301870 | 0.523540 | 0.5766 | 8.548e-14 | 44213 | 39.392312 | 31.390616 |

- **結論:** 数値一致（CSRとの比較）は約1e-14レベル。ABTM格納は8-rank合計で約20.3%削減、しかし現行のABTM数値SpMVはCSRより遅い。ABTMの「一度だけ構築」という前提はこの計測でも成立（カーネル時間に変換時間は含まない）。通常SpMVへの無条件採用は禁止。
- 8-rank Haloペイロードの目安: 44,213 f64 = **353,704 bytes**/refresh（全ランク受信値数の総和、通信プロトコル/メタデータ/一時copyは除外）。重複ghostは既にrank単位で排除されている。

### C5 additional CSV fields

| ranks | halo_setup_max_ms | abtm_build_max_ms | blocking_halo_max_ms | native_csr_overlap_max_ms | native_halo_wait_max_ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.707400 | 107.901500 | 0.001590 | 2.021860 | 0.000040 |
| 2 | 21.837500 | 52.579300 | 0.222940 | 1.448860 | 0.496250 |
| 4 | 20.407500 | 26.328900 | 0.276310 | 0.879530 | 0.458430 |
| 8 | 16.446100 | 19.731600 | 0.298420 | 0.587040 | 0.334140 |

- `halo_setup_max_ms`: 実MPI入力でrank-local CSR＋Haloを準備した時間。`abtm_build_max_ms`: CSRからABTM構築の追加時間。先の `TOPOLOGY_GATE` の8192行*合成replicated topology*の計画時間とは比較範囲が異なる。
- `blocking_halo_max_ms` は通信のみ、`native_csr_overlap_max_ms` は通信・CSR SpMVを含む全体、`native_halo_wait_max_ms` はwait。**異なるランク最大値同士を足したり、内訳比率として断定しない**。
- CSV: `D:\Work\hybit-mpi-baselines\g8c12d6c5-validation\nd3k_mpi_abtm_spmv.csv`。LOG同ディレクトリの `.log`。

## 補助的なC4A結果と不変の注意

- `[u64;3]` prepacked mask のXOR+POPCNTは `[u32;8]` 比で当該microbench **約1.64×速い**。毎回repackingを含めると **約92.8×遅い**。マスク単体の性能をABTM SpMVやend-to-end MPIの改善と混同しない。
- ビルド/数値gate済みと、productionで自動選択されることは別。これらのログは**設計判断の基準値**であり、採用済み性能改善の主張ではない。

## 次回のチェックポイント

1. ローカルrepoの最新HEAD / working treeを確認し、記録を更新する。2026-10-12の古い `docs/SESSION_HANDOFF.md` の「C5数値未実行」「bindgenが未解決」を最新状態として扱わない。
2. 新しい長期設計の要件を `docs/HYBIT_LONG_TERM_ARCHITECTURE.md` から読み取る。
3. 次は**混合DOF/field-aware構造のContractと正しさの小実験**を優先。計測上はABTM構造演算の便益と、Halo plan + owner partitionによる実通信削減をCSR基準と比較する。
