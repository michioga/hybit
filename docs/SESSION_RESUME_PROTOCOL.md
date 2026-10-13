# HyBIT セッション再開プロトコル / Session Resumption Protocol

**2026-10-13** — 会話履歴やAIメモリの代わりに、**リポジトリ内の設計契約・実測・直近handoff**を継承情報の正本とする。

## 新セッションの最初の行動（必須）

1. 読む: `AGENTS.md`, `docs/PROJECT_CHARTER.md`, `docs/HYBIT_LONG_TERM_ARCHITECTURE.md`, `docs/HYBIT_VALIDATION_BASELINE_2026-10-13.md`, `docs/SESSION_HANDOFF.md`, `docs/ROADMAP.md`。各文書の**日付と「検証済/提案」表記**を確認する。
2. ユーザーの `D:\Work\hybit\hybit` **ローカルrepoを第一の事実源**とする。`git branch --show-current`, `git log -5 --oneline`, `git status --short`, `git diff --stat`, `cargo --version` を確認。アクセスできない場合、ユーザーが提供した最新ログやZIPを利用し、古いGitHubリモートだけからローカル状態を断定しない。
3. 問題を切り分ける: (A) 設計方針、(B) 数値アルゴリズム、(C) MPI/LLVMビルド環境、(D) ベンチマーク、(E) Release。無関係な大規模パッチや環境診断の再開を避ける。
4. 目的・数値ゲート・計測スコープ・変更予定ファイル・rollback方法を明記し、既存実装（PCG/FGMRES、局所分解、ABTM、MPI Halo/Rayon）を先に確認する。
5. 段階実装・局所正確性テスト・ソース整合性・回帰テスト・実機測定の順に進める。未実行のテストはPASSと報告しない。
6. 検証を終えたら `docs/SESSION_HANDOFF.md` と実測baseline（必要なら新日付ファイル）、design decisionを更新する。commit/pushはユーザーの明示的承認・判断を尊重する。正式Release/crates.ioは別承認。

## 絶対に忘れてはならない要件

- 収束良好部分はKrylov反復、難領域のみ局所直接法で補強。PCGのSPD契約・前処理変更の数学的妥当性を守る。
- 公開APIはシンプル。解法・前処理・ABTM/CSR・MPI/Rayonは原則内部選択。重要な数学的契約は適切に公開し、判定不能を虚偽の自動認定にしない。
- ABTMはメタデータ論理演算による不要演算・データ移動の除外を中心とする。数値SpMVは測定で勝つprepared backendを選ぶ。
- Rayonはノード内共有メモリ、MPIはランク間haloとcollectiveの通信量・回数を減らす。ghost重複は現行実装で除去済み。
- 熱/ビーム/シェル/ソリッドの可変DOF、混合field、回転、温度、乗数、MPC。摩擦付き接触のopen/stick/slip、非対称/不定系、非線形outer+linear inner、動的contact haloを長期目標として設計する。
- ソルバはFEM側の要素実装・接触探索を自動代替しない。現状で未実装の機能を完成済みと表現しない。

## 開発ワークフロー

- Source Integrity: `MANIFEST.txt` に追跡対象（ドキュメントを含む）、`SOURCE_SHA256.txt` にマニフェストとSHAファイル自身を除くハッシュ。local treeの変更に応じて更新し、source integrity gateを実行する。
- Windows: `x86_64-pc-windows-msvc`, Microsoft MPI, `CLANG_PATH=C:\Program Files\LLVM\bin\clang.exe` / `LIBCLANG_PATH=C:\Program Files\LLVM\bin`。MinGW/OpenMPIへの安易な切替え禁止。
- テスト: `cargo fmt --all -- --check`, 対象crateの `cargo test --release`, 必要時 `cargo clippy`, MS-MPI実行、true residual、行列クラスに適切な検証、メモリ・elapsed。Release gatesは別。
- 実験では `setup`, `symbolic`, `numeric`, `halo`, `wait`, `reductions`, `iteration`, `end-to-end`, `peak memory` を可能な範囲で分ける。meanだけでなくmedian/分散、rank maximum、repeat countを記録。
- GitHubリモートがこの資料より古い可能性に注意。`SESSION_HANDOFF.md` の古い問題を再診断する前に、validation baseline と現行ユーザー結果を照合する。

## セッション終了時の更新テンプレート

```text
日付 (JST):
Branch / HEAD / last pushed commit:
Working tree status:
対象の設計契約 / 変更したファイル:
現状: implemented / prototype / proposal:
環境 (OS, target, compiler, MPI, threads):
変更・仮説・検証ゲート:
PASS / FAIL (未実施は明記):
ベンチマーク (行列, ranks, warmups, repeats, 数値残差, bytes, peak RSS, timing scope):
判断 (採用 / 非採用 / 継続実験) と理由:
残る障害・制約・rollback:
**次の一手（1つだけ）**:
```

## 新セッションへ貼る短い再開文

> HyBIT の作業を継続してください。リポジトリの `AGENTS.md`、`docs/PROJECT_CHARTER.md`、`docs/HYBIT_LONG_TERM_ARCHITECTURE.md`、`docs/SESSION_RESUME_PROTOCOL.md`、`docs/HYBIT_VALIDATION_BASELINE_2026-10-13.md`、`docs/SESSION_HANDOFF.md` を先に読み、設計の不変条件と実測に忠実に進めてください。ローカル作業ツリー `D:\Work\hybit\hybit` が最新の基準です。既存の検証済み実装を再利用し、事前に仮説・数値gate・変更ファイル・rollbackを整理してから次の1段階を提案してください。
