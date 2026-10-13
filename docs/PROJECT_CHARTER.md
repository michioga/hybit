# HyBIT project charter / 開発憲章

2026-10-12 — ユーザーが明示した長期目標の記録。これは設計上の優先順位を定める文書であり、各機能が実装済みであることを保証するものではない。

## 1. 製品としての最終目的

**HyBIT — Autonomous Hybrid Sparse Solver** は、FEMやHPCで用いる大規模疎線形方程式を対象に、**利用者にはできるだけ簡潔なAPI**を提供し、ソルバ内部では数値的頑健性、メモリ効率、MPI/Rayonによる絶対性能を自律的に最適化する。

利用者の基本動作は原則として「適切な行列と右辺を与える → 解と診断結果を受け取る」。反復法／直接法、前処理、ABTM内部表現、局所領域、通信最適化の詳細は、可能な限りソルバの責務とする。既存公開API `hybit::solve()` / `HybitSolver` / `analyze → prepare → solve-many` を尊重する。

## 2. ソルバの中核方針

### 2.1 低メモリ反復法を先行

- SPDにはPCGとSPD適合の前処理、一般非対称正方行列にはFGMRESなど、**数学的に適合する方法**を選ぶ。
- 残差（再帰残差だけでなく真の `||b-Ax||`）、進捗、停滞、反復上限、breakdown、時間・メモリ予算を計測する。
- 問題クラス（SPDかどうか）を収束結果だけで証明したと誤認しない。未検証データは明示する。

### 2.2 失速時にも計算を継続できる設計

- 収束不良に応じて、既存の選択的hard-region抽出、ABTMによる近傍拡張、局所Cholesky/Schwarzなど、より強い前処理に段階的に昇格する。
- 前処理器を変えた際はKrylovの数学的契約を守り、必要に応じて正しくrestartする。既存の進捗・ワークスペース・ファクタを可能な範囲で再利用する。
- **長期目標：反復法だけでは解けない場合、問題全体を解く適切な疎直接法へフォールバックできること。** ただし現行の局所Cholesky補正を「全体系の疎直接法切替えが完成」とみなさない。
- 全体系直接法はメモリ超過、特異行列、数値ピボットの失敗などで実行不能なことがある。利用者への明確な失敗通知、容量ガード、選択可能なpolicyを伴わせる。「常に必ず解ける」と約束しない。
- 解法の自動切替えは対象行列群に対する正確性・収束・費用の実測を経て段階的にproductionへ導入する。

### 2.3 ABTMは構造演算と準備の中核技術

- 疎行列のコンパクトな構造・数値表現を追求する。Sparse/Bitmap/Denseの物理レイアウトは占有率・行列・ハードウェア特性に応じて選び、普遍的なCSR置換と決めつけない。
- AND / OR / AND-NOT / XOR / popcount / rank / select を使って、構造交差、frontier/region探索、制約マスク、隣接関係、ILUのシンボリック処理、準備計算を加速する。
- ABTMを利用してdistributed partitionと**Haloの必要要素集合、送受信相手、インデックス、近傍、通信スケジュール**を効率よく生成・再利用する。
- ベクトルの `f64` Halo送受信量がABTMだけで減るわけではない。メタデータ圧縮、通信量削減、まとめ送受信、オーバーラップはそれぞれ独立して測定する。
- G6の測定では通常scalar ABTM SpMVがCSRに劣った（既存資料 `docs/ABTM_TOPOLOGY.md` を参照）。数値カーネルは勝つことが確認できたprepared CSR等を継続使用してよい。

### 2.4 Rayon × MPI による絶対性能

- Rayon：SpMV、前処理適用、ベクトル演算、準備処理のスレッド並列化。タスクの粒度・オーバーヘッド・メモリ帯域を測る。
- MPI：既存のrank-local CSR、Halo計画、非同期Halo交換、内部/境界行の重なり、分散PCGおよびcollective通信を再利用して拡張する。
- MPIのスレッドサポート (`MPI_THREAD_SINGLE` 等) を守る。MPI呼び出しとRayonの計算部分の責任分離を検証する。
- 性能の主指標は**end-to-end time-to-solution**（setupとsolve-manyを含む）・ピークメモリ・収束/解の妥当性・スケーリング。カーネル単体の速度だけで勝利としない。

## 3. 既存の成果を再利用する原則

- Public API: `hybit::solve`, `HybitSolver`, prepared solve-many; C/C++/Fortran連携。
- `hybit-auto`: adaptive controller、収束監視、段階的前処理昇格、メモリ予算。
- `hybit-precond`: Jacobi、選択的局所直接法、ILU等。
- `hybit-matrix`: CSRとABTMのstructural/numerical実装。
- `hybit-distributed`: 分割、Halo、MPI backend、overlap、分散PCG/前処理。
- 既存の試験群、ベンチマーク、SOURCE INTEGRITY gate、差分レビューを新規実装の出発点とする。

現在のAPI/機能の正確な状態はソースを優先し、上記をproduction機能の完了宣言と混同しない。古いGitHubリモートはローカルより遅れる可能性がある。

## 4. ユーザー向けAPI設計原則

- 一般利用者向けの既定APIは少数の引数で動く。高度なpolicyは**明示的なオプション**として公開する。
- matrix class, tolerance, resource budget, solver fallback policy等の重要契約を明示し、内部アルゴリズムの詳細を押しつけない。
- 結果にはtrue residual、収束status、反復回数、適用した前処理/経路、昇格理由、時間、メモリを診断として返す。
- 失敗時は `max_iterations`, `breakdown`, `factorization_failure`, `out_of_budget`, `unsupported_matrix_class` などを意味の異なる状態として扱う（ここに記載の名称は概念例であり既存enum名とは限らない）。
- 単発solveもsolve-manyも一貫した意味論を持つ。

## 5. 評価の原則

1. **Correctness** — 真の残差、参照解との一致、MPIランク間整合、determinism/許容差、SPD未証明の明記。
2. **Robustness** — 反復法成功、停滞、breakdown、メモリ不足、切替え・適用失敗の回帰テスト。
3. **Speed** — 1/2/4/8ランク、Rayonスレッド数、行列サイズと構造の違いを含む再現可能な測定。ランク最大時間と時系列/分布を記録。
4. **Memory** — CSR/ABTMのmetadata・values、変換中のpeak、Halo buffer、preconditioner factor、Krylov workspaceを区別する。
5. **MPI** — bytes/peers、pack/post/wait/scatter、interior/boundary、Allreduce、実効オーバーラップを分解。
6. **End-to-end** — setup、実際の反復、局所/全体直接法（導入後）、solve-many償却を計測する。

## 6. ソース管理・セッション間引継ぎ

- **方針変更：作業を長く凍結せず、検証済みの小さな節目ごとにcommit/pushする。** ユーザーが所有するブランチとリモートを尊重する。
- 不完全な段階を残す必要があるならWIPであることを明示し、必須gateの不合格を成功と表示しない。
- 各作業開始時に `git status --short`, `git log -5 --oneline`, branch、既存実装と現在のログを確認。
- 毎セッション終了時（可能なら各commit時）に `docs/SESSION_HANDOFF.md` を更新。会話memoryは補助情報であり、Git管理されたソース・資料・テスト結果を第一の継承源とする。
- `MANIFEST.txt` と `SOURCE_SHA256.txt` に含まれる整合性ルールを必ず保持する。新しい追跡ファイルを加えたらmanifest・hashを更新してsource integrity gateを実行する。
- environment/debugとアルゴリズム実験を混ぜない。原因未特定のままfixスクリプトを連発しない。

## 7. GitHub Release / crates.io の公開方針

**開発中のこまめなcommit/pushと、正式なGitHub Release・crates.io公開は別の操作として扱う。** ユーザーの方針では、大きな開発計画を十分に検証できた区切りでのみ、新しい正式リリースを行う。既に公開されている既存リリースを否定せず、公開版と開発版を区別する。

- 開発ブランチ：小さく検証可能な変更をcommitし、適宜GitHubへpushしてセッション間の作業状態を共有する。必要なら未完成の状態はWIPと明記する。
- マイルストーン：例えば「MPIの再現可能ビルド」「ABTM Halo最適化」「solver fallback制御」など、仮説と合格条件を先に決め、機能・数値正確性・性能・メモリの検証が通ったことを記録する。
- GitHub Release：対象マイルストーンの完了を確認後、ユーザーの明示的な判断でバージョン・タグ・release notes・配布アーカイブを確定して公開する。通常のpushや開発途中のcheckpointで自動的にreleaseを作らない。
- crates.io：公開APIとsemver互換性、`cargo test` / `cargo clippy` / doc-tests、`cargo package` / `cargo publish --dry-run`、実際の配布manifest、LICENSE、ドキュメント、依存関係とMPI featureのビルド条件を確認したうえで、ユーザーの明示的な公開指示を受けて実施する。
- 公開は不可逆な操作が多い。AIはユーザーの事前許可なしにpush、tag、GitHub Release作成、`cargo publish`を実行しない。
- 各版の公開判定結果、再現コマンド、ベンチマーク環境、既知の制約をリポジトリに残す。未完了機能や未検証の性能改善をrelease notesで完成済みと断言しない。

## 8. 今後の優先順位

1. **再現可能なWindows x64 MSVC / MS-MPIビルド**（既存動作環境を壊さず、クリーンビルドを確認）。
2. **現状のMPI/Halo性能の計測**（既存コード・instrumentationを再利用）。
3. **ABTMを使用する構造/Halo計画と通信準備の高効率化**。数値SpMVを無条件に置き換えない。
4. **反復法→強力な前処理→必要に応じた直接法のフォールバック**について、数学的契約・メモリ予算・再開手順・ユーザーAPIを設計・検証。
5. **Rayon/MPIを通じて実アプリケーションの総処理時間を改善**し、勝った処理だけを段階的に統合。

採用・昇格の判断は実験結果に基づいて行い、「現行済み」と「長期目標」の境界を常に記録する。

## 9. 2026-10-13: binding architecture extension
<!-- HYBIT_2026_10_13_ARCHITECTURE_HANDOFF -->
The agreed long-term architecture is documented in **`docs/HYBIT_LONG_TERM_ARCHITECTURE.md`** and maintained through `docs/HYBIT_DECISION_LOG.md` and `docs/SESSION_RESUME_PROTOCOL.md`. It preserves: (1) lightweight Krylov iteration wherever practical with difficult subregions reinforced using mathematically valid local direct methods, (2) minimal stable public API with autonomous internal decisions, (3) exact metadata-first ABTM pruning and adaptive prepared numeric layouts, (4) minimized Rayon memory movement and MPI halo/collective traffic, and (5) eventual mixed thermal/beam/shell/solid DOFs, variable field blocks, nonsymmetric/indefinite systems and frictional contact with separate nonlinear control. These are requirements, **not claims of completed implementation**. The last user-supplied MPI/C5 numerical results are in `docs/HYBIT_VALIDATION_BASELINE_2026-10-13.md`.
