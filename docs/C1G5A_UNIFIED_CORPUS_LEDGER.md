# HyBIT 0.10.0 — C1-G5A 統一コーパス台帳と Shadow Control

- **決定基準**：`docs/C1G5_AUTONOMOUS_SOLVER_SELECTION.md`（コミット `0bd874e`）
- **入力数値証拠**：`C1-G4`、元の計算HEAD `37bc52dfb265123267081afc58bd2e6e7b0b6c97`。72件の研究用PCGは全件元CSR真残差 `≤1e-10`、単一スレッド、最大600反復。原データとハッシュは `benchmarks/evidence/C1G4_AMG_20261113_*` を参照。
- **実装範囲**：オフラインの整形・監査・固定Shadow Controlのみ。HyBIT本番の `Auto`、API、FFI、PCG、AMG数値コードには変更を加えない。

## 1. 台帳は方式選択の教師データである

同じ `(operator, RHS, tolerance, iteration budget, execution environment, reuse scenario)` について **数値的に妥当な候補だけ** 比較する。選択問題の失敗事例・未実施候補を黙って削除しない。参照解や全方式の実測時間は oracle の答え合わせにのみ使い、実行前選択に使わない。行列クラスは入力ホストの数学的契約から始め、対称性・正対角・単なる収束からSPDを証明したことにしない。

今回の `C1G5A_20261114_ledger.csv` は **3構成FEM × 4 RHS × 6実測のTwo-Level/Multilevel構成 = 72実測行**、および **3構成FEM × 4 RHS × 2欠測（Jacobi-PCG・局所Cholesky）= 24 `NOT_RUN` 行**、合計96行を持つ。各RHSごとに同一のハッシュが全実測候補と対応するかを検査する。元データの36組すべては `BOTH_CONVERGED` だが、これは欠測方式を含めた性能優位や救済を証明しない。

外部行列 `boneS01`・`cant`・`thermal1` は `matrix_inventory.csv` に行列サイズ・ハッシュ・出典と `SPD_UNVERIFIED` 状態で掲載する。これらに対応する物理RHSはこの時点で未確認で、測定値も挿入しない。`16_heat_block` のGeneralSquare FGMRESは別の比較契約で扱い、PCG候補と同一oracleへ混ぜない。

## 2. 比較単位と費用の境界

C1-G4の測定から使えるのは `setup_ms`、`solve_ms`、`total_wall_ms`、ピークRSS、coarse-factor bytes、scratch bytes、反復・真残差・製造前進誤差など。**`total_wall_ms` は例プログラム内部の経過時間**であり、GNU time によるプロセス起動から終了までの時間と同義ではない。このC1-G5Aでは、予算付きの最終oracleを求めず、**6件の既存実測候補内で `inprocess_total_wall_ms` が最小の方式**を `observed_best_arm` として記録する。

これはあくまで `OBSERVED_ONLY` ラベルであり、次を満たす本来の oracle ではない。

- 同一条件のJacobi、Hybridを含む全候補の実行
- 実際のメモリ制約、準備済み因子の共用、RHS本数、特徴量計算・監視・切替費の計上
- 数値クラスの独立した妥当性、解の精度要件、実運用環境での実測
- 複数試行による順序・冷却・配置の影響の評価

`relative_x_error_offline_only` は製造RHSでのみ使える評価ラベル。`point_load` では空欄であり、**空欄は誤差ゼロを意味しない**。小さい真残差から許容前進誤差を推論できない。`island_shell` の Multilevel で前進誤差約2%が観測されたことは、選択器が要求精度と精度リスクを明示的に扱うべき理由である。

## 3. 行列ファミリー分割・汎化の制限

3種の `island_shell`、`multi_patch`、`rotated_anisotropy` はすべて **`constructed_c1g_n144` という1つのファミリー** に属する。同一ファミリーの問題・RHS・粗格子条件を train/test に分けても、未知ファミリーへの汎化は検証されない。外部行列3件は数値候補が欠測のため、現状では held-out 性能検証を行えない。

`C1G5A_20261114_summary.json` の `held_out_family_gate` は必ず `BLOCKED_ONLY_ONE_MEASURED_FAMILY` とする。**この状態では閾値学習・ルールの本番昇格禁止。** 外部・独立生成・現実のSPD問題と各種RHS、複数の数値クラスに対応した公平な対照測定を補う。

## 4. Shadow Control は学習済み選択器ではない

`C1G5A_20261114_shadow_audit.csv` に、初期の**固定対照**を載せる。入力として認めるのは `matrix_contract, nrows, nnz, requested_tol, max_iterations` のみ。測定されたSPD候補コーパスでは `two_level:graph_piecewise` を常に記録し、未認証の外部行列に対しては `NO_RECOMMENDATION` とする。これは**性能予測ではない**。本番ルーティングにも接続しない。

今回の実測データの限定比較では、後知恵の `observed_best_arm` は12群中11群が Multilevel、1群が Two-Level だった。固定対照の observed-best に対する内部総時間比の中央値は約 **1.48**。この数値は同一構成FEMファミリー内の単回測定に限り、汎化や本番での最適選択を主張するものではない。

`shadow_regret_ratio` は、既存6実測候補内で、固定対照の実測内部総時間を事後最小時間で割った記述統計。すなわち「あとで知っている最善とのずれ」を示すためだけの監査であり、実際に未知の問題で選べたことを意味しない。単一設定の時間比較、RHS分類、行列ID、ハッシュ、真の反復数・時間、参照解誤差、事後最適ラベルを **Shadowへの推論入力に渡さない**。

## 5. 次段階の受入条件

1. **C1-G5A**：入力72件の完全性・同一RHS・元CSR数値ゲート由来・欠測24件、ファミリーの単位、正しい費用範囲、台帳再生成の一致、リーク防止の単体テストを通す。
2. **C1-G5B**：Jacobi/Hybridを同一条件で追加し、複数ファミリー・複数RHS・準備再利用条件・メモリ予算を揃えて初めて完全なoracleを計算する。実測せずに埋めない。
3. **C1-G5C**：学習候補の特徴量・費用モデルを事前凍結し、Shadow選択の理由とregretを計算する。学習と検証のファミリーを分ける。
4. **C1-G5D〜F**：外部実FEMによるheld-out資格、最終真残差、メモリ、総費用、収束救済・退行をゲートし、opt-inを経た後だけ本番`Auto`へ昇格。

## 再生成と検証

```bash
cd ~/hybit
python3 -B benchmarks/scripts/C1G5A_corpus_ledger.py --repo "$PWD" --check
python3 -B -m unittest -v benchmarks.scripts.test_C1G5A_corpus_ledger
sha256sum --check --strict --quiet SOURCE_SHA256.txt
```

入力はリポジトリにコミット済みの `C1G4_AMG_20261113_comparison.csv` と `C1G4_AMG_20261113_summary.json`。別ファイルの原ログSHA-256リストと、オーナーの原ログはC1-G4の証拠として保持する。新しい測定や既存ログの置換を、この取り込み処理は行わない。
