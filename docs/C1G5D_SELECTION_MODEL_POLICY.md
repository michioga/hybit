# HyBIT 0.10.0 — C1-G5D: 内蔵標準モデルと任意外部モデル（設計決定）

**決定日：2026-11-14。位置付け：C1-G5C のライブラリ内 Shadow Selector を拡張する研究実装。**
基準は `develop/0.10.0` の `e6e96a7` と、その上に**適用済み・未コミット**の C1-G5C library-shadow。C1-G5C の元変更を失わない。

## 1. 確定した原則

**標準モデルはHyBIT Rustライブラリ内蔵、外部モデルは任意、学習用の全コーパスはライブラリ外で管理。**
アルゴリズム選択・安全制御・モデルのスコア計算は `hybit-auto` 内で完結する。Python・シェル・CSV/JSON・学習用Matrix Marketファイルは本番 `hybit::solve` から呼び出さない。将来の認証済みコストモデルをRustへコンパイル時に同梱できる設計とする。

- **標準モデル**：`SelectionModel::embedded_standard()` / `SelectionModel::default()`。現在は *未学習・未認証* で、積極的な候補選択をしない。別途凍結したC1-G5Cの `ShadowEvidenceQualification`（実測ファミリー1件、Hybrid退行2件、救済0件）を変更しない。認証のない「標準モデル」を勝手に成功モデルと呼ばない。
- **外部モデル**：利用側が `SelectionModel::load_from_path(path)` を**明示的に1回**呼び、得られたモデルを `&SelectionModel` として再利用可能。ファイルが存在しなくても、パスを渡さなくても、HyBITの既定解法に影響しない。外部ファイルの自動検索、起動時自動ロード、環境変数経由の暗黙設定は実装しない。
- **学習用全コーパス**：`benchmarks/evidence` および所有者の原ログなどを独立して保管。外部へのモデル出力はオフラインで実施し、**最適化・安全性の教師ラベルを実行時特徴量として流用しない**。

C1-G5C の feature gate `c1g5c-shadow` を共用し、今回も既定 Cargo features / `hybit::solve` / `HybitSolver::solve_csr32` / 公開 C ABI を変更しない。外部モデルで標準モデルを上書きするグローバル状態も設けない。

## 2. Rust API と適用境界

```rust
use hybit::{Csr32Matrix, HybitSolver, SelectionModel};

let model = SelectionModel::embedded_standard(); // 外部ファイル不要
// 明示的な研究用モデルの場合のみ:
// let model = SelectionModel::load_from_path("hybit-selection-v1.bin")?;

let solver = HybitSolver::new();
let a = Csr32Matrix::new(2, 2, vec![0,2,4], vec![0,1,0,1], vec![3.0,-1.0,-1.0,3.0])?;
let b = [1.0, 1.0];
let mut x = [0.0; 2];
let (report, shadow) = solver.solve_csr32_with_model_shadow(&a, &b, &mut x, 1, &model)?;
assert!(shadow.base.recommended.is_none());
let _ = (report, shadow.model_origin, shadow.diagnostic_candidate);
# Ok::<(), Box<dyn std::error::Error>>(())
```

外部モデルのスコアは `ModelShadowObservation::diagnostic_candidate` と `diagnostic_score` にのみ記録される。**`base.recommended` は常にC1-G5Cの fail-closed 判定（現状は `None`）のまま**であり、実際に解く前処理や反復法は変わらない。数値解は既存 `solve_csr32` と一致しなければならない。候補の優劣を教える「診断スコア」は予測した実時間（ms）ではない。

## 3. 外部モデルのバイナリ V1 契約

機械依存・外部依存のない固定長180バイト、little-endian。

| オフセット | 内容 |
| --- | --- |
| `0..8` | 固定magic `HYBITSM1` |
| `8..12` | `u32` schema version = `1` |
| `12..172` | 4候補 × 5個の `f64` 係数（40個ではなく20個、160バイト） |
| `172..180` | 前172バイトの FNV-1a 64-bit 値、`u64` |

候補の固定順序は `JacobiPcg`, `HybridLocalCholesky`, `TwoLevel`, `NativeMultilevel`。各候補の5係数は `bias`, `log_nnz`, `diagonal_spread`, `rhs_peak_fraction`, `inverse_rhs_reuse`。各係数は有限・非負・`<= 1e6`、各候補の研究スコアは以下で評価：

`bias + log_nnz*ln(1+NNZ) + diagonal_spread*max(0, log10(max_diag/min_diag)) + rhs_peak_fraction*(max|b_i|/||b||) + inverse_rhs_reuse/N_RHS`。

これは**任意の研究用線形ランキング関数の形式**であり、現時点で妥当な係数がコーパスから確定したという意味ではない。スコアの小さい候補を診断候補として記録するが、学習・汎化未認証である以上、最速・収束可能・安全とは推定しない。各候補が実際に実装済みかどうかにも関与せず、本番ルーティングは行わない。比較不能なG4/G5Bの実測時間を係数に埋め込まない。

読み込み時はサイズ・magic・バージョン・チェックサム・数値範囲を拒否型で検証する。読込みサイズを181バイト以下に制限する。**FNVは偶然の破損検出用であり、真正性・耐改ざん性の保証ではない。** 署名済みの信頼済みモデルを必須にするのは本番昇格の別ゲートとする。

## 4. 安全性と制約

1. 実行時に利用できるCSR/RHS特徴量だけを使う。行列ID、学習ラベル、参照解、実測完了時間を渡さない。
2. 外部モデル内の任意のバイト列や「合格」との自己申告に、SPD証明・収束保証・メモリ制限の解除権限を与えない。V1には昇格・認証フラグ自体を含めない。
3. 数値クラス `GeneralSquare`、ゼロRHS、正の対角が揃わない場合は診断ランキング自体を出さない。対角が正でもSPDと自動認証しない。
4. 同一初期解・RHSで、Shadow有効/無効および外部モデル有/無の**解ベクトル完全一致**を回帰テストで守る。
5. Multilevel は現時点で研究候補。本番HyBIT Autoに昇格していない。
6. 外部モデル読み込みの費用は明示ロードに発生する。将来実際に利用する場合は、モデルの準備、特徴量、選択、前処理準備、solve、認証、切替をすべて end-to-end 費用に計上する。

## 5. 次の昇格条件

ファミリー単位のheld-out検証、追加の物理的RHS、Jacobi/Hybridを含む共通ハーネスでの公平な四方式比較、メモリ上限、数値的退行（今回2件）のガード、再利用数別の費用モデルが必要。**外部モデルのロード機能ができたことと、自律解法選択が完成したことは異なる。**

元のC1-G5Cは適用済み未コミットなので、C1-G5Dはその差分を含めた一体の未コミットチェックポイントとして検証する。ownerの許可までcommit/pushは行わない。
