# HyBIT

> **開発状況:** 0.6.0 は現在の開発系列です。crates.io の最新公開版は 0.5.0 です。

**HyBIT — Autonomous Hybrid Sparse Solver** は、FEM/HPCで現れる大規模疎行列を対象としたRust-firstの線形ソルバーフレームワークです。

低コストな反復法から開始し、収束状況を観測し、進捗が悪い場合には数値的に難しい自由度を抽出して、限定された局所領域だけをCholesky直接法へ昇格させます。ABTMのbitmap topology metadataは、選択領域の近傍展開や構造処理に内部利用します。利用側は通常のCSR32行列を渡すだけです。

> **現在の位置づけ:** HyBIT 0.6.0（開発版） はpre-1.0の実験的リリースです。自動ソルバー経路は現在、実数の対称正定値（SPD）行列とPCGに限定されています。1.0までAPIが変更される可能性があります。

## 主な機能

- Rustを中核とし、C ABI経由でC/C++/Fortranから利用可能
- 公開入力形式はCSR32、ABTMは内部backend/topology engineとして利用
- 再利用可能なworkspaceを持つPCG
- 収束進捗の自動probeとselective local direct escalation
- 複数hard regionとweighted overlapping Schwarz correction
- SPD principal submatrixに対するbounded local dense Cholesky
- `analyze -> prepare -> solve-many`
- 学習済みlocal factorとKrylov workspaceの再利用
- 収束、時間、region、factor memory、reuseを返す`SolveReport`
- MIT License

## crates.ioから利用

```bash
cargo add hybit
```

または、`Cargo.toml`へ以下を追加します。

```toml
[dependencies]
hybit = "0.5.0"
```

Rust 1.73以降を対象とします。

## Rustでの最小例

```rust
use hybit::{Csr32Matrix, HybitSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Csr32Matrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
    )?;

    let b = vec![1.0, 0.0, 1.0];
    let mut x = vec![0.0; 3];

    let solver = HybitSolver::new();
    let report = solver.solve_csr32(&a, &b, &mut x)?;

    println!("x = {x:?}");
    println!("iterations = {}", report.iterations);
    println!("relative residual = {:.3e}", report.relative_residual);
    Ok(())
}
```

単発solveなら、より簡単に以下でも利用できます。

```rust
let (x, report) = hybit::solve(&a, &b)?;
```

## Analyze / Prepare / Solve Many

同じ行列に対して複数RHSを解く場合はprepared contextを使います。

```rust
let solver = HybitSolver::new();
let analysis = solver.analyze_csr32(&a)?;
let mut prepared = solver.prepare_csr32(&a, &analysis)?;

let mut x1 = vec![0.0; a.nrows()];
let r1 = prepared.solve(&a, &b1, &mut x1)?;

let mut x2 = vec![0.0; a.nrows()];
let r2 = prepared.solve(&a, &b2, &mut x2)?;
```

最初の難しいRHSでlocal regionとCholesky factorが構築された場合、2本目以降では同じ行列に対してfactorとPCG workspaceを再利用します。

## 現在の処理フロー

```text
CSR32
  |
  v
analyze
  |-- matrix profile
  |-- SPD baseline checks
  |-- backend policy
  |-- structure/value signature
  v
prepare
  |-- Jacobi
  |-- PCG workspace
  |-- optional ABTM
  v
solve #1
  |-- Jacobi-PCG probe
  |      |
  |      +-- progress良好 -> PCG継続
  |      |
  |      +-- progress不良
  |             |-- residual/risk mask
  |             |-- hard region分離
  |             |-- ABTM halo展開
  |             |-- local Cholesky
  |             +-- Hybrid PCG restart
  v
local factor cache
  v
solve #2..N
  |-- factor再利用
  |-- workspace再利用
  +-- probe/factorization省略
```

PCG反復の途中で前処理器を変更せず、前処理器を強化するときはKrylov系列をrestartする設計です。

## 0.5.0公開版を基準にした0.6開発

0.5.0はRust/C ABI/C/C++/Fortranのrelease gateとcrates.io外部smoke testを通過した公開基準版です。0.6.0では、この数値基盤を維持したまま実FEM行列を評価するMatrix Market入出力とベンチマーク経路を追加します。

synthetic SPD regressionでは、単一hard regionでJacobi-PCG 33反復に対してHyBIT Auto 13反復、2つのhard regionでは25反復に対して13反復でした。prepared solve-manyでは、最初の難しいRHSで13反復、2本目ではlocal factorを再利用して1反復となり、2回目のlocal factorizationは発生しませんでした。

これらは制御フローと数値挙動を確認するための**小規模synthetic regression**です。実FEM問題全般で同等の高速化を保証するものではありません。絶対時間の比較も、現段階では性能主張には使用しません。

## 現在の制約

現在は`f64`、square SPD、PCGが中心です。local directはbounded dense Choleskyで、既定では最大128 DOFのregionを最大8個まで使用します。prepared factor reuseは行列構造と係数bit列が完全に同一の場合に限ります。MINRES/GMRES/BiCGStab、coarse correction、MPI、GPU、out-of-coreはまだ未実装です。prepared contextは現段階ではsingle-threaded利用を想定しています。

したがって、現時点のHyBITは実験的な数値ソフトウェアです。工学的判断へ利用する場合は、残差だけでなく物理量・参照解・独立ソルバー等による検証を行ってください。

## 実FEM / Matrix Marketベンチマーク

0.6では、拘束条件適用後のSPD剛性行列をMatrix Market (`.mtx`) から読み込み、同じ初期値・許容誤差・最大反復数でplain Jacobi-PCGとHyBIT Autoを比較できます。

```powershell
.\bench-fem.ps1 D:\path\to\K.mtx
```

または直接、

```powershell
cargo run --release -p hybit --example fem_bench -- --matrix D:\path\to\K.mtx
```

`--rhs`を省略すると `x_exact = 1` として `b=A*x_exact` を生成します。両solverについて `||Ax-b||/||b||` を独立再計算し、反復数、wall-clock、hard DOF、region数、local factor memoryも出力します。詳細は [benchmarks/README.md](benchmarks/README.md) を参照してください。

## C/C++/Fortran

GitHubリポジトリにはC ABI、C++ wrapper、Fortran `ISO_C_BINDING` moduleを含みます。

```powershell
.\build.ps1
.\build-examples.ps1

.\build\hybit_c.exe
.\build\hybit_cpp.exe
.\build\hybit_fortran.exe
```

WindowsではRust/MSVCで`hybit.dll`を作成し、MinGW利用時はGNU import libraryを生成します。

## 公開先

GitHub: https://github.com/michioga/hybit

crates.io: https://crates.io/crates/hybit

API documentation: https://docs.rs/hybit

## 公開前ゲート

GitHub/crates.io公開前は、以下を実行します。

```powershell
.\public-release-gate.ps1
```

このゲートはRust/C ABI/C/C++/Fortranの実行確認に加え、crates.io向けpackage metadataとpackage生成を検証します。公開手順は [docs/PUBLISHING.md](docs/PUBLISHING.md) を参照してください。

## License

MIT Licenseです。詳細は [LICENSE](LICENSE) を参照してください。


## HyBIT 0.6 構造FEM実験

開発版には、scalar Jacobi、3x3 Block-Jacobi、並進のみのaggregationに加え、
3次元構造問題向けの6剛体モード（Tx/Ty/Tz/Rx/Ry/Rz）粗空間を比較する
Matrix Marketベンチマークを含みます。Structural Autoはgraph-connected aggregateを
第一選択とし、graph coarse factorizationが数値的に特異な場合だけRCM順contiguous
aggregationへfallbackします。検証済みの構造FEM経路は
`HybitSolver::solve_structural_csr32` と再利用可能な
`HybitPreparedStructuralSystem` に統合しました。generic `solve_csr32` の挙動は変更していません。


### 0.6 structural execution policy (development)
Structural Auto can independently select parallel CSR SpMV and parallel rigid-body preconditioner kernels through `StructuralSpmvPolicy` and `StructuralPreconditionerPolicy`. Large CSR structural systems default to parallel execution; small systems remain serial to avoid Rayon overhead. Rayon thread count is controlled externally (for example `RAYON_NUM_THREADS`).

> 開発メモ (0.6 r24): PCG のベクトル演算を並列化・融合した実験経路をベンチマーク用に追加しています。実 FEM で数値一致と実時間改善を確認するまでは、Structural Auto の本番 PCG 経路は変更しません。
