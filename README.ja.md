# HyBIT

> **開発状況:** このソースツリーは `develop/0.7.0` 上の HyBIT 0.7.0 を対象としています。0.7では検証済みの0.6構造FEM経路を維持しつつ、generic algebraic two-level coarseとcoarse-first resumable PCG controllerを追加しています。

**HyBIT — Autonomous Hybrid Sparse Solver** は、FEM/HPCで現れる大規模疎行列を対象としたRust-firstの線形ソルバーフレームワークです。

低コストな反復法から開始し、収束状況を観測し、進捗が悪い場合には数値的に難しい自由度を抽出して、限定された局所領域だけをCholesky直接法へ昇格させます。ABTMのbitmap topology metadataは、選択領域の近傍展開や構造処理に内部利用します。利用側は通常のCSR32行列を渡すだけです。

> **現在の位置づけ:** HyBIT 0.7.0 はpre-1.0の実験的リリースです。自動ソルバー経路は現在、実数の対称正定値（SPD）行列とPCGに限定されています。1.0までAPIが変更される可能性があります。

## 主な機能

- Rustを中核とし、C ABI経由でC/C++/Fortranから利用可能
- 公開入力形式はCSR32、ABTMは内部backend/topology engineとして利用
- 再利用可能なworkspaceと、前処理器が変わらないcontroller stage間で継続できるPCG session
- generic algebraic two-level coarse（Graph aggregation、Jacobi-smoothed transfer、transfer storage/value policy、coarse apply policy）
- 収束進捗の自動probeとselective local direct escalation。前処理器が実際に変わる場合だけPCG restart
- 複数hard regionとweighted overlapping Schwarz correction
- SPD principal submatrixに対するbounded local dense Cholesky
- `analyze -> prepare -> solve-many`
- 学習済みlocal factorとKrylov workspaceの再利用
- 収束、時間、region、factor memory、reuseを返す`SolveReport`
- MIT License

## crates.ioから利用

```bash
cargo add hybit@0.7.0
```

または、`Cargo.toml`へ以下を追加します。

```toml
[dependencies]
hybit = "0.7.0"
```

0.7.0がcrates.io indexへ反映される前のrelease-candidate検証では、GitHub上の同一ソースツリーを使用してください。

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
  |-- base preconditioner選択
  |      |-- algebraic coarse明示時はcoarse
  |      +-- それ以外はJacobi
  |-- 短いcontroller PCG stage
  |      |
  |      +-- progress良好 -> 同じPCG sessionを継続
  |      |
  |      +-- progress不良
  |             |-- residual/risk mask
  |             |-- hard region分離
  |             |-- ABTM halo展開
  |             |-- local Cholesky
  |             +-- 前処理器を強化した場合だけPCG restart
  v
local factor / coarse state cache
  v
solve #2..N
  |-- coarse/local factor再利用
  |-- workspace再利用
  +-- cached local Hybridならdiagnostics/factorization省略
```

PCG反復の途中で前処理器を変更せず、前処理器を強化するときはKrylov系列をrestartする設計です。

## 0.7.0のリリース内容

0.7.0では、0.6で検証した構造FEM経路を維持したまま、r23-r32で検証したgeneric algebraic coarse経路を追加します。主な追加点はGraph aggregation、Jacobi-smoothed transfer basis、parallel transfer、Wide/Compact index storage、F64/F32/Auto transfer-value storage、FactorSolve/ExplicitInverse/Auto coarse apply、明示coarseをiteration 0から使うcoarse-first controller、前処理器が変わらない場合の`PcgSession`継続、prepared coarse-only reuseです。

r33-r36で実験したwatchdog、energy gate、filtered spectral enrichmentは0.7.0には含めず、post-0.7の研究項目として分離します。ベンチマーク値は回帰・設計判断のための測定であり、一般的な高速化を保証するものではありません。

## 現在の制約

HyBIT 0.7.0は`f64`、square SPD、PCGが中心です。local directはbounded dense Choleskyで、既定では最大128 DOFのregionを最大8個まで使用します。prepared factor reuseは行列構造と係数bit列が完全に同一の場合に限ります。3次元構造FEM向けには6剛体モードの二段coarse correctionを実装済みですが、一般的なAMGではありません。MINRES/GMRES/BiCGStab、MPI、GPU、out-of-coreはまだ未実装です。prepared contextは現段階ではsingle-threaded利用を想定しています。

したがって、現時点のHyBITは実験的な数値ソフトウェアです。工学的判断へ利用する場合は、残差だけでなく物理量・参照解・独立ソルバー等による検証を行ってください。

## 実FEM / Matrix Marketベンチマーク

0.6で導入し0.7でも維持しているベンチマーク経路では、拘束条件適用後のSPD剛性行列をMatrix Market (`.mtx`) から読み込み、同じ初期値・許容誤差・最大反復数でplain Jacobi-PCGとHyBIT Autoを比較できます。

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

0.7.0 release candidateでは `release-candidate-gate.ps1` を正式なclean-tree gateとします。source hash、workspace metadata、fmt/Clippy、Rust 1.73 MSRV、Rust/C ABI/C/C++/Fortran、crates.io package、実L-angleの収束・独立残差・反復数guard、prepared solve-many reuseまで一括確認します。L-angleの大規模入力自体はrepositoryへ含めず、外部パスを渡します。

```powershell
.\release-candidate-gate.ps1 `
  -Matrix D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -RayonThreads 8
```

`-SkipMsrv` / `-SkipRealFem` は途中確認用です。skipを使ったrunはtag/publish可能なrelease gate PASSとは扱いません。公開手順は [docs/PUBLISHING.md](docs/PUBLISHING.md) を参照してください。

## License

MIT Licenseです。詳細は [LICENSE](LICENSE) を参照してください。


## HyBIT 0.6 構造FEM基盤（0.7でも維持）

0.7ソースツリーには、scalar Jacobi、3x3 Block-Jacobi、並進のみのaggregationに加え、
3次元構造問題向けの6剛体モード（Tx/Ty/Tz/Rx/Ry/Rz）粗空間を比較する
Matrix Marketベンチマークを含みます。Structural Autoはgraph-connected aggregateを
第一選択とし、graph coarse factorizationが数値的に特異な場合、または6剛体モードを構成できないほど小さい非連結componentを含む場合にRCM順contiguous
aggregationへfallbackします。検証済みの構造FEM経路は
`HybitSolver::solve_structural_csr32` と再利用可能な
`HybitPreparedStructuralSystem` に統合しました。generic `solve_csr32` の挙動は変更していません。


### 0.6構造FEM基準点

0.6 r25で検証したCPU構造FEM経路、すなわちGraph rigid-body aggregation、packed coarse Cholesky、Parallel CSR SpMV、Parallel rigid-body preconditioner、Parallel/fused PCG vector kernelsを0.7でも構造FEM基準経路として維持します。0.7のgeneric algebraic coarse/controller追加はこの構造APIを置き換えません。


### 構造FEM execution policy
Structural Auto は `StructuralSpmvPolicy`、`StructuralPreconditionerPolicy`、`StructuralPcgVectorPolicy` により、CSR SpMV、剛体二段前処理、PCG密ベクトルカーネルを独立に並列化できます。大規模構造問題では並列経路を選択し、小規模問題ではRayonオーバーヘッドを避けるためserialを維持します。PCG vectorの`Auto`は共有Rayon poolが4 worker以上の場合にのみ有効化されます。thread数は`RAYON_NUM_THREADS`等で外部から制御します。

> 0.6 r25 回帰基準値: 358065 DOF / 28.24M nnzのL-angle実荷重問題で、8 Rayon worker時にStructural AutoはGraph aggregation + Parallel CSR SpMV + Parallel rigid-body preconditioner + Parallel/fused PCG vectorsを選択しました。220反復、verified relative residual `9.378557e-9`、solve 1.726秒、analysis+prepare+solve 2.797秒でした（Ryzen 7 7800X3D上の開発測定値であり、一般的な性能保証ではありません）。r25 structural pathは0.7でも回帰基準として維持します。

### Hybrid coarse dimension sweep

`bench-fem-hybrid-coarse-sweep.ps1` は selective-direct only を基準として一度だけ Plain Jacobi-PCG を実行し、その後は `--skip-plain` を使って複数の algebraic coarse target を比較します。既定 target は `384, 512, 768, 1024, 1536` です。結果は画面の比較表と `hybit-hybrid-coarse-sweep.csv` に出力されます。
