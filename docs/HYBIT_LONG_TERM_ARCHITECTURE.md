# HyBIT 長期アーキテクチャ仕様 / Long-term Architecture Contract

- **決定日:** 2026-10-13 (JST)
- **対象:** HyBIT `develop/0.9.0` 以降。将来バージョンにも設計原則として継承する。
- **分類:** 設計上の確定方針。以下の「実装予定」を実装済みと扱わない。
- **優先順位:** 本文の不変条件 > 実験的な高速化案。衝突する最適化は採用しない。
- **参照:** `AGENTS.md`, `docs/PROJECT_CHARTER.md`, `docs/ABTM_TOPOLOGY.md`, `docs/ROADMAP.md`, `docs/SESSION_HANDOFF.md`。

## 1. 製品理念 — 変更不可の不変条件

HyBIT は **Autonomous Hybrid Sparse Solver**。FEM/HPC向けの大規模疎線形系で、ユーザーにとって単純、内部では頑健かつ自律的なソルバとする。

1. **反復法を基本とし、難しい部分だけ局所直接法で補強する。** 収束しやすい系・領域には余計な分解を行わず、PCG/FGMRES等を使用する。停滞・breakdown・局所的な悪条件を検出したら、メモリ予算内で局所Cholesky、ピボット付き局所LDLᵀ/LU、Schwarz・粗空間等を適切に適用し、全体系の反復解法を継続する。局所直接補正 ≠ 全体系直接解法。全体系の疎直接法フォールバックは、数学的適合性と予算を確認して将来導入する。
2. **公開APIは極力単純に保つ。** 代表例 `hybit::solve(&a, &b)` と `analyze → prepare → solve-many` を尊重する。解法、前処理、構造表現、並列方式、領域・通信の選択は可能な限り内部に隠蔽する。高度なオプションは明示的に利用する専門家用とする。
3. **正確性・収束診断を犠牲にしない。** 真の残差、有限性、停滞、breakdown、反復予算、メモリ予算、数学的行列クラスを監視する。未収束を収束と報告しない。数学的クラスを収束挙動だけから証明しない。
4. **性能は end-to-end と再利用を含めて評価する。** 事前計画・分解・変換、solve-many、通信、ピークメモリを測る。マイクロベンチだけで全体の高速化を宣言しない。
5. **既存の検証済み実装とAPI互換性を優先する。** 新アーキテクチャへの移行は小さなゲート付き変更とし、CSR数値実行経路等はベースラインとして保持する。

## 2. ABTM の役割: Metadata-first Execution Planner

「CSRをABTMに全置換してSpMVを高速化」ではなく、**事前のビット論理演算で、厳密に不要と判断できる演算・値ロード・通信依存を削る**のを中心にする。

- 構造: AND / OR / AND-NOT / XOR / popcount / rank / select; 行列グラフ、領域拡張、拘束・Active Set、接触ペア、記号分解、局所部分行列、Halo依存集合。
- 数値: 物理的な数値計算は、処理・デバイス・密度・ブロックに応じて prepared CSR / ABTM / DenseBlockCsr 等の適切なカーネルを選ぶ。**CSRの優位性を測定した場合はCSRを使用**する。
- メタデータはゼロの意味を明確化する。`structural_presence`（格納要素が構造上存在）、`numeric_activity`（数値的な有効性）、`free/constrained_dof`、`field`、`ownership`、`contact_candidate/active/stick/slip` を混同しない。
- 厳密な通常SpMVでは、構造上必要な非零項を任意に省略できない。優先対象はマスク・制約・局所補正・記号演算・ブロック単位の不要演算の除外。省略判定に数値許容差を使う場合は、誤差予算と独立検証を設け、既定では厳密路線を維持する。
- 長期的な正本は reusable symbolic topology + numeric-value storage。実行用preparedレイアウトは派生キャッシュとし、二重保持や変換ピークが不利なら保持しない。
- 既存の `AbtmTopology`, `AbtmDualTopology`, `AbtmMetadataFirstMatrix`, `AbtmMatrix`, G3/G4/G5/G6 を再利用。再実装の乱立を避ける。

## 3. 混合自由度・マルチフィジックス対応

熱・ビーム・シェル・ソリッドの**異なる自由度数**を同じ連成モデルで扱う。全節点一律3自由度や一律6自由度を前提にしない。

- 自由度のオーナーは node のみと仮定しない。node / edge / face / element / multiplier / constraint を許容し、グローバルDOFと物理fieldの対応を記述する `FieldAwareDofLayout`（仮称）を設計。
- 代表例: 熱温度T、3D変位3、梁の並進3+回転3、シェルの並進+定式化依存の回転、熱変位連成、接触乗数（法線・接線）。固定3×3/6×6だけでなく**可変ブロック径**とfield単位の分割・Schur補完を検討する。
- 行列の内部DOF順序とユーザーの物理DOF IDを、明示的なpermutation / inverse mappingで分離。強制変位、MPC、拘束消去、乗数法、penalty等がfieldを破壊しないようにする。
- 単位・スケール差（温度、並進、回転、乗数）に対応するスケーリング、field-wise norm とブロック前処理が必要。無条件に対称SPDと仮定しない。
- FEM側が要素定式化・組立・材料則を担う。HyBITの線形ソルバ側は行列、作用演算子、DOF/field/constraintメタデータを受け取り、物理要素そのものへ依存しない。

## 4. 摩擦付き接触・非線形対応

**摩擦付き接触は長期の必須対応目標**であるが、現時点で実装済みと記載しない。接触探索と線形方程式の解法を分離する。

- 外側: 非線形増分制御（Newton/semismooth Newton、line search / trust-region、負荷刻み調整、active-set更新、接触探索）。FEM/接触モジュールと連携。
- 内側: 線形化した接線系 `J_k Δx = -R_k` をHyBITが解く。接触により非対称、対称不定、鞍点、特異/近特異、急激な構造変化が生じ得る。
- 接触ステータスは `open`, `stick`, `slip` と候補集合を別々のマスクで管理。接触判定には幾何・法線・ギャップ・摩擦則の数値計算が必須で、ビット論理演算だけで状態を決定できない。
- 乗数法のブロック・Schur補完、penalty/augmented Lagrangian、active-set、拘束消去方式は**相互に異なる数値的契約**を持つ。実装選択を実測・正確性で判断する。
- 非線形の収束は線形残差だけで判定せず、釣合い残差、ギャップ/貫入、相補性、接触力の符号、摩擦円錐・stick/slip整合を評価する。
- Active Set変化時に変わるのが数値値のみか、構造/DOF/通信依存かを区別し、該当キャッシュだけ失効させる。
- 現状の `MatrixProblemClass::SymmetricIndefinite` は認識対象だがMINRES未実装。将来の対称不定系はMINRES（前処理の数学的条件を満たす場合）または適合する代替Krylov法。非対称系は既存FGMRESを起点とする。局所直接法はSPD Cholesky、不定 LDLᵀ（適切なピボット）、一般 LU（適切なピボット）等を行列クラスに応じて選ぶ。

## 5. Autonomous Solver Controller

- エスカレーション: 問題クラス確定/契約確認 → 軽量Krylov+適切な既定前処理 → true-residualの停滞/不安定監視 → hard-region抽出 → 局所直接分解/Schwarz/二段階粗空間 → 再スタート/可変前処理適合Krylov → 予算と根拠がある場合のみ全体系直接フォールバック。
- 難しい部分は**局所的な残差・強結合・スケール不均衡・スペクトル情報・拘束/接触状態**で識別するが、診断自体の費用が便益を上回らないよう軽量な観測から始める。
- 局所補正の寿命: 構造が同じなら記号因子や領域情報を再利用し、値変化時には必要な数値分解を更新。pivot破綻/数値非有限/メモリ超過で安全にフォールバックする。
- 前処理変更時のPCG/FGMRESの数学的契約を守る。固定SPD前処理を要するPCGを、可変不定前処理のまま継続しない。
- 結果には相対真残差、反復数、収束status、行列クラス、選択された経路/前処理、昇格理由、メモリ・時間、警告を記録。失敗は理由を区別し、成功を偽らない。

## 6. Rayon / MPI の役割と通信削減

- Rayonは**共有メモリ**上の並列実行。MPI型ネットワークHaloと同一視しない。共有read-only入力、排他的出力、偽共有・NUMA・余分なコピー・スレッド間cache移動の抑制を優先する。
- MPIはrank-local所有とghostを明確にし、正確に必要な一意ghost集合とpeer関係を最小化する。メタデータの集合演算はHalo**計画生成**を効率化するが、同一の所有分割・演算・厳密性の下で必要な`f64`値を魔法のように削れない。
- 構造/field/接触を考慮した所有分割とrank負荷均衡で通信量を下げる。評価関数は ghost数、受信/送信 bytes、peer数、メッセージ数、負荷不均衡、Allreduce回数、真のtotal time。
- 接触時は通常要素Haloとは別に、動的contact依存を受けられる設計とし、通信計画をepoch管理する。ランク間の接触ペアと所有の再マッピングに注意。
- 既存 `mpi_input.rs`, `mpi_overlap.rs`, `mpi_pcg.rs` とB3/C2の成功結果を再利用。MPIスレッド契約（現行のMPI_THREAD_SINGLE前提）を守り、MPI呼び出しとRayon計算の責務を分離する。
- 通信と内側行の計算を重ねる場合、通信待ちが残る可能性がある。MPI_Allreduceを含むKrylovのグローバル同期も測る。近似圧縮や遅延Haloは誤差制御・正確性が確立するまで自動採用しない。

## 7. Lifetime / キャッシュ失効契約（実装予定）

| 入力の変化 | 再利用できる主な候補 | 再評価・失効する主な候補 |
|---|---|---|
| RHSのみ変更 | topology, numeric matrix, partition, halo, factors（行列値不変なら） | Krylovの右辺依存状態 |
| 行列値のみ変更 | topology, DOF mapping, partition, structural halo | 数値分解、数値依存診断、収束モデル |
| 境界条件・拘束状態変更 | 不変な構造部分 | active mask、消去/乗数対応、数値と場合により構造 |
| 接触active-set変更 | 不変なfield layout、幾何キャッシュの一部 | 接触接線値、接触構造、動的halo、因子の該当部分 |
| reordering / partition変更 | 物理DOF ID対応 | internal mapping、local layout、halo、prepared kernels |
| mesh / DOFの追加・削除 | 検証済みの不変データのみ | 関連構造・数値・分割情報の再構築 |

構造/値/field/constraint/contact/partitionそれぞれのversionまたはgenerationを持ち、依存グラフを通じて失効させる。再利用可能という理論と高速化したという実証を分ける。

## 8. 実装・公開境界

- **既存:** SPD PCG+選択的局所Cholesky、GeneralSquare FGMRES+明示ILU0、ABTMの構造演算、G5固定3/6ブロックの研究、G6 prepared restricted CSR、Rayon処理、MPI rank-local CSR/Halo/PCGの実験と検証（詳細はvalidation baseline）。
- **未完:** 混合DOF/field-aware構造、摩擦付き接触の全体非線形制御、動的contact Halo、対称不定MINRES、局所ピボット付き一般LU/LDLᵀの自律統合、一般化された全体系疎直接法、混合物理クラスの完全自動選択。
- `hybit::solve` は疎**線形**方程式のAPI。将来の `nonlinear::solve(problem)` は*概念案*であり実装済みの公開APIではない。ソルバとFEM/接触アセンブラの境界を保つ。
- 全ての導入は参照実装との数値一致、真の残差、非線形収束条件、異なる行列クラス、peak memory、MPI実機、held-out corpus、end-to-end性能で判定する。

## 9. 実装の優先順位（上から順に、並列進行可）

1. **Contracts-first:** 数学的行列クラス、自律制御・局所直接法、混合field/可変DOF、構造/値世代、MPI/contact haloの不変条件をテスト可能に文書化。
2. **Field-aware topologyの最小実装:** 可変DOF+permutation+拘束マスク。CSRベースラインとの完全一致テスト。外部APIを増やさない。
3. **ABTM metadata executor:** 固定マスク・region・局所記号解析を対象に候補演算数/実演算数と費用を測る。既存G6方式を優先。
4. **頑強な線形系の拡張:** 一般非対称・不定系の妥当な局所補正、field/block Schur、真の残差、前処理可変性の保護。
5. **MPI/Rayon通信・メモリ最適化:** 分割/halo、hybrid threads、動的contact dependency、Allreduce、負荷均衡を段階検証。
6. **接触段階試験:** 線形混合field → 摩擦なし接触 → stick/slipを伴う摩擦付き接触 → 分散contact、の順で外側FEM/非線形コントローラと統合。
7. **自律制御とAPI統合:** 裏付けのあるpolicyのみデフォルトへ昇格、単純API・C/Fortran互換・release gates維持。

## 10. 採用ゲート / Definition of Done

- `Correctness`: CSR/ABTMと参照系の厳密または所定精度内一致。SPD宣言の根拠、非対称・不定系の安全なrouting。真の残差検証。
- `Convergence`: 接触なら補完性・力釣合い・摩擦則まで合格。局所直接化で収束悪化の可能性も記録。
- `Memory`: 持続・変換中ピーク・rank毎・因子/ワークスペース別。予算超過で安全な失敗。
- `Performance`: startup、preparation、反復、通信、solve-many、合計の測定とheld-out行列・実FEM行列での比較。
- `MPI`: 実MS-MPIで1/2/4/8 ranks、各epochで一致・no deadlock、collective順序・thread contract・ghost送受信一致。
- `API`: 既定は少ない引数、診断は充実、失敗理由は明確。後方互換性とソース整合性gate。

## 11. この設計を後のセッションに引き継ぐためのルール

最初に `AGENTS.md` → `docs/PROJECT_CHARTER.md` → **この文書** → `docs/SESSION_RESUME_PROTOCOL.md` → `docs/SESSION_HANDOFF.md` → `docs/HYBIT_VALIDATION_BASELINE_2026-10-13.md` → `docs/ROADMAP.md` を読む。最新の実装事実はユーザーのローカルcheckoutとログを優先する。方針に反する提案はしない。検証ごとにhandoff、baseline、必要ならdesign decision logを更新する。commit/pushはユーザーの判断・承認を尊重し、Release/crates.io公開は別承認とする。
