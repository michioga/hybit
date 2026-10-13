# HyBIT Architecture Decision Log

設計上の確定事項と保留事項を区別する。後のセッションで結論が変わる場合は、過去の決定を無言で上書きせず、日付・新証拠・互換性・rollback条件を追記する。

| ID | 日付 | 判定 | 決定 | 理由・帰結 |
|---|---|---|---|---|
| ADR-001 | 2026-10-13 | **確定** | 「収束しやすい系/領域は反復法、難しい部分は局所直接法」で全体Krylovを補強する | HyBIT本来の自律ハイブリッドソルバ思想。行列クラス・予算・真の残差に従う |
| ADR-002 | 2026-10-13 | **確定** | 公開APIは極力シンプル。内部の高度な判断は自動 | `hybit::solve` とprepared solve-manyは尊重。詳細policyは専門家向け |
| ADR-003 | 2026-10-13 | **確定** | ABTMの中心用途はmetadata-first pruning / symbolic planning | C5で普通のABTM SpMVはCSRより遅く、全置換しない |
| ADR-004 | 2026-10-13 | **確定** | Rayonは共有メモリ転送、MPIはghost/Haloとcollectiveを削減 | 似た「通信」でも実装対象と制約が違う |
| ADR-005 | 2026-10-13 | **確定** | 熱/ビーム/シェル/ソリッドの混合DOF・可変ブロックを設計要件とする | 一律3/6 DOFへの固定は不可、field/constraint metadataを別管理 |
| ADR-006 | 2026-10-13 | **確定** | 摩擦付き接触を長期必須目標とする | 外側の接触/非線形制御と内側の頑強な線形ソルバを分離する |
| ADR-007 | 2026-10-13 | **確定** | 非対称・不定系の安全なrouting、真の残差、メモリガード | PCGを不定系へ盲目的に適用しない。MINRES等は未実装の候補 |
| ADR-008 | 2026-10-13 | **確定** | 実測に基づいてバックエンドを選び、既存高速CSR経路を残す | ABTMを採用すること自体を目的にしない |
| ADR-009 | 2026-10-13 | **確定** | Session handoffとsource integrityを必須の開発工程にする | AI会話履歴だけに依存しない。Git管理文書を正本にする |
| OPEN-01 | 2026-10-13 | **要実験** | 混合fieldの公開メタデータAPIと内部表現 | 既存ユーザーAPI互換・C/Fortran APIへの影響を評価 |
| OPEN-02 | 2026-10-13 | **要実験** | 接触の非線形制御層をHyBITで直接提供する範囲 | 外部FEMとの境界・semismooth Newton・line search |
| OPEN-03 | 2026-10-13 | **要実験** | MPI partition, contact HaloとABTMの最良の実装 | ghost量と負荷均衡を実測、static/dynamic epochsを比較 |
| OPEN-04 | 2026-10-13 | **要実験** | 一般不定系向けlocal LU/LDLᵀ、Schur、global direct fallback | 信頼できるピボット・特異性・予算・Krylov数学的整合性 |
| OPEN-05 | 2026-10-13 | **要実験** | ABTM構造正本からCSR実行レイアウトをいつ派生するか | 準備コスト、solve-many償却、ピークメモリ・性能を比較 |

**重要:** このログは開発方針の合意事項であり、OPEN項目の機能完成を示すものではない。
