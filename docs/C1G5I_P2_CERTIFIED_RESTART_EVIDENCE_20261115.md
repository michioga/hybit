# C1-G5I — P2 certified restart, source-pinned differential (2026-11-15)

Research-only comparison at `b918b4591d27a6c23e3571075eb426babcf080bb` (branch `develop/0.10.0`). Existing `p2-certified-restart` is **opt-in**; production Auto/ABI remain unchanged.

## Protocol and complete evidence

Three constructed SPD FEM matrices, manufactured `ones` RHS, four arms: T direct, Adaptive coarse/local, coarse-only/no-monitor, and coarse-only/observe. P2-default-off twelve-process set compared to independently rebuilt P2-opt-in twelve-process set. Both process sets were reconstructed and independently verified against original stdout/stderr, GNU time/exit code, the three pinned matrices, separate binary SHA, original-CSR and Kahan residuals and RHS hashes. Exactly **106 original evidence files** are retained outside Git and their SHA-256 fingerprints are recorded in the raw index. Each CSV row is a retained measurement, including failures. The research helper code and error-handling tests are registered in Git.

## Key matched results

| Matrix | Direct T | Coarse-only P2 off | Coarse-only P2 on | Adaptive P2 on |
| --- | --- | --- | --- | --- |
| scalar_multi_patch / ones | Converged 126 | Breakdown 116, CSR 3.925231781887e-10 | Breakdown 124, CSR 1.301216671805e-10 | Breakdown 124, no local |
| elastic_stiff_lens / ones | Converged 430 | Breakdown 429, CSR 1.000107258137e-10 | **Converged 430**, CSR 8.675448239553e-11 | MaxIterations 600 with 8 local regions |
| elastic_shear_layers / ones | MaxIterations 600 | MaxIterations 600, CSR 9.092695679538e-8 | MaxIterations 600, CSR 9.092695679538e-8 | MaxIterations 600, local used, CSR 1.377106302179e-7 |

The scalar no-local controller path remains uncertified above `1e-10` even with one P2 retry; the elasticity stiff-lens coarse-only path is rescued by P2 but local escalation is independently detrimental. Observation-only and unsegmented controls have equal status/iterations/residual; no evidence of monitoring segmentation as the primary cause. `elastic_shear_layers` is far from the certificate threshold and requires low-energy-mode/coarse-space diagnosis.

## Guardrails

P2 final original-CSR verification is not relaxed and a nonconverged arm is never classified as successful. P2 retries and standalone direct-T certified retry differ and must not be conflated; the exact cause of the remaining scalar shortfall is not yet proven. Separate feature-enabled and default binaries, single process measurements, and constructed FEM do not support speed superiority or selector promotion. Do not activate P2 or local-to-coarse switching in production by default.

Next research: bound/replay scalar post-certificate residuals on a fixed coarse preconditioner, including restart attempts and remaining iteration budget; separately investigate local escalation and low-energy/near-null elasticity modes, followed by independent FEM validation.
