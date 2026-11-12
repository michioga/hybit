# P3-B — opt-in diagonally equilibrated local Cholesky (WSL research gate accepted)

**Owner WSL P3-B gate PASS (2026-11-12):** in 7 analytically classified 4-DOF cases, feature OFF had 3 contract discrepancies and feature ON had 0. Accepted SPD true residuals satisfied the research gate; 4-DOF stored factor bytes are 248. The strongly ill-conditioned congruence case retains a forward relative error of 0.17829873501122737; no forward-accuracy guarantee, real-FEM rescue, or production default promotion is claimed. Evidence: `~/c1h11_evidence/p3b_local_equil_9mVcNf`. The original `P3B_validate.sh` had a Python module launch error after the Rust and numerical passes; the module was run manually and passed. The checkpoint package includes a corrected validator, without modifying solver sources.


Baseline: owner WSL `develop/0.10.0` HEAD `fdb047e`, plus applied **uncommitted** P2-B FIX1 and P3-A audit. Owner P3-A evidence: `~/c1h11_evidence/p3a_local_scale_rsYbIP`, 7 cases, 3 discrepancies: `spd_small_1e16`, `spd_congruence` rejected despite constructive SPD, and `nonsym_small_1e12` accepted despite asymmetry (relative residual about 0.04). These are measurements, not successes. Do not overwrite the P3-A evidence.

## Hypothesis

Existing local Cholesky tests use `1e-11 * max(scale,1)` for symmetry and `1e-14 * max(scale,1)` for pivot acceptance, so even an SPD principal matrix obtained from `A = s D T D`, where `T = tridiag(-1,2,-1)`, can be rejected for small `s` and highly unequal positive `D`. Tiny nonsymmetric matrices can be falsely admitted. These issues prevent reliable local-direct rescue in mixed translation/rotation DOF units.

## Proposed isolated change

Only under `hybit-precond/p3-equilibrated-local-cholesky` (feature default **off**):

1. Require positive, finite local diagonals. Let `D_ii = sqrt(A_ii)` and set `B = D^-1 A_R D^-1` where `A_R = A[R,R]`.
2. Compare corresponding normalized offdiagonal pairs by a *pair-relative* tolerance `abs(bij-bji) <= 1e-11*max(abs(bij),abs(bji))`; do not use a global absolute floor. Average pairs passing the check.
3. Factorize packed lower `B` with Cholesky. Its unit diagonal gives a fixed dimensionless pivot floor `1e-14` (numerical, not a formal condition certificate).
4. Apply the local inverse as `D^-1 * (L^-T L^-1) * D^-1` using an `n`-element saved inverse diagonal square-root. No per-iteration allocations. One region adds `8*n` persistent bytes; the factor estimate and actual bytes must still match.
5. Preserve feature-OFF factor code and default solver behavior. No change to ABTM, MPI, public facade, C ABI, high-level routing, or PCG settings. Hybit facade does **not** expose this feature yet; this is a preconditioner-only research gate.

The feature adds one persistent f64 diagonal scale vector per local region; production local-factor memory budgets require separate end-to-end validation before enabling by default.

## WSL owner gates

Extract ZIP directly into `D:\Work` to obtain its dedicated subdirectory, then execute from WSL:

```bash
cd ~/hybit
PKG=/mnt/d/Work/HyBIT_0_10_P3B_EQUILIBRATED_CHOLESKY_20261112
bash "$PKG/P3B_apply.sh" --dry-run "$PWD"
bash "$PKG/P3B_apply.sh" --apply "$PWD"
bash "$PKG/P3B_validate.sh" "$PWD"
```

Validate feature OFF/ON Rust 1.99.1 unit/integration/example tests, strict Clippy, formatting, SHA-256, six independent Python gate tests and P3-A baseline audit. Capture full binaries and source hashes, seven-case logs, CSV/JSON, whole-process RSS/time. The gate requires baseline OFF 3 discrepancies and feature ON 0; all accepted SPD systems must have independent original-matrix true relative residual <=1e-8 and actual/estimated factor memory 248 bytes in the 4-DOF fixture. The 2D physical FEM / shell / beam class is **not yet tested**.

## Important: forward error can still be poor for the congruence case

The `diag([1e-8,1,1,1e8])` SPD case has severe conditioning. Forming `b=A*x_ref` and storing it in f64 may round away information about the small-scale DOF. Even an algebraically correct equilibrated factor may return a solution with small true residual but significant forward error. This is expected to be reported separately and is not accepted as a forward-accuracy guarantee. For the other four well-scaled SPD pilots, the gate requires forward error <=1e-8.

## Rollback and next stage

The commit remains unchanged and production feature defaults OFF. Do not commit/push until owner WSL logs are accepted. Before production promotion, add held-out *actual* structural/shell/beam matrices, compare rescue rates against Jacobi and coarse controls, inspect factor memory and peak RSS, run abi/MPI unaffected gates, and confirm solve-many reuse. No general SPD eigenvalue proof or large-FEM performance claim follows from these 7 small cases.
