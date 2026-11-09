# C1-H12-A4 owner WSL numerical evidence: true-residual discrepancy

Date: 2026-11-09. **Transcribed from owner's terminal output**, not raw
log/file ingestion. Owner Ubuntu 26.04 WSL2 `develop/0.10.0`, HEAD
`4593c20` plus uncommitted opt-in A2/A3/A4; Rust 1.99.1 Release.

Original process logs, GNU time files, SHA-256 matrix input manifest,
HEAD marker and numerical CSV remain at:

`~/c1h11_evidence/c1h12_a4_native_OPAkkT/`

All 21 of 21 processes exited zero, Rust A4 numerical gate unit tests
10/10 PASS, release Clippy/build PASS, but the real-FEM numerical gate
**FAIL (15/21 accepted, 6 false convergence declarations)**. Do not
change these six failures to successes without new true-residual evidence.

The three n=20,736 inputs share manufactured `b=A*ones`, start `x=0`,
relative tolerance `1e-10`, 600 max iterations, one CPU thread,
coarse target 512, dofs/node=1, transfer F64.

## Native coarse: six false recursive-convergence declarations on multi_patch

| Arm | Reported status | Iterations | Recursive relative residual | True relative residual | Error to ones | Setup ms | Solve ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Jacobi | MaxIterations | 600 | 1.025053e-9 | 1.152375e-9 | 3.425491e-8 | 0.180 | 60.635 |
| Contiguous / piecewise | Converged **FALSE** | 342 | 8.138751e-11 | 7.160352e-10 | 1.064641e-9 | 15.731 | 140.594 |
| Contiguous / smoothed | Converged **FALSE** | 213 | 5.789481e-11 | 4.937748e-10 | 1.921477e-8 | 19.298 | 102.460 |
| Graph / piecewise | Converged **FALSE** | 116 | 4.126375e-11 | 3.925232e-10 | 1.557181e-10 | 18.001 | 43.875 |
| Graph / smoothed | Converged **FALSE** | 94 | 7.484315e-11 | 4.087384e-10 | 4.199827e-11 | 19.812 | 42.684 |
| StrongGraph / piecewise | Converged **FALSE** | 128 | 8.993694e-11 | 3.423971e-10 | 3.376041e-11 | 22.873 | 49.294 |
| StrongGraph / smoothed | Converged **FALSE** | 94 | 9.910556e-11 | 4.096047e-10 | 7.800341e-12 | 25.818 | 42.134 |

The solution error is small for several coarse arms, but this does not
justify a success label against a strict true-residual tolerance.
The mismatch is consistent with accumulated PCG recursive-residual
roundoff; its precise numerical floor has **not been established**.

## Rotated anisotropy: five independently certified convergences

| Arm | Status | Iterations | True relative residual | Error to ones | Setup ms | Solve ms |
|---|---|---:|---:|---:|---:|---:|
| Jacobi | MaxIterations | 600 | 1.956624e-7 | 3.649857e-6 | 0.194 | 72.694 |
| Contiguous / piecewise | MaxIterations | 600 | 2.686215e-10 | 4.970944e-9 | 16.670 | 261.977 |
| Contiguous / smoothed | Converged | 547 | 9.782740e-11 | 1.888233e-9 | 20.638 | 277.305 |
| Graph / piecewise | Converged | 405 | 9.755936e-11 | 1.135723e-9 | 17.071 | 156.702 |
| Graph / smoothed | Converged | 325 | 9.557501e-11 | 1.044163e-9 | 21.436 | 157.151 |
| StrongGraph / piecewise | Converged | 378 | 9.571676e-11 | 1.099370e-9 | 20.951 | 147.405 |
| StrongGraph / smoothed | Converged | 300 | 9.860491e-11 | 8.846676e-10 | 24.078 | 142.143 |

In `island_shell`, Jacobi converged in 387 iterations with true relative
`9.429e-11`, and all six coarse arms also passed the true-residual gate.
Their individual counts/costs are retained in owner raw logs but were not
reprinted in this conversation. There is no general solve-time speedup
claim: coarse setup/solve is often more expensive despite fewer iterations.

The earlier PETSc pilot reported Jacobi not converged in `island_shell`
after 600 iterations. Matrix/RHS/effective solver normalization parity
must be checked before a PETSc iteration-count head-to-head comparison.

## Numerical decision

**Keep six multi_patch false declarations rejected.** Test an opt-in
true-residual-certified PCG restart with the unchanged original CSR,
preconditioner, absolute iteration budget and RHS. Record every rejected
recursive convergence and the extra SpMV/restart cost. If replacement
does not achieve the strict threshold, report MaxIterations or Breakdown,
not an inferred success. Source: `docs/C1H12_A5_CERTIFIED_PCG_PILOT.md`.
