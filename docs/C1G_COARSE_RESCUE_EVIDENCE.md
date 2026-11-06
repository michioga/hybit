# C1-G2: constructed SPD FEM coarse-space convergence-rescue evidence

Date: 2026-11-06 (JST). Status: WSL2 pilot evidence, **experimental opt-in only**.
Source baseline: HyBIT develop/0.10.0, commit `72677b6`. No solver or public API
change is proposed by this checkpoint.

## Goal and mathematical scope

The C1-G corpus consists of manufactured nonconstant solutions for scalar P1
triangle diffusion on a unit square with homogeneous Dirichlet boundary
conditions. Element diffusion tensors are strictly positive definite by
construction, giving an SPD stiffness operator in exact arithmetic. This is
not an independent floating-point sparse-Cholesky SPD certificate, a physical
3D elasticity model, or a guarantee for arbitrary input matrices.

The original symmetric Matrix Market entries, RHS, reference solution and
generation metadata are retained outside Git at `~/mtx/c1g` (small corpus)
and `~/mtx/c1g_g2_hard_z32Jkd` (n=144 corpus). The independent
`check_manufactured.py` verified all nine original/generated instances with
relative manufactured-RHS error printed as 0.0000e+00. Input SHA-256 gates
passed. Each run records the source HEAD; all seven evidence directories were
subsequently confirmed to match `72677b6`, and all recorded source SHA-256
checks passed. Some early individual runs did not save input hashes, but
their common n=144 corpus has a separately checked `input_sha256.txt`.

These are **constructed pilot matrices**, not held-out physical FEM evidence.
All results below are single-process, single-thread screening observations,
not timing medians or statistical performance guarantees.

## Predeclared comparison settings

- Solver: SPD PCG, explicit CSR numeric backend, initial guess zero.
- Target true relative residual: 1e-10; maximum iterations 600, then 1200
  to distinguish budget exhaustion from intrinsic nonconvergence.
- Jacobi-only control: standalone Plain Jacobi-PCG and/or exact matching
  HyBIT observe-only path; manufactured `--rhs` and `--xref`.
- Local Hybrid: monitor every 64 iterations after 12 probe iterations;
  rate limit 0.99, one escalation, eight regions of up to 128 rows, overlap
  one, 64 MiB local-factor cap. This is a research setting.
- Coarse-only: explicitly opt in with `--hybrid-coarse`, one scalar DOF per
  node, target coarse dimension 512, `graph` aggregation, piecewise basis,
  `--monitor-only` to prevent local Cholesky escalation.
- Correctness: distinguish reported `Converged` from process exit code 0;
  use independently verified original-CSR relative residual, and report
  forward error to the manufactured `x_ref` separately.

## n=144 hard corpus: primary 600-iteration gate

Each matrix has 20,736 DOFs. The isotropic-contrast cases store 61,920 lower
triangular Matrix Market entries; rotated anisotropy stores 82,369.

| Case | Jacobi / true relative at 600 | Local Hybrid / true relative at 600 | Opt-in graph coarse-only |
| --- | --- | --- | --- |
| local_soft | Converged 447 / 9.795186e-11 | Converged 447 / 9.795186e-11; no escalation | Not tested at n=144 |
| island_shell | MaxIterations / 7.749079e-9 | MaxIterations / 1.131198e-6; 1 escalation | **Converged 99 / 8.807643e-11** |
| multi_patch | MaxIterations / 2.528508e-8 | MaxIterations / 8.848107e-9; 1 escalation | **Converged 118 / 9.910340e-11** |
| rotated_anisotropy | MaxIterations / 1.984328e-7 | MaxIterations / 1.984328e-7; no escalation | **Converged 463 / 9.807269e-11** |

Thus the opt-in graph coarse-only mode rescued **3 of the 3 pre-existing
baseline failures at the declared 600-iteration budget**, under the original
CSR true-residual gate. Neither Jacobi nor the local Cholesky Hybrid rescued
any of those three within 600 iterations. This is **not** a measured rescue
rate on an unseen population and **not** an autonomous escalation result.

For the three opt-in coarse runs, coarse dimension was 508, factor memory
2.301 MiB, coarse setup approximately 17-19 ms, and process peak RSS
12,104 / 12,276 / 12,760 KiB (case order as above, excluding local_soft).
No local Cholesky regions were used; those factors are **coarse** factors.
Peak RSS includes additional allocations and must not be equated with
persistent coarse-factor bytes.

## Same n=144 corpus: 1200-iteration follow-up

| Case | Jacobi iterations | Local Hybrid iterations | Hybrid local correction |
| --- | ---: | ---: | --- |
| local_soft | 447 | 447 | none |
| island_shell | 637 | 898 | 8 regions; 0.575 MiB |
| multi_patch | 680 | 664 | 8 regions; 0.575 MiB |
| rotated_anisotropy | 944 | 944 | none |

All eight runs converged and passed the 1e-10 verified-residual gate by
1200 iterations. Thus `MaxIterations` at 600 was a **budget-specific
failure**, never a proof of mathematical nonconvergence.

## Small-corpus nonfailure controls (target 1e-10, budget 600)

| Case | Jacobi iterations | Graph coarse-only iterations | Coarse factor MiB |
| --- | ---: | ---: | ---: |
| uniform n48 | 105 | 43 | 1.713 |
| local_soft n72 | 228 | 60 | 1.793 |
| island_shell n72 | 320 | 61 | 1.793 |
| multi_patch n72 | 336 | 70 | 1.793 |
| rotated_anisotropy n72 | 428 | 223 | 1.808 |

All controls converged and passed true-residual checks; the coarse-only
mode did not cause a convergence-status regression. These small systems
are not evidence of end-to-end speedup, because coarse setup is extra work.

## Forward-error gate: the important limitation

At target 1e-10 for n=144 `island_shell`, the first 600-step local Hybrid
ended `MaxIterations`, residual 1.131198e-6, relative forward error
5.219228e-1. Graph coarse-only converged in 99 steps with residual
8.807643e-11 but relative forward error **4.415574e-7**.
At the same matrix's 1200-step budget, Jacobi converged at iteration 637
with relative forward error 8.655092e-9. Coarse residual success must not
be presented as identical solution accuracy.

For n=144 `island_shell`, raising the precision target to 1e-12 (maximum
2400 iterations) gave:

| Method | Iterations | Verified relative residual | Relative forward error |
| --- | ---: | ---: | ---: |
| Jacobi | 674 | 9.724959e-13 | 8.516875e-9 |
| contiguous coarse | 648 | 9.553264e-13 | **2.862521e-9** |
| graph coarse | 127 | 9.350182e-13 | 1.707861e-8 |
| strong-graph coarse | 125 | 9.491178e-13 | 2.164681e-8 |

Tightening the residual criterion strongly reduced the very large
`contiguous` forward error reported at 1e-10. The unusually different
forward errors are consistent with a severely conditioned operator, but
are **not an independent condition-number estimate or a root-cause proof**.
No single residual threshold guarantees a universal forward-error bound.

## Evidence locations (private WSL2 paths)

- `~/c1h11_evidence/c1g_g2_screen_1kidTv` (n48/n72, 600 iterations)
- `~/c1h11_evidence/c1g_g2_n144_5PZl6I` (n144, 600 iterations)
- `~/c1h11_evidence/c1g_g2_n144_1200_I3ueG3` (n144, 1200 iterations)
- `~/c1h11_evidence/c1g_g2_coarse_mDK3qQ` (n144 graph coarse)
- `~/c1h11_evidence/c1g_g2_coarse_controls_pKhgSo` (small controls)
- `~/c1h11_evidence/c1g_g3_aggregation_rl4XAT` (aggregation follow-up;
  directory name does not redefine the planned C1-G3 controller work)
- `~/c1h11_evidence/c1g_g2_forward_accuracy_g4vubT` (strict target)

## Decision and remaining gates

1. Retain production defaults: the beneficial coarse arm was explicitly
   enabled **before** the solve. Autonomous coarse escalation is unproven.
2. The local Cholesky rescue candidate has a serious negative result on
   `island_shell`; do not promote a one-window residual trigger by default.
3. Require manufactured forward-error reporting alongside true residual
   for these highly contrasted SPD systems.
4. Test previously unseen grid sizes, coefficients/RHS and real physical FEM
   cases; compare failures and regressions with the full 2x2 acceptance table.
5. Compare graph aggregation quality and construction mechanism separately
   from ABTM representation/runtime efficiency. Neither graph success nor
   these timings prove an ABTM-over-CSR kernel speedup.

This checkpoint records evidence only. No code, public API or release
qualification is implied.
