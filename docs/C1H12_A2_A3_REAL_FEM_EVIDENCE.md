# C1-H12-A2/A3: owner WSL symbolic evidence (2026-11-09)

## Provenance and scope

This is a **transcription of the owner's terminal results**, not a replay
performed inside the assistant workspace. The owner used Ubuntu 26.04 WSL,
Rust 1.99.1 Release, `develop/0.10.0` HEAD `4593c20` with both opt-in
experimental examples **uncommitted**. Inputs were the generated C1-G P1
`island_shell`, `multi_patch`, `rotated_anisotropy` n=20,736 matrices in
`~/mtx/c1g_g2_hard_z32Jkd/`. A2 and A3 each used one warmup plus five
timed kernel repetitions and reported their median. The preserved raw runs
on the owner's WSL machine are:

- A2: `~/c1h11_evidence/c1h12_a2_symbolic_FH3k8I/` (3 arms).
- A3: `~/c1h11_evidence/c1h12_a3_galerkin_3Qs0BC/` (9 arms).

These paths should contain `HEAD.txt`, `matrix_sha256.txt`, per-arm `.log`
and `.time`; their contents and their matrix SHA-256 values have **not yet
been independently ingested into this repository**. Do not interpret a
transcribed table as full raw evidence integrity validation.

PETSc 3.24.4 and its AMG implementations are **external comparator runs**.
HyBIT's probe code neither links to them nor depends on them.

## A2 — strength graph / two-hop proxy (3/3 exact PASS)

Shared normalized strength graph, threshold 0.25; all CSR/ABTM two-hop
candidate patterns `exact_equal=true`:

| Case | Strong edges | Two-hop nnz | CSR ms | ABTM ms | ABTM vs CSR | CSR staging + ABTM conversion ms |
|---|---:|---:|---:|---:|---:|---:|
| island_shell | 81,088 | 179,220 | 2.061053 | 2.227068 | +8.1% slower | 1.495198 |
| multi_patch | 81,516 | 181,438 | 1.768415 | 2.241998 | +26.8% slower | 1.495798 |
| rotated_anisotropy | 123,266 | 388,244 | 3.654784 | 3.140171 | 14.1% faster | 1.696403 |

A2 **does not** compute Galerkin `P^T A P` or measure any solve. The one
positive ABTM kernel case excludes conversion/setup and cannot establish a
production win by itself.

## A3 — one-hot contiguous Galerkin structural candidate (9/9 PASS)

The same one-hot tentative transfer `P` is used by every arm. The Boolean
candidate structure of `P^T A P` matches exactly between CSR-sort, optimized
CSR-stamp, and ABTM-OR. An independently accumulated sparse numerical
`P^T A P` reference reports zero exact numeric cancellations in these nine
cases (`symbolic_nnz = numeric_nnz`). A deliberately cancelling synthetic
unit test also passed, so zero is not assumed universally.

| Case | Aggregate nodes | Coarse n | Coarse candidate nnz | CSR sort ms | CSR stamp ms | ABTM OR ms | ABTM vs CSR-stamp |
|---|---:|---:|---:|---:|---:|---:|---:|
| island_shell | 4 | 5184 | 25560 | 0.402730 | 0.262567 | 0.410591 | +56.4% |
| island_shell | 16 | 1296 | 6174 | 0.309708 | 0.089142 | 0.135183 | +51.6% |
| island_shell | 40 | 519 | 3559 | 0.355629 | 0.063491 | 0.094592 | +49.0% |
| multi_patch | 4 | 5184 | 25560 | 0.404700 | 0.264787 | 0.377770 | +42.7% |
| multi_patch | 16 | 1296 | 6174 | 0.326198 | 0.091853 | 0.137304 | +49.5% |
| multi_patch | 40 | 519 | 3559 | 0.291597 | 0.063761 | 0.094052 | +47.5% |
| rotated_anisotropy | 4 | 5184 | 35570 | 0.516073 | 0.245826 | 0.330388 | +34.4% |
| rotated_anisotropy | 16 | 1296 | 8462 | 0.330458 | 0.099303 | 0.131794 | +32.7% |
| rotated_anisotropy | 40 | 519 | 3559 | 0.262656 | 0.063411 | 0.092742 | +46.3% |

- Additional ABTM temporary-CSR preparation plus conversion cost is
  approximately **0.947–1.432 ms per setup**, not included in either kernel.
- `csr_adjacency_bytes` is a collection of `Vec` capacities whereas
  `abtm_metadata_bytes` omits its construction-stage CSR; these numbers are
  **not directly comparable peak RSS**. Recorded whole-process peak RSS is
  6,704–19,292 KiB over the separate A2/A3 processes, not a backend-isolated
  memory measurement.
- The island_shell and multi_patch A3 structure hashes are identical at
  each aggregate size despite different numerical coefficients; this
  benchmark does **not** test convergence robustness or aggregation quality.

## Evidence decision

**A2 accepted for exact symbolic equivalence, no universal ABTM speedup.**
**A3 accepted for exact structural and sparse-numeric reference equivalence;
retain optimized CSR-stamp for the tested one-hot Galerkin symbolic path.**
ABTM remains eligible for different graph-reach/occupancy/Halo workloads,
subject to a new cost-inclusive comparison. No production routing change.

## Next convergence objective (C1-H12-A4)

The externally observed PETSc BoomerAMG iterations on these three C1-G
matrices were 7/7/14 at true relative tolerance roughly 1e-10. They are
**targets for comparison**, not HyBIT-native performance. Proceed with the
existing native `TwoLevelBlockJacobiPreconditioner` over contiguous/graph/
strong-graph aggregation, piecewise-constant/Jacobi-smoothed basis, and an
independent Jacobi-PCG control. Keep exact manufactured RHS (`b=A*ones`),
600 iterations, verified relative residual and error to `ones` visible.

This is a **two-level preconditioner pilot**, not a completed multilevel AMG.
Do not claim rescue, a speedup, or new solver capabilities until owner WSL
numerical results and source hash checks pass.
