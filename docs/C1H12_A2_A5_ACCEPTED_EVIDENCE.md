# C1-H12-A2–A5 WSL evidence checkpoint (2026-11-09)

Acceptance: **54/54 raw cases validated** (A2 3, A3 9, A4 21, A5 21).
Original study source: `develop/0.10.0` full HEAD `4593c2010722876b2b66bda83b78b62827ace77a`.
Owner environment: Ubuntu 26.04 WSL2, Rust +1.99.1, one process/thread.
Evidence accepted by `benchmarks/scripts/C1H12_A2_A5_evidence_accept.py`.

## Input provenance

Identical `HEAD.txt` and `matrix_sha256.txt` across all four raw runs;
matrix files were rehashed at acceptance. Matrix manifest SHA-256: `b6e5e10925ed287c90f7072b00137ef2feb4eed609e5ffeec7a4536c63163630`.

- island_shell: `2d3f2834e3737610357525a3fdae9abaebb9ee21969d1800b0bd9c15414be45f`
- multi_patch: `06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d`
- rotated_anisotropy: `03abf3e7d98f840dbb4a2ed09282d7abfbf28c928cff18837f0087af4fb51c42`

## Evidence locations (raw files remain owner-side)

- A2: `/home/michioga/c1h11_evidence/c1h12_a2_symbolic_FH3k8I`
- A3: `/home/michioga/c1h11_evidence/c1h12_a3_galerkin_3Qs0BC`
- A4: `/home/michioga/c1h11_evidence/c1h12_a4_native_OPAkkT`
- A5: `/home/michioga/c1h11_evidence/c1h12_a5_certified_QGyKD6`

The repository retains per-record summaries and SHA-256 hashes of all 108 `.log`/`.time` files.
Those hashes are references to owner-side raw evidence, **not** the raw logs themselves.

## Symbolic structure (A2/A3)

- A2: three exact CSR/ABTM two-hop support comparisons (not a Galerkin operator).
- A3: nine exact one-hot piecewise-constant `P^T A P` Boolean candidate comparisons;
  independently assembled numeric reference. ABTM construction/conversion cost is separate.
- Do not generalize isolated kernel wins into AMG or whole-solve speedups.

## Native two-level PCG (A4/A5)

- A4 numerical acceptance was **15/21**, not 21/21: six `multi_patch`
  coarse arms falsely declared recursive `Converged` with true residual above 1e-10.
- A5: **21/21 independent true-residual and provenance gate PASS**;
  18 true `Converged`, 3 honest `MaxIterations`, no Breakdown.
- All six `multi_patch` coarse arms were repaired using opt-in certified restarts
  with the unchanged total 600-iteration budget; native PCG defaults are untouched.
- No multilevel AMG, PETSc/hypre linkage, automatic routing, or whole-solve speedup claim.

| multi_patch arm | A4 iterations | A4 true rel | A5 iterations | A5 true rel | certified restarts |
|---|---:|---:|---:|---:|---:|
| contiguous_piecewise | 342 | 7.160352e-10 | 432 | 7.126238e-11 | 5 |
| contiguous_smoothed | 213 | 4.937748e-10 | 339 | 9.834315e-11 | 7 |
| graph_piecewise | 116 | 3.925232e-10 | 126 | 5.473644e-11 | 3 |
| graph_smoothed | 94 | 4.087384e-10 | 458 | 7.536696e-11 | 34 |
| jacobi | 600 | 1.152375e-09 | 600 | 1.152375e-09 | 0 |
| strong_graph_piecewise | 128 | 3.423971e-10 | 189 | 9.356816e-11 | 7 |
| strong_graph_smoothed | 94 | 4.096047e-10 | 143 | 9.127270e-11 | 10 |

## Machine-readable accepted artifacts

- `benchmarks/evidence/C1H12_A2_A5_symbolic.csv`: 12 isolated symbolic cases.
- `benchmarks/evidence/C1H12_A2_A5_native_pcg.csv`: 42 A4/A5 arms.
- `benchmarks/evidence/C1H12_A2_A5_raw_sha256.csv`: 108 raw `.log`/`.time` hashes.

## Next prioritized experiment

A6: instrument periodic true-residual checks and recurrence drift in an opt-in
PCG experiment; evaluate early safe restarts versus A5 under matched conditions.
Preserve SPD invariants, original `b-A*x` certification, 600 global iterations
and independent A5 negative controls. Never mutate production PCG by default.
