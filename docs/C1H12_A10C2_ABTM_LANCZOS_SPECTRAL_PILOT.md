# C1-H12-A10C2: ABTM adjacency, low-Ritz candidates, and coarse projection

Status: **standalone research experiment**, not production multilevel AMG.

## Scientific questions

1. Is `multi_patch` unusually close to a low-energy mode after Jacobi diagonal scaling?
2. Do the A8/A10B `graph_piecewise`, `strong_graph_piecewise`, and
   `strong_graph_smoothed` first-level coarse spaces represent identified
   low-energy test vectors differently?
3. Can bit-packed adjacency preserve the deterministic A8 graph/strong-graph
   aggregate construction across 64-column word boundaries?

No test below claims to measure the true condition number of the original
matrix, eigenvalues of the complete V-cycle preconditioned operator, an
unavoidable floating-point error floor, or ABTM-only acceleration.

## Independent spectral method

Let `A` be the original symmetric CSR matrix and `D = diag(A) > 0`. Use

`B = D^(-1/2) A D^(-1/2)`

with two deterministic starts: `sqrt(D) * ones` and a fixed-seed pseudo-random
vector. The first start's Rayleigh quotient is independently cross-checked
against frozen A10C1 `diag_scaled_rayleigh` for `ones`.

For each start compute **at most 48 Lanczos steps**, fully and twice
reorthogonalizing every new basis vector. Diagonalize the small symmetric
tridiagonal matrix using an independent, dependency-free Jacobi eigensolver.
Reconstruct the lowest and highest Ritz vectors, then independently calculate
their actual residuals `||Bv - theta*v||_2`. Confirm agreement with the
Lanczos recurrence residual `beta_next * |last_component|` and bound the
basis's loss of orthogonality.

**Interpretation warning:** the reported interval
`[theta_min - ||r||, theta_min + ||r||]` contains *some* eigenvalue of the
symmetric original scaled operator, but does not certify that it contains
its globally smallest eigenvalue. `ritz_low` / `ritz_high` are extrema of
the generated **Krylov subspace only**. Small residual improves confidence
in the associated eigenpair, not completeness of the spectrum.

## ABTM and first-level coarse-space method

Mirror A8's size-4 breadth-first aggregates and `strength_threshold = 0.25`.
Each row's symmetric-union graph adjacency is represented by `(u64_word,
u64_mask)`; neighbors are traversed using least-set-bit extraction and
`bits &= bits - 1`, with ascending word and bit order. The measured
`abtm_words` and `abtm_edges` report structural work; this is a standalone
ABTM-style mask implementation, **not** a benchmark of production ABTM.

Construct each first-level prolongation `P`:

- graph / strong_graph piecewise: `P[i, owner(i)] = 1`.
- strong_graph smoothed: `P = (I - omega*D^-1*A)*T`, where
  `omega = min(0.67, 1.6/max_i(sum_j |A_ij|/D_ii))`, as in A8.

For each coarse space and three fine vectors (`ones`, lowest Ritz vector
from the scaled-ones start, lowest Ritz vector from the random start),
measure the **D-orthogonal first-level projection**:

`min_c ||v - P*c||_D / ||v||_D`, where `||v||_D^2 = v^T D v`.

For smoothed `P`, use diagonal-preconditioned CG on `(P^T D P)c=P^T D v`;
require a separately re-evaluated relative normal residual ≤ 1e-9.
For piecewise `P`, use each aggregate's exact D-weighted average. Report
`captured_energy_d = 1 - projection_error_d^2` and actual solver iteration
count. A small projection error is necessary evidence of first-level
expressivity for the given Ritz candidate, not proof of multilevel V-cycle
convergence. A Ritz candidate with large `ritz_low_residual` is provisional.

## Frozen inputs and provenance

- Current committed repository HEAD: `60104c9`, branch `develop/0.10.0`.
- Frozen A10B evidence: original 720-record prepared hierarchy measurements,
  historical HEAD resolved from `1f3eaa5`.
- Frozen A10C1: 12 accepted independent precision records, HEAD resolved
  from `60104c9`; original CSV checksum captured in A10C2 output.
- Original FEM SHA-256 manifests must agree across A10B/A10C1 and the
  source corpus. All original matrices remain unchanged.
- No new Rust dependencies, no PCG re-execution, no modifications to
  production or ABI, no commit or push.

`spectral_ritz.csv` contains 3 matrices × 2 starts = 6 rows.
`coarse_projection.csv` contains 3 matrices × 3 arms × 3 probes = 27 rows.

Results must pass the A10C2 independent schema, range, provenance, frozen
status, and cross-file consistency gate before interpreting the numbers.
Stop rather than inferring a result from an uncertified Ritz recurrence or
projection solve. The research gate does not replace a fully independent
recomputation of all numerics or a production numerical acceptance gate.

## Next decision

Review `multi_patch` first. If the low-Ritz candidate has small residual and
a particular aggregate space has much larger D-projection error, investigate
its near-null-space enrichment. Otherwise inspect residual replacement and
finite precision PCG stagnation before changing production policies.
