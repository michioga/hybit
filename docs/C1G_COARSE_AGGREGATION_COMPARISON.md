# C1-G2 follow-up: coarse aggregation and forward accuracy

Date: 2026-11-06 (JST). Experimental evidence on `develop/0.10.0`,
commit `72677b6`. This is a **C1-G2 aggregation follow-up**, not the
C1-G3 controller-robustness milestone described in
`docs/C1G_ROBUSTNESS_CORPUS.md`.

## Protocol

Constructed SPD P1 diffusion matrices, 20,736 DOFs; n=144
`island_shell`, `multi_patch`, `rotated_anisotropy`.
All runs use original-CSR true-residual verification and a manufactured
reference solution; fixed maximum 600 iterations, target 1e-10,
`--backend csr`, one thread, zero initial guess, `--monitor-only` to
disable local Cholesky, `--hybrid-coarse --coarse-dofs 1 --coarse-target 512`,
piecewise coarse basis. The only deliberately varied setting is
`--coarse-aggregation contiguous|graph|strong-graph`.
All input SHA-256 gates passed; source and HEAD checks matched.

## Full single-run results

| Case | Aggregation | Status / iterations | True relative residual | Relative x error | Coarse dim | Factor MiB | Setup ms | Total wall ms | Peak RSS KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| island_shell | contiguous | Converged / 342 | 9.874271e-11 | **2.382841e-2** | 506 | 2.127 | 16.692 | 166.432 | 11392 |
| island_shell | graph | Converged / 99 | 8.807643e-11 | 4.415574e-7 | 508 | 2.301 | 21.141 | 66.691 | 12324 |
| island_shell | strong-graph | Converged / 100 | 9.714946e-11 | 8.305859e-7 | 509 | 2.309 | 23.432 | 66.868 | 13864 |
| multi_patch | contiguous | Converged / 345 | 9.899120e-11 | 1.573025e-9 | 506 | 2.127 | 16.032 | 165.247 | 11492 |
| multi_patch | graph | Converged / 118 | 9.910340e-11 | 1.401397e-8 | 508 | 2.301 | 17.558 | 66.395 | 12400 |
| multi_patch | strong-graph | Converged / 118 | 8.719585e-11 | 1.039530e-8 | 509 | 2.309 | 22.456 | 72.017 | 13976 |
| rotated_anisotropy | contiguous | **MaxIterations / 600** | 4.629944e-9 | 6.906424e-8 | 506 | 2.127 | 16.552 | 286.630 | 12376 |
| rotated_anisotropy | graph | Converged / 463 | 9.807269e-11 | 1.150465e-9 | 508 | 2.301 | 17.450 | 205.452 | 12688 |
| rotated_anisotropy | strong-graph | Converged / 445 | 9.950162e-11 | 8.690078e-10 | 507 | 2.293 | 21.842 | 203.871 | 15348 |

**Process exit status 0 does not mean solver convergence.** The
rotated-anisotropy contiguous case correctly reports MaxIterations and
fails the target residual; it remains in the comparison denominator.

### Interpretation

- At essentially the same target coarse size, both graph-based strategies
  outperform contiguous in iteration count on the three observed cases.
  Contiguous is not guaranteed to solve within the 600-iteration budget.
- Normal graph is 99 vs strong-graph 100 iterations on island_shell and
  both are 118 on multi_patch. Strong graph improves rotated anisotropy
  from 463 to 445 iterations, but raises RSS and setup cost on this sample.
- The actual coarse dimensions differ slightly (506-509), and persistent
  coarse factor bytes differ. Those must be reported with performance.
- Timings and RSS are **single process samples** with no warmup/repeated
  confidence estimate. Do not assert a universal speedup or rank variants
  from millisecond-level noise.
- At target 1e-10, island_shell contiguous has a 2.38% relative forward
  error in spite of a verified residual below 1e-10. At target 1e-12 and
  2400-step cap, it converges in 648 iterations with forward error
  2.862521e-9; graph reaches 1.707861e-8 in 127 steps and strong-graph
  2.164681e-8 in 125. Residual accuracy and solution accuracy are
  different numerical contracts on ill-conditioned matrices.

## Relation to ABTM

These comparisons test the **mathematical aggregation choices** using the
existing software implementation. They do **not** isolate the execution
cost or memory of ABTM bitmap operations. A valid ABTM performance claim
requires equivalent final partitions/coarse matrices on CSR and ABTM
symbolic implementations, followed by setup, conversion peak, and
end-to-end measurements with numerical equivalence checks.

## Evidence provenance

- `~/c1h11_evidence/c1g_g3_aggregation_rl4XAT`:
  nine `{case}_{aggregation}.log/.time` pairs, Git HEAD and source hash.
- `~/c1h11_evidence/c1g_g2_forward_accuracy_g4vubT`:
  independent stricter-tolerance check, with matrix/RHS/xref SHA-256.
- `~/mtx/c1g_g2_hard_z32Jkd/input_sha256.txt`:
  common original input checksums for all n=144 runs.

No production algorithm, autonomy policy or public API is changed here.
Retain `graph` as an **experimental comparison baseline**, not as a
universally superior default. Follow with held-out numerical cases,
several RHS, repeated timing trials and a separate ABTM-vs-CSR symbolic
construction experiment.
