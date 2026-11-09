# C1-H12-A3 — native symbolic Galerkin CSR / ABTM comparison

Date: 2026-11-09 (JST). Baseline: `develop/0.10.0` commit `4593c20`,
with **uncommitted C1-H12-A2 comparison patch already applied**.

## Hypothesis and scope

Measure whether ABTM word-OR accelerates actual *structural* Galerkin
coarsening with an explicitly fixed tentative interpolation `P`, versus two
CSR implementations. Keep PETSc/hypre external comparison-only; use native
HyBIT Rust and preserve existing PCG/preconditioner/public API unchanged.

The experiment uses the already implemented **Contiguous / PiecewiseConstant**
coarse basis with `dofs_per_node=1`:

`P[i, floor(i/aggregate_nodes)] = 1`, all other entries zero.

For this one-hot `P`, Boolean symbolic support of `E = P^T A P` is the union of
coarse-column neighbors among fine rows of each aggregate. This **is** the
Galerkin *candidate structural pattern* for this `P`; exact numerical zero
cancellations may make the final `E` sparser. It is **not** the smoothed
Jacobi transfer, a complete AMG hierarchy, or a PCG speedup. Other
aggregate policies or the numeric values of `P` would change the pattern.

The same canonical fine-row `A P` Boolean pattern is built **once** and given
to three arms:

1. **CSR-sort**: concatenate adjacent row lists, sort, deduplicate.
2. **CSR-stamp**: use a direct-address stamp array to remove duplicates,
   then sort only distinct touched columns.
3. **ABTM-OR**: convert that same structural `A P` into ABTM words, OR masks
   with word stamps and enumerate set bits.

The CSR-stamp arm is important: comparison against sort/dedup alone may
overstate the benefit of ABTM. ABTM pays `graph_csr_prepare_ms` and
`abtm_conversion_ms` in addition to the shared `graph_build_ms`.

## Correctness and numerical audit

- Every candidate row must exactly match between all three variants.
- Independently accumulate a **sparse numeric** `Z^T A Z` reference using
  finite f64 values and assert all its structural keys match the Boolean
  candidates. Count stored *exact* zero coarse values as
  `exact_zero_candidates` to expose cancellation without dropping candidates.
- Rust tests cover word boundaries, isolated/zero connections,
  true off-diagonal cancellation, invalid aggregate sizes and rectangular
  input. Positive definiteness is **not** asserted or inferred from success.

## Experimental matrix sweep

Three difficult C1-G P1 FEM cases, each n=20,736:

- `island_shell`
- `multi_patch`
- `rotated_anisotropy`

Aggregate sizes: `4, 16, 40` fine DOF per group, for 9 cases total.
One warmup plus 5 timed repeats per kernel, median reported. Every run
records input SHA-256, HEAD, stdout, GNU time wall and peak RSS; graph
construction, CSR staging, ABTM conversion, numerical audit and symbolic
kernels are measured independently. The kernel numbers **exclude** matrix
read, setup and reference numeric assembly. All storage byte figures are
component estimates and **not** interchangeable whole-process peak memory.

Use:

```bash
cd ~/hybit
bash benchmarks/scripts/C1H12_A3_galerkin_symbolic.sh --dry-run
bash benchmarks/scripts/C1H12_A3_galerkin_symbolic.sh --run
```

The runner requires the uncommitted A2 and A3 patches atop exactly
`4593c20`, Rust 1.99.1 and `/usr/bin/time`. No files from prior evidence
runs are removed or overwritten. The output directory is
`~/c1h11_evidence/c1h12_a3_galerkin_XXXXXX/`.

## Assistant-side synthetic smoke (not the 3 FEM cases)

A constructed n=20,736 regular five-point Poisson matrix, 103,104 CSR nnz,
produced exact matching patterns and no numeric cancellations. Single-process
five-repeat medians, **not** user-WSL benchmark evidence:

| aggregate nodes | CSR sort | CSR stamp | ABTM OR |
|---:|---:|---:|---:|
| 4 | 0.576 ms | **0.353 ms** | 0.466 ms |
| 16 | 0.516 ms | **0.123 ms** | 0.225 ms |
| 40 | 0.384 ms | **0.078 ms** | 0.193 ms |

The ABTM candidate wins versus CSR-sort for larger aggregates in this one
smoke, **but loses versus CSR-stamp for all three sizes**. No positive
generalization or automatic backend selection is supported by this data.

## Decision rule, limitations and rollback

Accept A3 as a **diagnostic probe** only after user WSL 9/9 exact gates,
source integrity, formatting, tests and strict Clippy pass. Consider ABTM
promotion only with repeated whole-setup and memory advantage against the
*best* CSR arm on representative inputs, plus subsequent full numerical
coarse correction and true-residual checks. A single P does not capture
Jacobi-smoothed P, multiple AMG levels, arbitrary aggregate geometry or
numeric cancellation effects on actual factorization. Do not equate the
Boolean candidate pattern with the actual numerical nonzero values.

The new example and script are isolated and opt-in; rollback does not touch
production code. Do not commit/push until the owner reviews measured evidence.
