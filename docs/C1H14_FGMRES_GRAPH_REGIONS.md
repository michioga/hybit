# C1-H14-B2: graph-connected residual capture on real GeneralSquare FEM

Status: **opt-in postsolve diagnostic; user WSL2 Rust gates PASS and two
real-FEM cases reproduced** (2026-11-05). GeneralSquare local-direct rescue
is **not implemented or demonstrated** by this diagnostic.
Baseline: user's pushed `develop/0.10.0` at `155ab59` (C1-H14-B1).

## Question / falsification

B1 ranked individual residual rows, but the top-k rows need not form a
connected local-direct region. Here we ask how much of the *same* true
post-solve residual can be covered by graph-connected regions of a bounded
size using existing ABTM structural topology.

Controls (all real FrontISTR `.mm` / physical `.rhs` snapshots):

- `11_contact_2beam`: 756 rows; B1 raw top 32 rows captured 70.06% of the
  358-row dominant-component residual energy; restart-30 FGMRES remained
  `MaxIterations` at true relative residual `1.318012e-8` after 360 steps.
- `03_hyperelastic_cylinder`: 1,887 rows; ABTM ILU(0) with restart 70
  converged at 104 steps (tolerance `1e-10`). This is a no-rescue control.
- `01_elastic_hinge`: 252,168 rows, 18,846,277 CSR nnz; B1 raw top 512
  covered just 20.19% of the 250,080-row dominant-component energy.

This is a *postsolve diagnostic only*. It neither proposes nor performs a
local LU/LDLᵀ factorization. In particular, GeneralSquare does **not**
permit assuming that a Cholesky correction or an unpivoted small LU would
be a valid rescue. No changes to public APIs, solver/preconditioner,
FGMRES iterations, problem class or true residual calculation.

## Algorithm

- Reuse the existing C1-H13 original-CSR true residual and weak-component
  partition, and C1-H14-B1's dominant component and raw / row-max-scaled
  residual magnitudes.
- Build the existing `AbtmDualTopology` for **both** `A` and `A^T` once
  after the timed solve. This handles one-sided nonsymmetric edges.
- A region starts at the highest-ranked *uncovered* row. Its next row is
  the highest-scored **frontier neighbor** of an already selected row,
  using deterministic smaller-index tie-breaks. ABTM `u64` masks perform
  eligibility and visited-set `AND-NOT` filtering. Each individual region
  is connected in the original `A ∪ A^T` structural graph.
- Test total row budgets **32/64/128/256/512** and **1/2/4** nonoverlapping
  regions. Each region gets `floor(k/count)` rows, with remainders assigned
  to early regions. An exhausted frontier can leave a region undersized;
  unused capacity is not reassigned. Zero residual creates no regions.
- Compare the actual selected set's squared-L2 residual fraction with the
  **unconstrained B1 top-k oracle** under the *same* metric and budget.
  Their ratio is a capture-efficiency diagnostic, **not** an optimality or
  convergence certificate. Separately record selected rows, seed DOFs,
  region sizes, inspected ABTM words and enqueued neighbor bits.
- Treat row-max scaling as a heuristic rather than dimensional
  preconditioning. If B1 declares it unavailable, skip only the scaled
  graph metric; always preserve the raw metric.

Output lines: `C1H14_GRAPH_SETUP`, `C1H14_GRAPH` (with `metric=raw` or
`metric=row_max_scaled`), optional `C1H14_GRAPH_SCALED` when scaled
diagnostics are unavailable, and `C1H14_GRAPH_DIAGNOSTICS`. Setup and diagnostic
timings are **outside** the existing timed solve wall; peak RSS includes
the diagnostic topology. The additional ABTM dual transpose/metadata has
nonzero temporary memory overhead and must be measured on the large case.

## WSL2 validation and measured results (2026-11-05)

The user applied the B2 FIX1 patch followed by the formatting FIX2 patch.
The final local tree passed `git diff --check`, strict SHA-256 source
integrity, `cargo +1.99.1 fmt --all -- --check`, all **11/11** release
example tests, release example Clippy with `-D warnings`, and release
example build. These gates were run on the user's WSL2 machine, not on the
assistant's scratch copy. The graph test did not alter solver results.

With `--tol 1e-10 --max-iters 360 --restart 30 --restart-policy fixed
--precond ilu0-abtm` and all three component/hotspot/graph diagnostics:

| Case | True relative residual | Solver status | ABTM dual metadata | Graph setup | Graph diagnostics | Process peak RSS |
|---|---:|---|---:|---:|---:|---:|
| `11_contact_2beam` (756 rows) | `1.318012e-8` | `MaxIterations` | 94,184 B | 0.265 ms | 0.930 ms | 8,208 KiB |
| `01_elastic_hinge` (252,168 rows) | `1.890115e-1` | `MaxIterations` | 66,994,172 B | 317.010 ms | 320.383 ms | 929,836 KiB |

Evidence directories (outside Git):

- `~/c1h11_evidence/c1h14_b2_graph_AmGOWb` is **not** a validated B2
  result: it was a failed pre-patch CLI invocation. Do not cite it as such.
- `~/c1h11_evidence/c1h14_b2_graph_KTf1ws` contains the successful
  `11_contact_2beam` graph probe, process timing and input hashes.
- `~/c1h11_evidence/c1h14_b2_hinge_ku5Ayq` contains the successful
  large-case `fgmres.log`, `fgmres.time`, input SHA-256 and baseline HEAD.

For the hinge dominant weak component (250,080 rows), raw squared-residual
energy capture was:

| Total selected rows | Unconstrained top-k oracle | One connected region | Two regions | Four regions |
|---:|---:|---:|---:|---:|
| 32 | 3.242% | 1.111% | 1.425% | 1.685% |
| 64 | 5.373% | 2.014% | 2.719% | 3.025% |
| 128 | 8.535% | 4.919% | 5.009% | 5.528% |
| 256 | 13.199% | 8.722% | 7.950% | 9.481% |
| 512 | 20.191% | 15.559% | 13.338% | 14.793% |

On the small contact case, a single connected region of 64 or 128 rows
matched the corresponding raw top-k oracle (84.01% / 94.34% of the
dominant-component residual energy). On the large hinge case even the
*unconstrained* best 512 rows contain only 20.19%; the best observed
connected 512-row proposal contains 15.56%. Consequently, a small number
of fixed local patches is a plausible **test candidate** for the contact
case, whereas the hinge result motivates testing broader/coarse correction.
Neither observation proves numerical conditioning, successful local-LU
factorization or actual FGMRES convergence rescue. The number of graph
regions has a budget-dependent effect, not a universal winner.

The reported dual-topology size is metadata alone, not an independent
measurement of incremental peak RSS. Single process runs are not a
statistically robust timing or memory-speedup comparison. The third planned
no-rescue-control graph run (`03_hyperelastic_cylinder`) remains pending.

## Reproduction protocol

From a clean branch at `155ab59` (an unrelated untracked `__pycache__`
may remain), apply the delivered patch with `git apply --check`, followed
by `git apply`. Then run:

```bash
cd ~/hybit
git diff --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit --example fem_fgmres_bench
cargo +1.99.1 clippy --release --locked -p hybit --example fem_fgmres_bench -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example fem_fgmres_bench
```

Then run all three cases with `--component-diagnostics --hotspot-diagnostics
--graph-diagnostics`, preserving output, input SHA-256, exact HEAD, process
exit status, wall and peak RSS. Use `--restart 30 --max-iters 360` on
`11_contact_2beam` and `01_elastic_hinge`; use `--restart 70
--max-iters 360` on `03_hyperelastic_cylinder`. Common `--tol 1e-10`,
`--restart-policy fixed`, `--precond ilu0-abtm`.

## Acceptance and rollback

1. Original `status`, `iterations`, verified residual, and B1 `C1H14_TOPK`
   must match without the new diagnostics (within printed precision; exact
   equality expected on the same machine and run inputs).
2. Directed-edge, deterministic tie, disjoint region, exact-solution and
   capacity tests pass. Every reported selected set consists of at most
   the budgeted number of rows; within each region, its growth path must
   be connected. No duplicated selected row across regions.
3. Graph energy fraction is finite, in `[0,1]`, no greater than the
   top-k oracle apart from rounding; zero residual yields zero/no candidates.
4. Judge *observed graph capture*, build time and peak memory before any
   local-direct algorithm. A poor result means keep B2 diagnostic-only and
   explore stronger edge weighting / several local regions / coarse-space
   approaches, **not** enable automatic promotion.

Rollback: revert only the example's `--graph-diagnostics` functionality and
this experimental documentation, maintaining the already-committed B1
baseline. No core production routing or public API requires rollback.
