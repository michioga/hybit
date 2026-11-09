# C1-H12-A9: right-hand-side diversity of native A8 multilevel AMG

Status: **experimental; WSL owner verification pending**. No production AMG,
public API, default PCG, or A8 source is modified by this add-on.

## Why this follow-up is required

A8 reported 18/18 certified convergences and 1,042 total iterations for the
multilevel arm, versus 17/18 and 5,047 for the frozen two-level control.
These are genuine results for the particular A8 input, but **all nine A8
Jacobi-smoothed multilevel cases took just one PCG iteration**. This can arise
from the A8 manufactured test `x* = ones`, `b = A x*`: piecewise-constant
aggregation has `T ones = ones`, Jacobi-smoothed interpolation has
`P = (I - omega D^{-1} A) T`, and the symmetric V-cycle pre-smoother shares
the same Jacobi factor. The favorable manufactured solution therefore does not
independently establish general-purpose fast convergence.

A9 holds A8 aggregation, Galerkin/ABTM kernels, symmetric V-cycle,
coarsest-factor budget, certified PCG, tolerance, and iteration limit fixed.
It changes **only the right-hand side**, independently for each process:

| RHS mode | Construction | Exact solution available? |
| --- | --- | --- |
| `ones` | `x_i = 1`, `b = A x` | Yes; exact A8 replay |
| `index_smooth` | `x_i = 0.5 + 0.3 sin(2 pi i/n) + 0.2 cos(6 pi i/n)`, `b=A x` | Yes |
| `seeded_random` | fixed-seed xorshift64 values in `[-1,1)`, `b=A x` | Yes |
| `point_load` | `b[n/2]=1`, other loads zero | No; certified true residual only |

`index_smooth` is smooth with respect to the **matrix index**, not physical FEM
coordinates, and `point_load` is a single algebraic DOF load. Neither should
be described as a full set of physical forcing scenarios.

## Experiment matrix and reproducibility

The original three 20,736-DOF FEM matrices remain unchanged:
`island_shell`, `multi_patch`, and `rotated_anisotropy`. Three coarse arms were
selected from the A8 observations:

- `graph_piecewise`
- `strong_graph_piecewise`
- `strong_graph_smoothed` (including the A8 one-iteration anomaly)

Run exactly **3 FEM × 3 arms × 4 RHS × 2 preconditioners = 72 independent
processes** (one WSL process per cell). The two policies are the A8 additive
two-level control and the independent native multilevel symmetric V-cycle.
Parameters: `tol=1e-10`, `max_iterations=600`, `RAYON_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, 4-node AMG aggregation, at most 8 levels, 512 coarsest
nodes, F64 transfers. The same exact solver and preconditioner work is run on
both policies for each RHS.

Each log contains one `C1H12A9|...` record, including `rhs_mode`,
`rhs_hash` (16 hexadecimal digits, FNV-1a of every binary64 RHS coefficient),
`rhs_norm`, true residual measured from original CSR, and the original A8
performance/cost counters. In `point_load` mode only, `relative_x_error=n/a`
because no manufactured exact solution is assumed.

The gate checks all 72 records (including honest nonconvergences), input
identity, matched RHS fingerprints across six process configurations per
matrix/mode, distinct fingerprints for different modes, level structure,
SpMV/call accounting, process outcomes, true original-CSR residual, and the
`ones` replay against frozen A8 on every selected arm. Both HEAD and matrix
SHA lists must match the prior run. The fresh CSV and all raw `.log`/`.time`
files remain outside the Git repository.

## WSL owner commands

After applying the companion package (first dry-run, then apply), execute:

```bash
cd ~/hybit
BASE="$HOME/c1h11_evidence/c1h12_a8_multilevel_xsBeeb"

bash benchmarks/scripts/C1H12_A9_rhs_diversity.sh --dry-run "$BASE"
bash benchmarks/scripts/C1H12_A9_rhs_diversity.sh --run "$BASE"
```

The shell script runs Python negative tests, Rust tests, Clippy `-D warnings`,
release build, 72 processes, and evidence gates. It reports the new raw
directory and `rhs_diversity_comparison.csv` even if the evidence gate rejects
one or more cells. Preserve this raw evidence for review.

## Interpretation and acceptance

**Numerical gate PASS does not automatically mean a useful AMG speedup.**
Review each non-`ones` mode separately, especially whether smoothed AMG still
converges in a single iteration, the StrongGraph piecewise improvement, any
`MaxIterations` or `Breakdown`, and setup/solve/total cost. This is one
process measurement per configuration; repeated timings and PETSc/hypre
comparison remain future work.

Do not commit/push the experiment or promote it to the production solver
until owner numerical review and further independent SPD corpus coverage.
