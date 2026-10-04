# HyBIT documentation

HyBIT 0.7.0 is the current published release. Documents describing
`develop/0.8.0` are development documentation and are not claims about the
immutable `v0.7.0` release.
Rendered development documentation: <https://michioga.github.io/hybit/>

The site is built with mdBook from the Markdown files in this directory.
During the 0.8 development cycle, GitHub Pages deploys from
`develop/0.8.0`; the deployment branch should move to `main` when the 0.8
release documentation is frozen.

## Start here

For library users:

1. [User guide](USER_GUIDE.md) — choose a problem class, configure a solver,
   prepare reusable state, solve, and validate the result.
2. [Solver selection and Krylov theory](SOLVER_SELECTION.md) — understand why
   SPD uses PCG, GeneralSquare uses FGMRES, how preconditioning changes the
   mathematical contract, and what MINRES/LSQR-class future routes require.
3. [Prepared execution](PREPARED_API.md) — understand
   `analyze -> prepare -> solve-many`, reuse boundaries, and memory ownership.
4. [General-square systems](GENERAL_SQUARE.md) — FGMRES, restart policies,
   Jacobi versus ILU(0), pivot stabilization, and current limitations.
5. [Structural Auto API](STRUCTURAL_AUTO_API.md) — 3-D structural SPD systems
   with rigid-body coarse correction.

For numerical background:

- [Solver selection and Krylov theory](SOLVER_SELECTION.md) — matrix-class
  contracts, PCG/FGMRES mechanics, restart cost, compatibility, and breakdown.
- [Hybrid SPD mathematics](HYBRID_MATH.md) — selective local direct correction,
  overlap weighting, and resumable PCG logic.
- [Architecture](ARCHITECTURE.md) — crate boundaries, execution architecture,
  resident CPU work, FGMRES checkpoints, and internal policy flow.
- [ABTM topology algebra](ABTM_TOPOLOGY.md) — planned symbolic/topology layer,
  metadata-first numerical pruning, adaptive/block layouts, and validation
  sequence for CPU, GPU, and distributed execution.

For project status:

- [Development status](DEVELOPMENT_STATUS.md) — published release versus the
  current development branch.
- [Roadmap](ROADMAP.md) — completed and planned work.
- [FEM benchmark](FEM_BENCHMARK.md) and
  [benchmark README](https://github.com/michioga/hybit/blob/develop/0.8.0/benchmarks/README.md) — reproducible validation.
- [Publishing](PUBLISHING.md) — release discipline and immutable release tags.

## Documentation layers

| Layer | Primary question | Main documents |
| --- | --- | --- |
| User | How do I solve my system correctly? | `USER_GUIDE.md`, `PREPARED_API.md`, `GENERAL_SQUARE.md`, `STRUCTURAL_AUTO_API.md` |
| Numerical | What mathematical method is being applied, and why? | `SOLVER_SELECTION.md`, `GENERAL_SQUARE.md`, `HYBRID_MATH.md` |
| Architecture | How are reusable/execution and symbolic topology states organized? | `ARCHITECTURE.md`, `ABTM_TOPOLOGY.md` |
| Development | What is published, validated, experimental, or planned? | `DEVELOPMENT_STATUS.md`, `ROADMAP.md`, `CHANGELOG.md` |

Benchmark results are evidence for tested cases, not universal performance
claims.