# HyBIT documentation

HyBIT 0.7.0 is the current published release. Documents describing
`develop/0.8.0` are development documentation and are not claims about the
immutable `v0.7.0` release.

## Start here

For library users:

1. [User guide](USER_GUIDE.md) — choose a problem class, configure a solver,
   prepare reusable state, solve, and validate the result.
2. [Prepared execution](PREPARED_API.md) — understand
   `analyze -> prepare -> solve-many`, reuse boundaries, and memory ownership.
3. [General-square systems](GENERAL_SQUARE.md) — FGMRES, restart policies,
   Jacobi versus ILU(0), pivot stabilization, and current limitations.
4. [Structural Auto API](STRUCTURAL_AUTO_API.md) — 3-D structural SPD systems
   with rigid-body coarse correction.

For numerical background:

- [Hybrid SPD mathematics](HYBRID_MATH.md) — selective local direct correction,
  overlap weighting, and resumable PCG logic.
- [Architecture](ARCHITECTURE.md) — crate boundaries, execution architecture,
  resident CPU work, FGMRES checkpoints, and internal policy flow.

For project status:

- [Development status](DEVELOPMENT_STATUS.md) — published release versus the
  current development branch.
- [Roadmap](ROADMAP.md) — completed and planned work.
- [FEM benchmark](FEM_BENCHMARK.md) and
  [benchmark README](../benchmarks/README.md) — reproducible validation.
- [Publishing](PUBLISHING.md) — release discipline and immutable release tags.

## Documentation layers

| Layer | Primary question | Main documents |
| --- | --- | --- |
| User | How do I solve my system correctly? | `USER_GUIDE.md`, `PREPARED_API.md`, `GENERAL_SQUARE.md`, `STRUCTURAL_AUTO_API.md` |
| Numerical | What mathematical method is being applied? | `GENERAL_SQUARE.md`, `HYBRID_MATH.md` |
| Architecture | How is reusable/execution state organized? | `ARCHITECTURE.md` |
| Development | What is published, validated, experimental, or planned? | `DEVELOPMENT_STATUS.md`, `ROADMAP.md`, `CHANGELOG.md` |

Benchmark results are evidence for tested cases, not universal performance
claims.