# HyBIT 0.9 H1-C3K8-D9C1: communication-aware ABTM boundary refinement experiment

**Base:** `develop/0.9.0` commit `4e589997bcd248c976519c4518503e3beae168ee` (D9-B).

## Why

D9-A thermal1 quality using the same A union A^T graph and HyBIT telemetry, 8 ranks:

| Partitioner | Cut NNZ | Unique ghosts | Directed peer relations |
|---|---:|---:|---:|
| ABTM Multilevel | 7318 | 3687 | 36 |
| METIS (WSL) | 3806 | 1910 | 14 |
| SCOTCH (WSL) | 3700 | 1857 | 14 |

The updated G8-A6 already implements multilevel coarsening, 3% maximum-load tolerance, greedy cut refinement and bounded pair swaps. Adding the same features again is not a meaningful contribution. D9-C1 instead tests an *incremental unique-ghost objective* without replacing production code.

## Experimental algorithm

1. Generate original ABTM Multilevel partition from the unmodified library.
2. For row rank `r` and column `c`, maintain `demand[r,c] = count of CSR nonzeros in rows owned by r that reference c`. Its ghost indicator is `(demand[r,c]>0 && r!=owner[c])`; summation over (r,c) **exactly matches** HyBIT's `communication_volume` for arbitrary directed CSR.
3. Construct column-to-row incidence once; compute one vertex's directed-cut and unique-ghost deltas from incident CSR arcs and `O(ranks * row_degree)` demand cells. The ownership change affects both a row's rank and a column's owner.
4. Perform bounded deterministic boundary sweeps, allowing a **small temporary uphill** weighted score (`4*ghost_delta+cut_delta <= 6`). Retain the best prefix meeting *both* nonincreasing ghost and cut conditions, and revert the suffix. This is FM-*inspired*, NOT a general global FM implementation.
5. Recompute the independent original HyBIT `partition_telemetry_assignment` after each proposed pass. Reject and roll back if directed cut, unique ghost volume, or directional peer relations worsen. The initial A6 load upper cap is enforced. The lower bound is never tightened beyond A6's existing minimum load.
6. Export deterministic labels for optional D9-B external-owner P2P PCG evaluation, *only if the reference gates pass*.

### Critical limitations

- The score coefficient `4` and uphill threshold `6` are **experimental**, not tuned by an out-of-sample objective. No claim of METIS/SCOTCH superiority is made before measurement.
- A6 produces a balanced but not necessarily minimum-load bounded initial assignment; preserve the existing minimum rather than demanding exact equal-DOF or a new lower bound.
- The first probe optimizes unique ghosts and CSR cut, preserving rather than directly optimizing the peer relations. More advanced incremental peer/overlap objectives belong in D9-C2.
- No production `multilevel.rs`, `partition.rs`, D7/D9-B solver, MPI routing, or numerical kernel modification.
- An independently run baseline D9-A METIS/SCOTCH quality CSV is required; its per-partition label files are not edited.
- Windows MSVC is used for the Rust executable; WSL is not required in this stage. No MPI calls are made by the quality probe.
- A successful trial is a *candidate*. Later D9-C2 verifies the resulting labels in identical Prepared P2P PCG ABBA against METIS/SCOTCH, including end-to-end setup cost.
- No performance was measured on Windows before handing over this package. The apply gate runs native release compilation and tests on the user machine.

## Apply and experiment

Use `APPLY_HYBIT_D9C1.ps1` from the archive. Preflight first. Apply requires clean exact base HEAD, SHA256-pinned payload, unchanged SOURCE_SHA256 baseline (369/367), and rollback on failure. Successful Apply has 372 manifest paths / 370 hashed, **five** staged changes, no commit/push.

Run `RUN_HYBIT_D9C1_REFINEMENT.ps1` with `-Matrix D:\Work\thermal1.mtx -QualityCsv D:\Work\k8d9a_thermal1_fix1_quality_summary.csv -Ranks @(4,8) -Rounds 3 -Passes 2 -MaxMoves 12000 -Prefix k8d9c1_thermal1` and the repository and output paths. Results are written to a common CSV and per-round owner-label files. A rejected improvement is a valid zero-improvement result, not a quality success.
