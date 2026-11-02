# HyBIT 0.10.0 — FEM-host-neutral integration, FrontISTR-first qualification

**Decision (2026-10-30):** HyBIT is a general-purpose sparse linear solver for many FEM applications. **FrontISTR is the first integration and numerical-validation host, not the sole supported target or a dependency of the solver.**

## Scope boundaries

- **Core**: CSR32 / linear operator, RHS, constraints or field/DOF metadata where necessary, mathematical matrix-class contract, solve / analyze-prepare-solve-many, residual/status, resource budget. No FrontISTR modules, control-file enums, mesh structs, or filenames in `hybit-*` library APIs.
- **Host adapter**: imports a host's matrix, RHS, ownership/DOF map and, where possible, trusted reference solution. FrontISTR is currently supported **only** through serial Matrix Market and RHS dump evidence. Future FrontISTR C/Fortran in-process binding is a separate integration milestone.
- **Other FEM hosts**: can use the same generic Matrix Market + RHS tool and generic FGMRES/PCG APIs. Future adapters should remain out of solver core.
- **Future**: Fortran/C ABI adapter for FrontISTR, real-host solve in nonlinear iterations, MPI rank ownership, and other FEM engines (FrontISTR-first, not FrontISTR-only). These are not implemented by this patch.

## Verified FrontISTR source behavior relevant to ingestion

`hecmw1/src/solver/matrix/hecmw_matrix_dump.f90` emits `dump_matrix_<call>_<rank>.mm` and `.rhs`; MM header is `%%MatrixMarket matrix coordinate real general`, containing explicit lower/diagonal/upper blocks; `.rhs` stores scalar numeric entries. It uses `hecMAT%NP * NDOF` for dimensions (rather than promising a global assembled matrix). **A rank-0 file from an MPI run is not automatically the complete global system.** A simple rank-local concatenation is invalid without owner/ghost/DOF mappings and a distributed contract.

Perform initial **single-process** dumps of fixed, known boundary-condition and solver-state snapshots. Do not assume symmetry/SPD from a positive diagonal, a symmetric header, or low true residual. For SPD PCG comparisons, independently validate matrix class. For general square matrices use the existing FGMRES high-level path (Jacobi default, ILU(0) opt-in). Neither method guarantees convergence.

## C1-H1: no core change / no solver policy change

`crates/hybit/examples/fem_fgmres_bench.rs` accepts generic `--matrix FILE --rhs FILE` and optional `--xref FILE` with `--precond jacobi|ilu0|ilu0-fallback|ilu0-abtm`; it uses `MatrixProblemClass::GeneralSquare` and prepared FGMRES. It reports status, original-CSR verified residual, iterations, setup and solve costs, optional reference forward error. Unsuccessful `MaxIterations` is not a shell-level exception; consumers must check the numerical result, not merely the program exit code.

`benchmarks/scripts/fem_linear_system_probe.py` is FEM-neutral for `--matrix` + `--rhs`. An optional `--frontistr-dump-dir DIR` adapter pairs a numbered FrontISTR `.mm` and `.rhs`. The runner **never** combines MPI rank files; FrontISTR execution requires the operator to acknowledge `--frontistr-serial-confirmed`. An audit checks finite values, indices, counts, dimensions and SHA-256 but **does not certify SPD, solution accuracy, or correct host-to-dump provenance**.

For the default general-square comparison, invoke FGMRES + Jacobi and FGMRES + ILU(0) with identical RHS, tolerance, iteration budget, initial guess and execution policy. ILU(0) numerical failure must remain a visible result, not a silent substitution. With independently established SPD, optionally include observe-only Jacobi-PCG and opt-in Hybrid-PCG. Do not infer causality from a single elapsed-time run.

## Example: generic FEM host

```bash
cd ~/hybit
cargo build --release --locked -p hybit --example fem_fgmres_bench
python3 -B benchmarks/scripts/fem_linear_system_probe.py \
  --matrix /path/to/K.mtx --rhs /path/to/b.txt \
  --run --hybit-dir ~/hybit --out-dir ~/fem_probe
```

## Example: first host FrontISTR

Prepare a **single MPI-rank / serial** FrontISTR run with MM dump and matching RHS. Confirm physically that no distributed ranks or ghosts were left unassembled.

```bash
cd ~/hybit
python3 -B benchmarks/scripts/fem_linear_system_probe.py \
  --frontistr-dump-dir /path/to/frontistr/dump --call 1 \
  --frontistr-serial-confirmed \
  --run --hybit-dir ~/hybit --out-dir ~/frontistr_probe
```

If the dump contains multiple calls, select the correct call; do not arbitrarily choose the first. If MPI rank files are present, refuse to treat the collection as one global system. Preserve the original FrontISTR model, solver control, boundary conditions, rank/thread counts, and physical load case with the snapshot. `--declared-spd` is explicit and requires external evidence.

## Acceptance criteria before integrating a FEM host

1. Matrix and RHS have matching dimensions, finite values, independent SHA-256 records, reproducible case definition, and clear DOF ownership and constraints.
2. For the correct mathematical class, verify original operator residual and status. If reference solution is available, measure forward error and physical quantities. A small residual alone may hide substantial forward error for ill-conditioned matrices.
3. Classify cases as baseline failed / Hybrid succeeded (**rescue**), both succeeded, both failed, or Hybrid regression. Record `MaxIterations`, breakdown, unsupported class, factor budget and setup errors distinctly.
4. Measure independent warm/cold timings and peak RSS only after correctness and rescue classification. Preserve negative results.
5. For MPI, first implement and validate real owner/ghost global mapping; never load a rank-local `.mm` into a serial solver as if it represented the global matrix.

**Checkpoint base**: `develop/0.10.0` commit `949f00b` plus user-validated, uncommitted C1-G2 `--xref` benchmark instrumentation. This is documentation and opt-in benchmarks only. No production HyBIT solver/API/FFI changes and no automatic FrontISTR-internal integration.
