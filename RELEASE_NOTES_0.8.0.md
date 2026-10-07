# HyBIT 0.8.0

HyBIT 0.8.0 extends the 0.7 SPD/structural baseline with a broader prepared
execution architecture, GeneralSquare support, ABTM topology/preparation work,
and explicit Rayon execution.

## Highlights

- Prepared `GeneralSquare` route using restarted FGMRES.
- Jacobi remains the GeneralSquare default; canonical ILU(0) is explicit
  opt-in, with an explicit missing-diagonal fallback policy.
- Reusable FGMRES workspaces and prepared solve-many state.
- ABTM G1-G7 topology, metadata-first, local-region, ILU(0), block, restricted
  operator, and prepared-Rayon validation.
- Explicit fixed-size B3/B6 dense block-CSR operator for suitable workloads.
- Explicit prepared `A*M` and graph-local `R*A*R` compact CSR operators.
- Full-Rayon and task-limited prepared CSR execution APIs.
- C/C++/Fortran thread-count interoperability with OpenMP-oriented hosts.
- Rust 1.73 MSRV retained.

## GeneralSquare

0.8 adds the prepared nonsymmetric/general-square route:

- restarted FGMRES with distinct `V` and `Z` bases;
- reusable workspace capacity across RHS vectors;
- fixed, escalating, and budget-aware restart control;
- canonical ILU(0);
- selective row-relative factor-pivot stabilization;
- explicit Natural/RCM ordering choices;
- explicit `Ilu0Fallback` for matrices outside strict ILU(0) applicability.

The F1-F7 corpus includes examples where RCM/ILU(0) is strongly beneficial and
examples where it is neutral or regressive. HyBIT therefore does not promote a
hidden universal ordering or Jacobi-to-ILU selector in 0.8.

## ABTM G1-G7

The ABTM work establishes a clearer architectural role.

ABTM is retained primarily as a topology/symbolic/preparation layer:

```text
topology / Boolean algebra
        |
        +--> pruning and structural intersection
        +--> region growth / overlap
        +--> local extraction
        +--> symbolic/numeric preparation
        |
        v
target-specific numerical representation
```

Ordinary scalar ABTM SpMV did not beat CSR on the development corpus and is not
promoted as a universal replacement.

For repeated restricted work, G6 validated compact prepared CSR for fixed
column restrictions and graph-local restrictions. G7 then validated explicit
Rayon execution of those prepared operators.

No hardware-specific automatic prepared-operator nnz selector is embedded in
the 0.8 API.

## Rayon and OpenMP-oriented host applications

The repository C ABI exposes:

```c
hybit_set_num_threads(uint32_t threads);
hybit_num_threads(void);
```

Before the first solver is created, thread-count precedence is:

```text
1. explicit hybit_set_num_threads(n)
2. RAYON_NUM_THREADS
3. OMP_NUM_THREADS
4. Rayon default
```

C/C++ OpenMP builds can use `hybit_sync_openmp_threads()` and the C++ wrapper
can use `hybit::sync_openmp_threads()`.

Fortran callers can pass `omp_get_max_threads()` through
`hybit_set_num_threads()`.

HyBIT does not directly link a second OpenMP runtime merely to query thread
settings. See `docs/THREADING.md` for process-wide pool lifetime, MPI/OpenMP
embedding, affinity, and oversubscription guidance.

## Important threading operational rule

Do not call one parallel HyBIT solve from every thread of an active OpenMP team.
Use one controlling host thread for a solve and let HyBIT perform its internal
Rayon work. Otherwise nested OpenMP x Rayon execution can oversubscribe the
machine.

Rayon's global pool is process-wide and cannot be resized after initialization.
Configure it before the first solver or Rayon query.

## C/C++/Fortran

`hybit-ffi` remains repository-only and is not published as a Rust crate.
The release gate builds the DLL/shared-library boundary and runtime-tests C,
C++, and Fortran consumers.

0.8 adds thread-count functions to the stable C ABI and matching C++/Fortran
bindings.

## crates.io

Published crates:

1. `hybit-core`
2. `hybit-matrix`
3. `hybit-krylov`
4. `hybit-precond`
5. `hybit-auto`
6. `hybit`

The facade crate is installed with:

```text
cargo add hybit@0.8.0
```

## Compatibility and scope

- Rust MSRV: 1.73.
- Real `f64` arithmetic.
- SPD automatic route: PCG/Hybrid/structural.
- GeneralSquare route: prepared FGMRES.
- Symmetric-indefinite systems are recognized but MINRES is not yet routed.
- No distributed-memory solver in 0.8.
- No production GPU backend in 0.8.
- Prepared contexts should not be mutated concurrently by multiple host
  threads; internal Rayon parallelism remains supported.
- Numerical and performance evidence is workload-specific. Validate residuals
  and physical results independently for engineering use.

## Next development direction

The post-0.8 priority is distributed topology/partition/halo work for MPI-hosted
execution. GPU/CubeCL work remains planned but is deferred until the CubeCL API
is sufficiently stable for HyBIT's production boundary.
