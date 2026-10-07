# Threading and host-runtime interoperability

HyBIT uses Rayon for validated CPU-parallel execution. C, C++, and Fortran
applications frequently already use OpenMP, and FEM hosts such as FrontISTR may
combine MPI ranks with OpenMP threads. HyBIT 0.8 therefore treats thread-count
selection as an explicit process-level interoperability contract.

## Thread-count precedence

Before the first HyBIT solver is created, the effective Rayon worker count is
selected in this order:

```text
1. hybit_set_num_threads(n)
2. RAYON_NUM_THREADS
3. OMP_NUM_THREADS
4. Rayon default
```

For an OpenMP nesting list such as:

```text
OMP_NUM_THREADS=8,4,2
```

HyBIT uses the first value (`8`) for its Rayon worker pool.

`RAYON_NUM_THREADS` intentionally takes precedence over `OMP_NUM_THREADS`.
This permits a host application to keep its OpenMP team size and HyBIT's Rayon
pool size different when that is deliberate.

## OpenMP API settings

An environment-variable fallback cannot observe a later call such as:

```c
omp_set_num_threads(8);
```

For applications that select the OpenMP team size through the OpenMP API, call
HyBIT's thread API before creating the first solver:

```c
omp_set_num_threads(8);
int rc = hybit_set_num_threads((uint32_t)omp_get_max_threads());
```

When the C/C++ translation unit is compiled with OpenMP, `hybit.h` also exposes
the header-only convenience bridge:

```c
hybit_sync_openmp_threads();
```

and the C++ wrapper exposes:

```cpp
hybit::sync_openmp_threads();
```

Fortran can use the normal OpenMP module and the ISO_C_BINDING wrapper:

```fortran
use omp_lib
use hybit
use, intrinsic :: iso_c_binding

integer(c_int) :: rc

call omp_set_num_threads(8)
rc = hybit_set_num_threads(int(omp_get_max_threads(), c_int32_t))
```

`hybit_num_threads()` reports the effective Rayon worker count and is intended
for diagnostics. Because querying it may initialize Rayon's global pool, set
the desired policy first.

## Process-wide lifetime

Rayon's global worker pool is process-wide and cannot be resized after
initialization. HyBIT therefore does not pretend that thread count is a
solver-local property.

Repeated calls to `hybit_set_num_threads(n)` with the same value are accepted.
A request for a different value after initialization is rejected.

Configure the thread count before:

- `hybit_solver_create`;
- any call to `hybit_num_threads`;
- any other Rust component in the process that may initialize Rayon's global
  pool.

If another Rust library initializes Rayon first with a different size, HyBIT
cannot resize that pool.

## OpenMP runtime linkage

HyBIT does **not** link directly to `libgomp`, LLVM `libomp`, Intel `iomp`, or
another OpenMP runtime merely to discover a thread count. This avoids silently
introducing a second OpenMP runtime into a host process.

The OpenMP helper in `hybit.h` is header-only and calls the OpenMP API selected
by the host application's own compiler/linker.

## Avoid nested OpenMP x Rayon oversubscription

Do not invoke the same parallel HyBIT solve simultaneously from every worker of
an active OpenMP team.

A typical integration should call one HyBIT solve from one controlling host
thread, outside an active OpenMP parallel region when practical. Application
threads should be synchronized before and after the call as required by the
host program.

Otherwise an OpenMP team of `T` threads can each enter a Rayon computation with
up to `T` workers, causing severe oversubscription.

Prepared HyBIT contexts are not intended to be concurrently mutated by multiple
host threads. Internal Rayon parallelism is separate from concurrent host calls.

## MPI + OpenMP hosts

HyBIT 0.8 does not yet implement a distributed MPI solver, but it can be linked
into an MPI/OpenMP application.

For a host with `R` MPI ranks per node and `T` CPU threads allocated to each
rank, the normal integration is:

```text
MPI rank 0 -> one HyBIT call -> Rayon pool of T workers
MPI rank 1 -> one HyBIT call -> Rayon pool of T workers
...
MPI rank R-1 -> one HyBIT call -> Rayon pool of T workers
```

Size the job allocation so the node has enough CPUs for approximately `R * T`
software workers. The thread count is per process/rank.

If the MPI launcher sets a CPU set or process affinity for each rank, Rayon
workers normally execute within the process scheduling constraints established
by the OS/launcher. HyBIT does not translate OpenMP `OMP_PLACES` or
`OMP_PROC_BIND` policy into Rayon-specific affinity rules. Thread-count
synchronization and thread affinity are separate concerns.

## FrontISTR-style integration

For a FrontISTR-like MPI/OpenMP host, two straightforward configurations are:

Environment-controlled:

```text
OMP_NUM_THREADS=8
RAYON_NUM_THREADS unset
```

HyBIT then uses 8 Rayon workers when its pool is first initialized.

API-controlled:

```c
omp_set_num_threads(8);
hybit_set_num_threads((uint32_t)omp_get_max_threads());
```

The second form is preferred when the application changes the OpenMP team size
through API calls rather than through the environment.

If `RAYON_NUM_THREADS` is set, it overrides the `OMP_NUM_THREADS` fallback.

## Performance policy

A matching thread count does not imply that every sparse workload should execute
in parallel. HyBIT's validated execution policies can retain serial kernels for
small problems where Rayon scheduling overhead dominates.

G7 also showed that one universal hardware-independent nnz threshold is not
sufficient for all prepared local operators. Explicit serial, full-Rayon, and
task-limited prepared execution therefore remain available without embedding a
machine-specific automatic selector.

Thread count controls the size of the available worker pool; it does not force
every operation to use every worker.
