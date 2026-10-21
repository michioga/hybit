# H1-C3K7-A — bounded-memory noncontiguous root CSR scatter

Base: `develop/0.9.0` commit `12af447beab0dae69652e922020d001ef43ee942`.
This is an experimental MPI input/SpMV improvement, **not** a general sparse solver release.

## Motivation and implementation

K6 `scatter_assignment_from_root` materialized `Vec<AssignmentOwnedCsrRows>`
for every rank before any sends. With a global root CSR already resident,
this added approximately the sum of the rank-local CSR payloads to rank 0's
peak heap. K7-A keeps the public function signature and its downstream
`prepare_assignment_owned_rows` / `spmv_assignment` interface. It:

1. Preflights global and per-rank row / nnz / MPI-i32 limits, finite coefficients,
   column validity, and owner assignment on root. The preflight allocates O(P)
   count metadata and uses no per-rank CSR chunks.
2. Broadcasts global dimensions, O(n) owner labels (unchanged from K6), and
   O(P) per-rank nnz counts so receivers allocate bounded buffers.
3. Processes destination ranks sequentially: constructs exactly one owned CSR
   chunk with original coefficients, collectively checks construction, sends
   the four CSR arrays, drops the chunk, collectively checks the receiver.
4. Builds rank 0's own CSR chunk **after** the other destinations were sent.
5. Uses existing reciprocal MPI Halo preparation and numerical SpMV unchanged.

The only new synchronization overhead is extra collective error gates per
recipient. K7-A is intentionally not a throughput/latency optimization.

## Measurements and limitations

New CSV fields appended to the backward-compatible K6 example:

- `root_stream_chunk_bytes_est`: root-only maximum of one rank's owned CSR
  payload; row indices u64, row pointers u32, column indices u64, values f64.
- `root_k6_all_chunks_bytes_est`: root-only sum of *all* corresponding chunks,
  representing K6's former simultaneous CSR-chunk allocation footprint.
- `max_concurrent_root_chunks`: 1 for this single root-streaming path.

These are **deterministic payload estimates**, not measured peak resident
set size or allocator high-water marks. In particular they exclude root's
full CSR, Matrix Market parsing temporaries, owner-map duplication, allocator
capacity/fragmentation and MPI internal buffering. MPI results and numerical
accuracy must be checked before claiming any practical memory improvement.

This is still root-read + point-to-point scatter, not true parallel file I/O.
Non-root ranks never hold the global CSR, but root still must read the entire
matrix. The O(n) owner map is still broadcast to every rank. The K7-B/K7-C
checks and K8 solver integration remain separate work.

## Verification

Apply script performs: clean baseline/commit checks; payload SHA checks; source
integrity gate; native MSVC + LLVM + **MS-MPI** initialization; Rustfmt;
strict Clippy (`-D warnings`); distributed library and K6 example unit tests;
release build; source integrity reconstruction; staged-diff checks. On failure
it restores the previous state from byte-level backups and unstages only its
five controlled paths. No automatic commit/push.

Native run after successful Apply:

```powershell
cd D:\Work\hybit\hybit
& .\benchmarks\scripts\bench-k6-root-scatter.ps1 `
  -RepoRoot D:\Work\hybit\hybit `
  -Matrix D:\Work\thermal1.mtx `
  -Ranks @(4) `
  -Policies @('equal','region','multisource') `
  -Repeats 8 -OutputDir D:\Work -Prefix k7a_thermal1

& .\benchmarks\scripts\bench-k6-root-scatter.ps1 `
  -RepoRoot D:\Work\hybit\hybit `
  -Matrix D:\Work\boneS01.mtx `
  -Ranks @(4,8) `
  -Policies @('equal','region','multisource') `
  -Repeats 32 -OutputDir D:\Work -Prefix k7a_boneS01
```

Every native case must report `max_rel=0.000e0` (or tolerable relative
error `<=1e-11`), `root_only=true`, `concurrent_chunks=1`, with owned CSR nnz
accounting and Halo exchange preserved. Expect timing variation relative to K6
because of the per-destination collective gates and sequential construction.
