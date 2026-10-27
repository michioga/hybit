# HyBIT 0.9.0 — release notes (published 2026-10-27)

HyBIT 0.9.0 was published to crates.io as seven crates and tagged
[`v0.9.0`](https://github.com/michioga/hybit/releases/tag/v0.9.0) at source
commit `0a8cd32784a4a8ba5e3fe5840b526963bb7b1eb4`. The tag is immutable.
This public-status correction is a later documentation update to `main` and
does not modify the tagged source archive.

## Main changes since 0.8.0

- Advance the 0.9 development workspace to Rust 2024 Edition and minimum
  Rust 1.99.1.
- Add `hybit-distributed` to the public crate set, providing
  explicit distributed sparse-matrix, ownership, partition and Halo-related
  building blocks. MPI remains an **opt-in** feature, not a default.
- Extend ABTM-based sparse topology, symbolic region and field-aware
  ownership/partition experiments. Ordinary scalar numerical SpMV retains
  established CSR and prepared-CSR paths; ABTM is not claimed to universally
  outperform CSR.
- Develop native-MS-MPI, original-CSR prepared solve-many, rankwise phase
  profiling and independent numerical verification examples.
- Preserve the user-facing objective of a simple `hybit::solve` /
  `HybitSolver` API with iterative-first solving and bounded local
  Cholesky corrections where supported. A general whole-system sparse
  direct fallback is **not** claimed as complete.

## 2026-10-26 release-candidate evidence

- P1 static regression: rustfmt, strict workspace Clippy, release workspace
  tests and MPI example tests **PASS**.
- P2 numerical gate: **12/12 PASS** on `k8d1_fem64_shift002`, `thermal1`
  and `boneS01`, all combinations of Equal/Multilevel partition and
  P2P/Alltoallv communication, 8 native MS-MPI ranks, original CSR,
  independent residual/solution checks.
- P3-A: **32/32 numerically valid** instrumented trials on `thermal1`,
  1/2/4/8 ranks. Measured median 1-to-8 rank prepared-batch ratios were
  4.83 (Equal) and 5.46 (Multilevel); they are diagnostic scaling ratios,
  **not** production application speedups.
- P3-B: sampled MPI rank-to-PID and process affinity masks; all observed
  process masks spanned logical CPUs `0xFFFF`. Exact placement of Rust
  solver threads and enforced physical-core pinning were **not** proven.
- P3-C: **24/24 numerically valid** 8-rank thermal1 placement-crossovers
  using native-default, `seq:P:L` and `spr:P:L` layouts. The last two are
  experimental tuning choices, not a universal default.
- D9-C8G: original-CSR `cant` Jacobi PCG at 8 ranks, `tol=1e-10` and
  `max_iters=3000` recorded true residual `6.996658368670e-11` but
  maximum manufactured-solution error `1.179879512210e-3`, and did **not**
  achieve the required solution certificate. No threshold was relaxed.

The experimental large-matrix SPD screen is not a general SPD proof. A
small true residual does not guarantee a small forward error for a
poorly-conditioned matrix. These limitations are retained and documented.

## Packaging and supported boundaries

The completed crates.io publication sequence was:

1. `hybit-core`
2. `hybit-matrix`
3. `hybit-krylov`
4. `hybit-precond`
5. `hybit-distributed`
6. `hybit-auto`
7. `hybit`

`hybit-ffi` remains repository-only (`publish = false`). Nine native
C/C++/Fortran ABI runtime checks passed under Windows MSVC/LLVM Flang. Native
MPI qualification uses Microsoft MS-MPI, not MinGW or Open MPI.

## Completed publication qualification

- P4-A documentation/CI blockers resolved and checkpointed.
- P4-B native ABI: **9/9 PASS** after the prepared-Jacobi reuse-report fix.
- P4-C all seven published crate packages include the MIT license text.
- P4-D dependency-aware seven-crate `cargo publish --dry-run`: **PASS**.
- P5 final technical gate: **7 PASS / 0 FAIL** (source hashes, remote source,
  formatting, strict Clippy, workspace release tests, mdBook and evidence).
- P6-B seven crates published to crates.io: **7/7 registry-visible**.
- P6-C immutable annotated tag and GitHub Release: **PASS**.

The P3-B CPU placement limitation is not closed by publication. Exact
solver-thread affinity, physical-core pinning, and general application
speedups remain unverified; do not turn rank-scaling diagnostics into
production performance claims. Publication does not certify arbitrary
ill-conditioned systems or a universal sparse direct fallback.

See `docs/PUBLISHING.md` for the release process and
`docs/SESSION_HANDOFF.md` for detailed measurements and limits.
