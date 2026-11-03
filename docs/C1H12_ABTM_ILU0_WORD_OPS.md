# C1-H12 — ABTM ILU(0) bitwise symbolic preparation experiment

Date: 2026-11-03 (JST). **Research candidate; not promoted to default.**
Branch baseline: `develop/0.10.0`, pushed commit `9625955`.
The local WSL worktree remains the authoritative validation environment.

## Why test this

The user's real `16_heat_block` nonsymmetric 37,386-DOF FEM example
converged with GeneralSquare FGMRES using either canonical CSR ILU(0) or
ABTM ILU(0): both took 33 iterations with true relative residual
`8.856032e-11` at `tol=1e-10`. In 9 balanced repeated runs per ILU arm,
ABTM preparation was consistently slower (median about 26.244 ms versus
22.453 ms), although its solver-only timing was lower (80.570 ms versus
86.877 ms). Persistent factor storage and triangular application are the
same; the difference in solver-only timings has **no proven causal explanation**.
No PETSc/hypre result or independent physical model validation is implied.

Hypothesis: constructing ABTM word metadata directly from already canonical
CSR and replacing its 64-byte-per-word rank lookup table with `POPCNT` may
lower factor preparation time and transient memory without changing any
floating-point operation, factor, or solve semantics.

## Controlled arms

| CLI arm | Topology source | Rank | Word enumeration |
| --- | --- | --- | --- |
| `csr` | Canonical CSR only | Binary search | Existing CSR ILU(0) |
| `legacy-lut` | Temporary cloned canonical CSR then ABTM topology | 64-entry LUT | `AND`, `trailing_zeros`, clear-lowest |
| `direct-lut` | Direct canonical CSR word masks (`OR`/shift) | 64-entry LUT | Same as legacy |
| `direct-popcnt` | Direct canonical CSR word masks | Shifted mask + `count_ones` | Same as legacy |
| `direct-rotate-popcnt` | Direct canonical CSR word masks | Shifted mask + `count_ones` | Pivot-word `rotate_right`, `trailing_zeros`, `XOR` clearing |

All arms keep row sorting, duplicate accumulation, zero removal, pivot
stabilization, *numerical update order*, and persistent CSR factor layout.
The unmodified `GeneralSquarePreconditionerPolicy::Ilu0Abtm` selects the
existing `legacy-lut` algorithm. Direct variants are accessible only when
building with the opt-in Cargo feature `ilu0-abtm-experiments`; there is no
automatic route, SPD promotion, or API requirement for ordinary users.

The rotation arm is a controlled alternative, **not a predicted speedup**:
only the pivot word is rotated, after masking strictly above the pivot.
Rotating its live high bits to the low positions retains ascending numeric
update order. Clearing that enumerated bit with `XOR` is equivalent to
`bits &= bits - 1`; `OR` assembles topology, `AND` finds overlaps, shifts
produce bit masks, and `POPCNT` computes ranks. Arbitrarily substituting
`OR`, `XOR`, or rotation for an intersection would be incorrect.

## Correctness gate

The `hybit-precond` feature-gated tests compare each variant with canonical
CSR ILU(0): row pointers, columns, diagonal positions, pivot adjustments,
all floating-point LU entries **bit for bit**, and application to a RHS.
Cases include unsorted duplicate input, removed exact-zero off-diagonals,
columns crossing 63/64 and 127/128, and dense all-64-bit masks. Ordinary
non-feature tests still check production ABTM against CSR.

The benchmark's `--verify` mode separately builds CSR and the selected
variant and checks the complete factor and application bitwise. Any mismatch
terminates with an error, stopping the script. The benchmark without
`--verify` runs each arm in a fresh process for fairer peak RSS accounting.

## Reproduce on WSL2

```bash
cd ~/hybit

cargo +1.99.1 test --release --locked \
  -p hybit-precond --features ilu0-abtm-experiments

cargo +1.99.1 fmt --all -- --check

bash benchmarks/scripts/C1H12_run_abtm_ilu0_word_ops.sh \
  ~/mtx/16_heat_block/dump_matrix_1_0.mm
```

The script requires `/usr/bin/time`, calls the experimental example on the
real Matrix Market matrix, records matrix SHA-256, git HEAD/status, and first
runs all five correctness gates. It then runs three order-rotated rounds of
five **separate processes** (2 warmups, 7 timing samples each). The log records
`C1H12_SAMPLE`, `C1H12_RESULT`, `.time` process elapsed and peak RSS. Matrix
parse time is included in process elapsed and peak RSS, **not** the reported
`median_prepare_ms`. Verification runs allocate both reference and candidate
factors and are therefore deliberately excluded from RSS comparisons.

For ordinary end-to-end FGMRES, keep the existing `fem_fgmres_bench` baseline:
all variants should yield the same factor/iterations/true original-CSR
residual before interpreting any performance result. This research example
measures *factor construction*, not a claimed solve speedup.

Additional gates before commit/push: `cargo +1.99.1 test --release --locked
--workspace`, strict Clippy, `git diff --check`, `sha256sum --check --strict
--quiet SOURCE_SHA256.txt`, and manifest membership. The assistant's scratch
environment does not have Rust installed; the reported WSL observations below
are user-supplied logs, not locally reproduced Rust executions.

## 2026-11-03 WSL2 observations — original real-FEM matrix

Evidence directory: `C1H12_word_ops_R937wj`, preserved outside the repo.
Matrix SHA-256: `e1ec3d3de281efd928ab61c4e4383f7e21f40506197750d65d9d017f26b6c8b6`.
Run HEAD: `9625955` plus the uncommitted C1-H12 research patch.
All five factor and application bitwise verification runs passed; the
feature-gated `hybit-precond` test suite passed 30/30 Release tests; the
experimental example compiled. All 15 timed processes exited successfully.

| Arm | Median of three round medians, setup ms | Median peak RSS, KiB |
| --- | ---: | ---: |
| `csr` | 16.187133 | 31,296 |
| `legacy-lut` | 18.493661 | 40,224 |
| `direct-lut` | **14.883610** | 34,916 |
| `direct-popcnt` | 17.846216 | **31,252** |
| `direct-rotate-popcnt` | 19.012949 | 31,260 |

Each arm has 3 separately launched processes, 2 warmups and 7 timed
preparations per process (105 recorded prepare samples total). All arms had
703,366 canonical nonzeros, zero adjusted pivots, and 8,739,484 persistent
factor bytes. `direct-lut` beats both `csr` and `legacy-lut` in each of the
three rounds; against `legacy-lut`, its median-of-rounds setup is 19.5%
faster and its process RSS is 13.2% lower. `direct-popcnt` has nearly CSR's
RSS but longer setup. Rotation adds no observed speed benefit here.

**Limits:** this is one real nonsymmetric thermal matrix, and setup timing
is not a measured FGMRES end-to-end advantage for the new variants.
Process RSS includes input parsing. The script rotates launch order but
does not control machine-level load, and only three launches per arm are
available. No external-solver comparator is included. The first WSL
`cargo fmt --all -- --check` reported formatting-only diffs; the user
subsequently applied the formatting/SHA correction. SHA, rustfmt, and all
30 feature-gated Release tests then passed. The first strict workspace
Clippy attempt found one `collapsible_if` warning in the research example;
the follow-up correction combines the nested conditions without changing
the factorization or timing loop's behavior. The corrected Clippy gate is
pending WSL confirmation. Full-workspace tests remain unverified.

## Decision / rollback

Promote no implementation solely on one matrix or median timing. Require
bitwise factors on multiple structurally different GeneralSquare matrices,
true-residual equivalence, bounded RSS and repeatable whole preparation
benefit with balanced launch order. If the result is neutral or worse,
retain `legacy-lut`; if the result is better, consider an isolated follow-up
proposal with independent release and regression gates. Rollback is to omit
the feature entirely (current default), or revert this research-only patch.

Current candidate for further research: `direct-lut`; keep
`direct-popcnt` as a memory-oriented alternative. Do not change the
production `Ilu0Abtm` selection yet.

Single next action: apply the Clippy/SHA correction and rerun strict
workspace Clippy, then compare direct LUT with the existing policy on an
independent nonsymmetric matrix before proposing promotion.
