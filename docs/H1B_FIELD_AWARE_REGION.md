# HyBIT 0.9 H1-B — Field-aware ABTM / Schwarz symbolic region checkpoint

**Design baseline:** `81f653d` on `develop/0.9.0`; H1 mixed-DOF layout already tested in Rust/C/C++/Fortran. **Status:** implementation supplied for developer-machine validation; do not treat it as accepted until tests below pass.

## Contract and scope

1. `close_to_field_blocks(layout, mask)` includes *all* DOFs in every `(entity_kind, entity_id, field_id)` block touched by the input. It never joins **distinct fields on the same physical entity** implicitly. No solver-internal permutation is assumed; all mask bits are in physical DOF ordering.
2. `grow_field_aware_region(topology, layout, seeds, hops)` computes undirected structure `A union A^T` and repeats one **matrix-edge hop followed by field-block closure**. Seed closure applies even for zero hops. Explicitly stored numeric zeros still count as symbolic edges. Each step uses the existing G3 `grow_undirected_region` implementation.
3. `growth.local_pattern(&topology)` reuses G3 `extract_local_submatrix_pattern` for `A[R,R]`. This is **symbolic preparation for Schwarz**, not a direct solver or Schur complement; existing local numeric-plan APIs can subsequently gather changing values.
4. `field_block_owners(layout, per_dof_rank_ids, rank_count)` checks that a candidate MPI assignment does not split physical field blocks and returns an owner per block. This is **not** a new distributed partitioner, communicator or Halo plan. Different fields on the same node may be on different ranks.
5. The C ABI adds `hybit_dof_layout_expand_blocks` and `hybit_dof_layout_grow_region` with packed 64-bit outputs (word bit `k%64` corresponds to DOF `k`). `hops` and `seed_count` are `size_t`, field indices and DOFs are zero-based. New C++ methods and Fortran `ISO_C_BINDING` procedures are wrappers only; existing `hybit_solve` ABI and report layout remain unchanged. The ABI graph grow **builds `AbtmDualTopology` from CSR each call** and is not suitable for a hot loop until a separate prepared object is validated.
6. No implicit conversion to SPD, numerical factorization, automatic preconditioner selection, contact geometry or MPI communication is introduced. Convergence and API simplicity constraints in `docs/HYBIT_LONG_TERM_ARCHITECTURE.md` remain authoritative.

## Correctness gates

- Rust tests: input dimension mismatch, directed graph edges, explicit zero structural edge, block closure over 64-bit boundaries, mixed solid/thermal/shell/contact multiplier, split-owner rejection, and `A[R,R]` pattern cross-check against CSR reference.
- FFI Rust integration: packed mask layout, single-step region, zero-hop case, invalid seed/output dimension, no seed.
- C/C++/Fortran: compile, link and execute real `hybit.dll` calls with correct WinLibs runtime search priority; report per-language PASS individually.
- Keep old PCG/FGMRES, Source Integrity and Clippy gates PASS.

## Run on Windows (developer machine)

```powershell
cd D:\Work\hybit\hybit
cargo fmt --all -- --check
cargo test --release --locked -p hybit-matrix -p hybit-ffi -p hybit
cargo clippy --locked -p hybit-matrix -p hybit-ffi -p hybit --all-targets -- -D warnings
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
cargo build --release --locked -p hybit-ffi
```

Compile the added language examples with `-I .\include`, link to `target\release\hybit.dll.lib`, and put matching WinLibs `mingw64\bin` before conflicting GNU runtimes on PATH. The package's `README.md` provides executable commands with strict native exit checks.

## Next step (after H1-B tests pass)

H1-C: cached prepared topology and **real owner/partition + HaloPlan** experiments. Measure block-induced duplication, global ghost count, peer count and load imbalance. Do not imply that per-call field-aware growth or block ownership validation already improves MPI runtime. Any promoted automatic routing requires end-to-end convergence and memory evidence.
