# H1 — Field-aware DOF metadata (experimental, 2026-10-14)

## Why

HyBIT remains an **autonomous hybrid sparse linear solver**: simple Rust/C/C++/Fortran user APIs, iterative Krylov methods for easy parts, selective local direct factorizations for hard regions, guarded fallback and true-residual verification. ABTM masks and Rayon/MPI optimize *required* work and data motion, not the mathematical solver contract. Follow `docs/HYBIT_LONG_TERM_ARCHITECTURE.md` and `docs/HYBIT_DECISION_LOG.md`.

This H1 checkpoint **adds only reusable mixed-DOF metadata**. It does not claim mixed-field coupling-aware preconditioning, frictional contact solving, matrix permutation, automatic field/block Schur, or dynamic contact Halo are implemented.

## Model / invariants

- Physical DOF blocks have `(entity_kind, entity_id, field_id, components)`. Supported entities: node, edge, face, element, multiplier, constraint, other. A node may have several distinct fields (translation, rotation, temperature).
- `field_id` is an application-defined `u32`, **not** a fixed enum or automatic physical-field detector. Components are positive; duplicate `(kind,id,field)` triples are rejected. Blocks are **ordered physical DOF ranges**; the solver-internal permutation is still future work.
- DOF positions, block indices, and components use **zero-based indexing** in Rust, C, C++, and Fortran. Total DOFs and physical IDs are `u64`; per-block component counts are `u32`. ABI uses `size_t` for block indices and `uint32_t` for entity/field/component identifiers. C/Fortran `entity_id` is 64-bit; Fortran callers should use nonnegative values fitting `c_int64_t`.
- Rust: `FieldAwareDofLayout::new`, `.total_dofs()`, `.blocks()`, `.block_range()`, `.locate()`, `.field_mask()`, `.validate_square_matrix_size()`. `.field_mask()` builds `DofMask` using 64-bit word masks; it does not access matrix numerical values.
- C ABI: opaque `hybit_dof_layout_t` handle, create/destroy/num_dofs/block/locate. Existing C ABI symbols and report layout remain unchanged; handles own a **copy** of metadata.
- C++: `hybit::DofLayout` RAII wrapper. Fortran: `type(hybit_dof_block)` and `bind(C)` procedures in module `hybit`.
- H1 does not attach a layout to the solver or change the numerical solve. A dimension check is provided in Rust only at this checkpoint. Future additions must check layout/CSR dimension, field-wise norm/scaling, ownership, permutation and epoch lifetime before changing routing.

## Run validation on the developer's Windows 11 x64 MSVC environment

```powershell
cd D:\Work\hybit\hybit
cargo fmt --all -- --check
cargo test --release --locked -p hybit-matrix
cargo test --release --locked -p hybit-ffi
cargo test --release --locked -p hybit
cargo clippy --locked -p hybit-matrix -p hybit-ffi -p hybit --all-targets -- -D warnings
pwsh -NoProfile -File tools/gates/source-integrity-gate.ps1
```

C/C++/Fortran syntax checking is separately possible with GCC/G++/gfortran (WinLibs in the developer environment). **Link/run** requires the built `hybit.dll` plus import library as documented for the existing FFI. H1 has no MPI runtime dependency; MPI correctness remains covered by the existing B3/C2 gates.

## Acceptance gate and planned next step

1. Rust layout and FFI unit tests pass; all old solver tests remain green.
2. C, C++ and Fortran header/module compilation pass. Verify `hybit.def` exports the five added DOF functions without changing old names.
3. Source integrity manifest/hash gate passes, including this document and language smoke examples.
4. Next checkpoint **H1-B** will validate field groups/masks against ABTM sparse adjacency, mixed block cuts/owner layout, and CSR equivalence. Only after correctness, measured reuse and cost gates may the solver's autonomous planner use this metadata.

For session continuity, append measured PASS/FAIL gates and the exact local HEAD to `docs/SESSION_HANDOFF.md` after running the tests. Do not mark these tests as passed merely because the patch applied.
