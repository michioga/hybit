# Contributing to HyBIT

HyBIT is experimental numerical software. Contributions are welcome, especially when they include reproducible numerical evidence.

## Useful contributions

The most valuable reports include a small matrix or generator, right-hand side, solver settings, expected behavior, actual behavior, and the complete `SolveReport`. Performance reports should include build profile, Rust version, operating system, CPU, matrix size/NNZ, and enough commands to reproduce the measurement.

## Development workflow

1. Fork or clone the repository.
2. Create a focused branch.
3. Run `cargo test --workspace --release`.
4. If the change affects the C ABI on Windows, run `./tools/release/0.7/release-gate.ps1` from PowerShell until a dedicated 0.8 ABI/release gate is frozen.
5. Keep numerical changes separate from formatting/documentation-only changes where practical.
6. Explain numerical assumptions and add a regression test for bug fixes or algorithmic changes.

## Numerical changes

Changes to Krylov methods, preconditioners, convergence tests, matrix transformations, or local factorization should include at least one regression test. Avoid benchmark-only acceptance criteria: correctness, residual behavior, and failure handling come first.

## API and ABI changes

HyBIT is pre-1.0, so the Rust API may evolve. The C ABI is intended to remain stable where practical. Any C ABI layout or function change must update `include/hybit.h`, `include/hybit.def`, the C++ wrapper, the Fortran module, and the runtime examples together.

## Style

Prefer small, explicit numerical kernels and clear error paths. Avoid hidden global state. Keep allocation out of Krylov inner loops unless a change is explicitly justified and measured.

## License

By contributing, you agree that your contribution will be licensed under the MIT License used by this repository.


## Current development line

HyBIT 0.7.0 is published and the immutable release source is tagged `v0.7.0`. New numerical work belongs to the post-0.7 / 0.8 development line rather than modifying the released 0.7.0 source. Keep experimental solver changes separate from maintenance and documentation-only changes where practical. Any C ABI change must continue to update and test C, C++, and Fortran consumers together.
