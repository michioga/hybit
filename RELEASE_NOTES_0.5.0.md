# HyBIT 0.5.0 — first public release

HyBIT 0.5.0 is the first public release of the Autonomous Hybrid Sparse Solver project.

The release provides a Rust API for real SPD sparse systems, ordinary CSR32 input, PCG, adaptive selective local Cholesky correction, weighted overlapping Schwarz for multiple difficult regions, ABTM-driven topology expansion, and reusable `analyze -> prepare -> solve-many` execution.

The repository also contains a stable C ABI layer with C++, C, and Fortran examples. The Windows release gate has been validated with an MSVC-built Rust DLL and MinGW/GCC/GFortran consumers.

## Validation carried into 0.5.0

The numerical implementation is unchanged from the 0.4.1 release gate. Small synthetic SPD regression problems showed:

- single difficult block: Jacobi-PCG 33 iterations, HyBIT Auto 13 iterations;
- two difficult blocks: Jacobi-PCG 25 iterations, HyBIT Auto 13 iterations;
- prepared solve-many: the learned local factors and Krylov workspace were reused on the second RHS without repeating the adaptive probe or local factorization.

These are regression validations, not general performance claims. Real application-scale FEM/HPC benchmarking is still required.

## Current limits

The automatic path is currently limited to real SPD systems and PCG. Local direct factors are bounded dense Cholesky factorizations. Prepared factor reuse requires an exactly unchanged matrix. The project does not yet provide nonsymmetric Krylov methods, distributed-memory execution, GPU execution, or coarse-grid correction.

HyBIT is experimental pre-1.0 numerical software. Validate results independently for engineering use.

## Install

```bash
cargo add hybit
```

Repository: https://github.com/michioga/hybit

Documentation: https://docs.rs/hybit

License: MIT
