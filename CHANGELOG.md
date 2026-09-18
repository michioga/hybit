# Changelog

All notable changes to HyBIT are documented here.

## 0.5.0

First public release candidate for GitHub and crates.io.

- Preserved the numerical and ABI implementation validated by the 0.4.1 release gate.
- Added crates.io-ready package metadata and versioned internal path dependencies.
- Added the public repository URL `https://github.com/michioga/hybit`.
- Added English and Japanese public READMEs.
- Added publication, contribution, roadmap, and release documentation.
- Added GitHub Actions CI for Rust workspace tests on Linux and Windows.
- Kept `hybit-ffi` repository-only while publishing the Rust facade and its internal Rust dependencies to crates.io.
- Removed release-candidate wording and version-specific validation text from runtime errors.

## 0.4.1

- Hardened MinGW external-language examples against PATH-dependent GNU runtime DLL mismatches.
- C++ example links libstdc++ and libgcc statically while keeping `hybit.dll` as the public ABI boundary.
- Fortran example links libgfortran and libgcc statically while keeping `hybit.dll` as the public ABI boundary.
- Release gate prints PE DLL imports when `objdump` is available.
- Added reusable `analyze -> prepare -> solve-many` execution through `HybitAnalysis` and `HybitPreparedSystem`.
- Added reusable `PcgWorkspace` and allocation-free prepared PCG work vectors.
- Made weighted local Schwarz application allocation-free with preallocated per-region scratch.
- Added lazy Hybrid preconditioner learning and local Cholesky factor reuse across subsequent RHS vectors for an unchanged matrix.
- Added matrix structure/value signatures that reject stale prepared contexts when coefficients change.
- Added `prepare_seconds`, `preconditioner_reused`, `solve_sequence`, and `krylov_workspace_bytes` diagnostics.
- Added C ABI prepared handles: `hybit_prepare`, `hybit_solve_prepared`, and `hybit_prepared_destroy`.
- Added C++ RAII `Prepared` wrapper and matching Fortran bindings.

## 0.3.0

- Generalized selective-direct PCG to multiple ABTM-expanded local subdomains.
- Added topology overlap and symmetric `1/sqrt(multiplicity)` weighted Schwarz correction.
- Added local-factor memory and per-stage timing diagnostics.

## 0.2.0

- Added poor-progress probing and automatic selective-direct escalation for SPD problems.
- Added residual/risk hard-DOF selection, ABTM one-hop topology expansion, and local dense Cholesky.
- Restarted PCG after changing the preconditioner.

## 0.1.1

- Stabilized Windows DLL/import-library handling for MSVC Rust with MinGW C/C++/Fortran consumers.

## 0.1.0

- Initial Rust workspace, CSR32/ABTM, Jacobi-PCG, C ABI, C++ wrapper, and Fortran binding.
