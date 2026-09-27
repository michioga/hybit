# Experimental spectral coarse basis (0.7-r36)

## Purpose

The r34-r35 diagnostics separated controller detection from correction quality. Late coarse stagnation can be detected, but current residual-centered local Cholesky regions did not rescue the tested long-tail matrices. Coarse-size and StrongGraph sweeps further showed that `s3dkq4m2` is not primarily limited by the number of aggregate constants. r36 therefore experiments with richer algebraic coarse content.

## Construction

For each aggregate HyBIT keeps the existing component-constant tentative modes and adds `extra_modes` coupled vectors. Each extra vector starts from a deterministic hash seed, is diagonal-orthogonalized against component constants and earlier extra vectors, and is passed through 16 aggregate-local damped-Jacobi filter steps. The filter uses only entries whose row and column lie in the same aggregate. The filtered vector is rescaled to a component-constant-like diagonal norm before it is inserted into the tentative transfer.

The entire tentative transfer then receives the existing one-step global Jacobi smoothing. The existing transfer storage/application policies and Galerkin builder are reused unchanged. The dense coarse factor/inverse path is also unchanged.

## Coarse target semantics

`target_coarse_dimension` remains a soft target. Auto sizing divides the target by `dofs_per_node + extra_modes`, so spectral enrichment trades aggregate count for additional modes instead of automatically inflating the coarse matrix. Graph topology can still cause target overshoot or undershoot.

## CLI

- `--coarse-basis spectral1`: one extra low-energy mode per aggregate.
- `--coarse-basis spectral2`: two extra modes.
- `--coarse-basis spectral3`: three extra modes.

Use `bench-fem-spectral-coarse.ps1` for fixed-target comparison against `smoothed`.

## Current limitations

- Experimental; no default policy change.
- Requires positive diagonal and enough local DOFs in every aggregate for the requested number of independent modes.
- Uses a filtered approximation to low-energy local modes, not an exact generalized eigensolve/GenEO construction.
- Stable C ABI/C++/Fortran configuration does not expose the spectral selector yet. Existing external-language APIs remain unchanged and must continue to pass the release gate.
