# Publishing HyBIT 0.7.0

Public repository: `https://github.com/michioga/hybit`

User-facing Rust crate: `https://crates.io/crates/hybit`

HyBIT is a Cargo workspace. The facade crate depends on internal crates, so crates.io publication must follow dependency order. `hybit-ffi` remains repository-only and is not published to crates.io in 0.7.0; its C ABI is built and runtime-tested by the release gate.

## Release branch and exact-source rule

The 0.7 release candidate is prepared on `develop/0.7.0`. The production freeze point is r32; later watchdog/energy-gate/spectral experiments are not part of 0.7.0. Keep `main` at the last published release until the exact candidate commit passes the complete gate. Merge that exact commit to `main`, rerun the gate from `main`, and only then publish immutable crates and create the tag.

## Complete release-candidate gate

Run from a clean `develop/0.7.0` worktree:

```powershell
.\release-candidate-gate.ps1 `
  -Matrix D:\Work\mf_solver-hybit-export\L-angle-K.mtx `
  -Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords `
  -Rhs D:\Work\mf_rhs\L-angle-b.txt `
  -RayonThreads 8
```

The complete gate includes source-integrity verification, workspace metadata/version checks, formatting, Clippy, workspace release tests, all release targets, Rust 1.73 MSRV, Rust/C ABI/C/C++/Fortran build and runtime checks, package inspection, a dry-run for the dependency-root crate, and the real L-angle structural regression. `-SkipMsrv` and `-SkipRealFem` are diagnostic conveniences only; a run using either switch is not eligible for publication.

Before the final gate, regenerate `MANIFEST.txt` / `SOURCE_SHA256.txt` after the final formatting/documentation changes and ensure the worktree is clean.

## crates.io authentication

Authenticate locally with a crates.io API token using `cargo login`. Treat the token as a secret; never commit Cargo credentials or paste tokens into logs.

## Publish order

Publish only after the exact source is pushed to GitHub. Immediately before each real publication, run the corresponding dry-run. Higher-level dry-runs can resolve only after same-version internal dependencies are visible in the crates.io index.

```powershell
cargo publish -p hybit-core --dry-run
cargo publish -p hybit-core

# Wait until 0.7.0 hybit-core is visible in the index.
cargo publish -p hybit-matrix --dry-run
cargo publish -p hybit-krylov --dry-run
cargo publish -p hybit-matrix
cargo publish -p hybit-krylov

# Continue after each dependency level is indexed.
cargo publish -p hybit-precond --dry-run
cargo publish -p hybit-precond

cargo publish -p hybit-auto --dry-run
cargo publish -p hybit-auto

cargo publish -p hybit --dry-run
cargo publish -p hybit
```

Published crate versions are immutable. Fix a serious post-publication problem with a new version rather than rewriting an existing release.

## Tag and GitHub Release

After the six crates are published from the validated source commit:

```powershell
git tag -a v0.7.0 -m "HyBIT 0.7.0"
git push origin v0.7.0
```

Use `RELEASE_NOTES_0.7.0.md` as the GitHub Release text. Verify the crates.io and docs.rs pages after publication, including the installation command `cargo add hybit@0.7.0`.
