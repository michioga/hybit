# Publishing HyBIT

Public repository: `https://github.com/michioga/hybit`

User-facing Rust crate: `https://crates.io/crates/hybit`

## HyBIT 0.7.0 publication record

HyBIT 0.7.0 was published on 2026-09-27 from the exact validated source commit:

```text
1fdcd6a1b8127c84306c38c3fdbad42563538ad8
```

The immutable release tag is `v0.7.0`, and the GitHub Release uses `RELEASE_NOTES_0.7.0.md`.

The published Rust crates are:

1. `hybit-core`
2. `hybit-matrix`
3. `hybit-krylov`
4. `hybit-precond`
5. `hybit-auto`
6. `hybit`

`hybit-ffi` remains repository-only and is not published to crates.io. Its C ABI, C++ wrapper, and Fortran `ISO_C_BINDING` consumer path are built and runtime-tested by the repository release gate.

## 0.7.0 qualification record

The r32 numerical production line was frozen before release. Later watchdog, local-energy-gate, and filtered spectral-enrichment experiments were excluded from 0.7.0.

The exact release commit passed the complete `release-candidate-gate.ps1` on `develop/0.7.0`, was fast-forwarded to `main`, and passed the complete gate again on `main` before publication.

The qualifying gate covered:

- source-integrity verification, including tracked-file/`MANIFEST.txt` agreement;
- workspace metadata and version checks;
- formatting and Clippy with warnings denied;
- workspace release tests and all release targets;
- Rust 1.73 MSRV;
- Rust/C ABI plus C, C++, and Fortran build/runtime checks;
- package inspection and crates.io dry-run validation;
- the real L-angle structural regression;
- independently verified residual, iteration guard, and prepared solve-many reuse.

The release-reference L-angle solve converged in 220 iterations with independently verified relative residual `9.378557e-9`. Prepared reuse also passed.

## Immutable release rule

Published crate versions and release tags are immutable.

Do not move `v0.7.0` to a later documentation or development commit. A serious defect in 0.7.0 must be corrected with a new version such as 0.7.1 or a later release, not by rewriting the existing tag or crates.io artifacts.

Post-release documentation commits may advance `main`; they do not alter the 0.7.0 release source.

## crates.io authentication

Authenticate locally with a crates.io API token using `cargo login`. Treat the token as a secret; never commit Cargo credentials or paste tokens into logs.

For GitHub Release automation, GitHub CLI may be authenticated with `gh auth login`. Authentication tokens and device codes must not be committed or included in release logs.

## Dependency order for future workspace releases

HyBIT is a Cargo workspace. Higher-level crates depend on lower-level crates, so publication must preserve this dependency order:

```text
hybit-core
    |
    +--> hybit-matrix
    +--> hybit-krylov
              |
              v
         hybit-precond
              |
              v
          hybit-auto
              |
              v
            hybit
```

`hybit-matrix` and `hybit-krylov` may be published at the same dependency level after `hybit-core` is visible in the crates.io index.

For each future release:

1. Freeze the exact numerical source set and exclude unintended experimental/generated artifacts.
2. Regenerate `MANIFEST.txt` and `SOURCE_SHA256.txt`.
3. Run the complete release-candidate gate without release-qualifying skip switches.
4. Merge the exact validated commit to `main`.
5. Rerun the complete gate on `main`.
6. Publish crates in dependency order, confirming each dependency level is visible in crates.io before publishing dependents.
7. Create an immutable annotated version tag on the exact published source commit.
8. Create the GitHub Release from the matching release notes.
9. Perform any post-release documentation synchronization in a later docs-only commit without moving the release tag.

The release gate records performance measurements for regression context, but correctness, residual verification, compatibility, and prepared reuse remain the publication criteria.