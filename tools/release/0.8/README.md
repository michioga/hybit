# HyBIT 0.8 release qualification tooling

These scripts qualify HyBIT 0.8.0.

Release-critical policy:

- exact workspace version 0.8.0;
- Rust 1.73 MSRV;
- exact Rayon 1.10.0 / rayon-core 1.12.1 pins in every crate that directly
  configures or uses Rayon, including repository-only `hybit-ffi`;
- source-integrity allowlist/hash verification;
- formatting, Clippy, workspace tests, and all-target builds;
- C ABI plus C/C++/Fortran runtime checks;
- OpenMP/Rayon interoperability checks;
- crates.io package inspection and dry-run;
- physical L-angle structural regression and prepared reuse.

`hybit-ffi` remains repository-only.

The OpenMP interoperability gate verifies both environment fallback and explicit
OpenMP-API synchronization while keeping HyBIT itself independent of a specific
OpenMP runtime.

The release-candidate gate must pass without `-SkipMsrv` or `-SkipRealFem`
before tagging or publishing.

Expected real-FEM reference inputs on the development workstation:

```text
Matrix      D:\Work\mf_solver-hybit-export\L-angle-K.mtx
Coordinates D:\Work\mf_solver-hybit-export\L-angle-K.coords
RHS         D:\Work\mf_rhs\L-angle-b.txt
```

After the exact release commit is qualified on `develop/0.8.0`, fast-forward or
merge that exact source to `main`, rerun the complete gate on `main`, publish
crates in dependency order, create immutable tag `v0.8.0`, and create the
GitHub Release from `RELEASE_NOTES_0.8.0.md`.

GitHub Pages is configured to deploy the frozen 0.8 documentation from `main`.
