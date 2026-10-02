# HyBIT 0.7 release qualification tooling

These scripts are retained to document and reproduce the HyBIT 0.7.0 release
qualification policy. They intentionally keep the 0.7.0 version checks,
`develop/0.7.0` / `main` release-candidate branch guard, Rust 1.73 MSRV check,
and the validated L-angle regression criteria.

They are **not** the future HyBIT 0.8 release gate.

The immutable `v0.7.0` tag still contains the original release-tree layout in
which these scripts lived at repository root. Moving the retained copies here
on `develop/0.8.0` does not alter that published source.

Current layout:

```text
tools/
  build/
    build.ps1
    build-examples.ps1
  gates/
    source-integrity-gate.ps1
  release/
    0.7/
      crates-package-gate.ps1
      public-release-gate.ps1
      real-fem-release-gate.ps1
      release-candidate-gate.ps1
      release-gate.ps1
      workspace-metadata-gate.ps1
```

For routine current-workspace validation use Cargo tests/Clippy plus the generic
source-integrity gate. Define an explicit 0.8 release qualification gate only
when the 0.8 release policy, version metadata, ABI surface, and representative
numerical regressions are frozen.