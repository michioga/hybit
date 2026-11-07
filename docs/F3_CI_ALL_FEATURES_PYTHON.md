# F3 — CI coverage expansion: all-features, MPI, Python benchmarks

Status: **candidate for WSL validation and GitHub Actions acceptance**.
Source baseline: `develop/0.10.0` at commit `3d0633f`, immediately following
the accepted F1/F2 numerical-reliability fix.

## Scope

The default CI job matrix, mdBook build, and package-manifest gate are
preserved. The only workflow change is in `.github/workflows/ci.yml`:

1. Keep no-default-features Clippy; replace the redundant default-features
   Clippy invocation with `--all-features` and install Open MPI development
   libraries on the Linux lint runner.
2. Add a Linux Rust 1.99.1 all-feature test job. Open MPI is installed to
   build/test the `mpi` feature and the experimental ILU(0)-ABTM feature.
   The job has a bounded 20-minute timeout.
3. Add a Python 3.12 benchmark-test job using
   `python -m pytest --import-mode=importlib benchmarks/scripts/ -q` from the
   repo root, matching the third-party review's 68-test passing command.

This change introduces no Rust API, solver policy, or benchmark numerical
changes. It also does not claim multi-rank MPI execution: the new job runs
Rust unit tests with Open MPI available as a system dependency. A separate
`mpirun -np 2` smoke test can be added after this gate is stable.

## Local preflight and acceptance

The guarded installer validates the exact current workflow SHA256
`ea31a5588496a314b5e9d2b8c71ca9f5109d442d2b71ade4b0f9660b91573b10`,
the HEAD, source manifest, and expected YAML insertion points. Dry-run
performs no writes. The applied change updates the workflow, this document,
`docs/SESSION_HANDOFF.md`, `MANIFEST.txt`, and `SOURCE_SHA256.txt`.

In WSL Ubuntu 24.04 (with Open MPI available), run before committing:

```
cargo +1.99.1 test --workspace --all-features --locked
cargo +1.99.1 clippy --workspace --all-targets --all-features --locked -- -D warnings
python3 -B -m pytest --import-mode=importlib benchmarks/scripts/ -q
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
```

If pytest is not installed, do not claim local Python acceptance; report the
missing prerequisite before proceeding. The final gate is a successful new
GitHub Actions run, which may reveal feature/MPI concurrency or runner-only
issues not seen in single-host WSL tests.

## Explicit limits

- Neither Windows/MS-MPI nor multiple concurrent MPI ranks are exercised by
  the added CI job. Existing Windows default-feature release tests remain.
- A passing compile does not imply distributed numerical correctness.
- Third-party review observations are input, not a substitute for a fresh
  Actions run on this exact revision.
- The A2 shadow horizon remains explicitly opt-in and non-promoted.
