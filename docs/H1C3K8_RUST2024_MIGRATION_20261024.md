# HyBIT 0.9 development: Rust 2024 Edition migration and D9-C1 experiment

Baseline `develop/0.9.0` commit `4e589997bcd248c976519c4518503e3beae168ee`.

This migration intentionally changes **development HEAD** to `edition = "2024"`, `rust-version = "1.99.1"`, and `resolver = "3"`. Every workspace crate inherits the package settings. The 0.8.0 immutable release and its historical 1.73 MSRV claims are unchanged. A pinned 1.99.1 toolchain file is included. The CI development matrix is updated from 1.73.0 to 1.99.1, while retaining the latest stable lane. Linux CI does not attempt to emulate Microsoft MS-MPI. Native MS-MPI enabled tests remain a Windows qualification gate.

## Edition migration approach

1. Fail-closed check: exact baseline HEAD/branch, clean Git worktree, source integrity, rustc/cargo version, payload hashes.
2. Prepare native Windows MSVC/LLVM/Microsoft MS-MPI environment.
3. Add previously blocked D9-C1 experimental probe and run `cargo fix --edition` **while packages are still edition 2021**. Cargo's migration fixes can update multiple Rust source files (unsafe extern declarations, unsafe environment mutations, identifiers, etc.).
4. Update workspace edition, declared Rust version and resolver, Rust toolchain pin, CI matrix, documentation.
5. Fix the 7 known Clippy 1.99.1 diagnostics with exact source anchors (5 is_multiple_of conversions, 2 collapsible if-let blocks), and lift the Schur probe tail-expression temporary into an explicit binding to address the Rust 2024 drop-order warning. Run up to three bounded Clippy machine-applicable auto-fix passes and require a separate zero-warning native MS-MPI strict Clippy pass. Clippy automatic changes are included in the source-integrity hashes and Git diff, not silently ignored.
6. Reformat and run release tests, including D9-C1 probe.
6. Rebuild `MANIFEST.txt` and `SOURCE_SHA256.txt` entries, stage ONLY reproducible tracked migration files and the five explicitly introduced files. Verify source integrity. Do not commit or push automatically.

Rollback on any failure restores the exact clean committed baseline via `git reset --hard <verified original HEAD>` (the script refuses to start if there are any existing modifications) and deletes only the five new payload targets. It does not call `git clean`, preserving out-of-repository user data.

## Rust 2024 audits

**Human review is required** for any migration-generated `unsafe { std::env::set_var(...) }` / `unsafe { std::env::remove_var(...) }` blocks: in multithreaded programs, environment mutation safety is not proven by wrapping a call in unsafe. Also audit auto-migrated FFI `unsafe extern` blocks. Ensure CI and numerical output reproduce baseline. **Clippy PASS is a necessary but not sufficient safety proof.**

The `Cargo.lock` is retained at the existing dependency versions unless Cargo needs a reproducible lock change. Release-level quantitative performance regression checks for D9-C1 (against METIS/SCOTCH) remain a separate next step.

## D9-C1 notes

D9-C1 remains isolated in `examples/partition_d9c1_refine_probe.rs`; no production partition API or MPI numerical kernel is changed by the experiment itself. The previously reported `is_none_or` and `File::create_new` MSRV warnings are resolved by updating the declared development MSRV to Rust 1.99.1, not by suppressing Clippy.

The existing D9-C1 benchmark script remains under `benchmarks/scripts/bench-k8d9c1-refine.ps1` and can be run only after Apply succeeds. No external partitioner is linked: METIS/SCOTCH remain WSL-only comparison sources; native PCG is Microsoft MS-MPI only.
