# HyBIT 0.9 H1-C3K7-D — Sparse owner setup optimization (experimental)

Baseline: commit `158158e03f472459332341905882debae2a9b162`, native Microsoft MS-MPI, Windows MSVC. Existing K7-C root CSR ownership and public API remain unchanged.

## Changes

1. Root directory: FIX1 restores the proven K7-C `BTreeMap` implementation after the original K7-D `HashSet` implementation repeatedly increased `scatter_ms`. Entries remain sorted and deduplicated. This isolates and retains the independently faster Halo optimization.
2. Verify directory coverage using a temporary distinct-column hash set: remove locally owned globals, then require exact membership of every remote directory entry. This preserves strict missing/extra detection without a `BTreeSet` insertion per nonzero.
3. For Halo setup, derive per-peer request vectors *directly* from the previously validated and globally sorted sparse remote owner directory. By construction, each peer's request vector is sorted and duplicate-free. No repeated `owner_of()` or balanced-tree insertion per nonzero.

No MPI collective sequence changes, no numerical SpMV modifications, and no changes to the K6/K7-B path or production automatic solver selection. The temporary hash sets are *not* retained. The sparse Owner Directory's logical bytes, root CSR lifetime, and exact-original validation remain the same.

## Scope and limitations

This is a performance hypothesis to be measured, not a guaranteed improvement. FIX1 restores BTreeMap on root. The separate validation stage still uses a temporary HashSet; request construction uses the validated directory directly. The root still constructs one CSR chunk per rank and the distribution still uses per-target collectives; those can be addressed in K7-D2 after measurements. Peak process RSS has **not** been measured.

## Paired comparison

The APPLY script first rebuilds the exact committed K7-C release example and copies it to `target/release/examples/mpi_k7c_sparse_owner_spmv_k7d_baseline.exe` *before* changing source; then it builds the optimized example. The new `bench-k7d-sparse-owner-paired.ps1` runs old/new pairs in alternating order, checks exact-original CSR, numerical relative error, matched NNZ/ghosts/owner capacity, and writes per-pair metrics. Because runs share the same OS environment but are sequential and not physically isolated, review paired scatter+halo distribution, not a single fastest execution.

Examples (PowerShell):

```powershell
& .\benchmarks\scripts\bench-k7d-sparse-owner-paired.ps1 -RepoRoot D:\Work\hybit\hybit -Matrix D:\Work\thermal1.mtx -Ranks @(4) -Policies @('equal','region','multisource') -Repeats 8 -Cycles 3 -OutputDir D:\Work -Prefix k7d_thermal1
& .\benchmarks\scripts\bench-k7d-sparse-owner-paired.ps1 -RepoRoot D:\Work\hybit\hybit -Matrix D:\Work\boneS01.mtx -Ranks @(4,8) -Policies @('equal','region','multisource') -Repeats 32 -Cycles 3 -OutputDir D:\Work -Prefix k7d_boneS01
```

Integrity: `MANIFEST.txt` and `SOURCE_SHA256.txt` update only after rustfmt. Script stages all six touched paths before Source Integrity Gate. Failure rolls back tracked paths and unstages patch (preexisting target binaries may remain; target is ignored by Git).
## FIX1: paired benchmark-driven root rollback

Original K7-D validation (21 October 2026): 27/27 paired setups regressed compared with committed K7-C even though Halo setup improved. Root scatter became materially slower. FIX1 restores only the K7-C root directory constructor; the fast Halo request construction and strict directory validation remain. Treat improved setup performance as an unverified hypothesis until paired MS-MPI tests pass. Preserve the previous trial binary under target/release/examples/mpi_k7c_sparse_owner_spmv_k7d_hashset_trial.exe. Do not commit on static correctness alone.

## FIX2: isolate validation cost (21 October 2026)

Paired native MS-MPI FIX1 results: 17 of 18 boneS01 pairs improved in scatter+Halo total, but all 9 thermal1 pairs regressed. FIX2 restores the committed K7-C BTreeSet-based strict remote-directory validation, isolating its cost from the retained fast Halo request construction. Root BTreeMap (FIX1) and all numeric and MPI code paths are otherwise unchanged. Performance benefit remains an unverified hypothesis until repeatable paired measurements. Keep previous FIX1 binary under target/release/examples/mpi_k7c_sparse_owner_spmv_k7d_root_fix1.exe; committed K7-C baseline binary is unchanged.
