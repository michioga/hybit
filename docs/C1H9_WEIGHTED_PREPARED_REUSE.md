# C1-H9 — RHS-gated reuse of a prepared weighted ABTM local factor

**Base:** `df0e05a22966d5ad78db0e0293abb7d19fa8f257` on `develop/0.10.0`.

## Objective

Earlier C1-H6 disabled local-factor caching when experimental weighted ABTM
routing was enabled: a factor that helped one RHS must not automatically
be applied to a different RHS. C1-H9 retains the per-RHS initial PCG probe
and original-CSR residual diagnostic and **reuses only the matching Cholesky
factor** for a bounded, naturally closed ABTM strong-coupling region.

## Contract

- Opt-in only; the default HyBIT Auto and general-square execution are unchanged.
- Every RHS is independently probed and diagnosed. Cached factors do **not**
  bypass this check, even on the same prepared context.
- A candidate must pass the C1-H8 closed-region, weak-boundary, canonical CSR,
  positive-diagonal, region capacity, and memory guards.
- Reuse is permitted only if its sorted candidate DOFs **exactly match** the
  cached region, and the prepared matrix structure **and numeric values**
  still match the analysis-time signatures. Otherwise factor a new region.
- If the current RHS needs no promotion, keep the Jacobi PCG recurrence,
  irrespective of any cached factor. A rejected RHS does not evict the cache.
- On an actual promotion, restart PCG with the new preconditioner, including
  when the factor itself is reused. Never continue the old Jacobi recurrence
  across a preconditioner change.
- Cache only a promoted solve that passes the final independent CSR
  residual certification. If the promoted Krylov
  stage breaks down, do not add that factor to the cache.
- On a reused factor: `preconditioner_reused=true`, zero newly charged
  `local_factor_seconds`, one actual promotion stage with populated factor
  bytes. On a factor newly built in this experiment:
  `preconditioner_reused=false`.
- Factor storage remains single-region/single-factor and subject to the
  existing user memory cap. This is not a multi-region factor dictionary.

## Evidence gate

The public prepared-API regression test uses a controlled SPD matrix with a
96-DOF strong component and 128-DOF region cap:

1. First hard RHS: one newly built local factor, certified convergence.
2. Scaled hard RHS: full re-diagnosis, matching factor reused, independently
   certified convergence and zero refactorization time.
3. A cheap RHS concentrated away from the hard component: **no promotion**,
   Jacobi result bitwise-identical to uninterrupted PCG.
4. A changed matrix numeric value: prepared context rejects reuse.
5. Return to the hard RHS: cached factor still reusable.
6. Fresh independently prepared context: rebuilds the same factor and reaches
   the same solution bits.

Run:

```bash
bash benchmarks/scripts/C1H9_run_prepared_reuse.sh
cargo +1.99.1 test --release --workspace --locked
cargo +1.99.1 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.99.1 fmt --all -- --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
```

## Limits

This is **not** approval for production reuse on general FEM systems. A
scaled RHS reuses the same residual footprint; more realistic distinct
load patterns and independent real-FEM matrices are needed. Factor reuse
reduces setup for repeated hard RHS patterns, but does not establish an
end-to-end speedup for small 160-DOF cases. A hash/signature precheck is
not a proof of collision-free matrix identity; production hardening should
consider stronger identity constraints and input mutation contracts.
