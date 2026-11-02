# C1-H8: Natural-sized ABTM strong-coupling components (opt-in)

Base: `7a1793f7acdcc4bdbd599333198cb843c2101fa4` on `develop/0.10.0`.
Status: candidate patch; WSL2 Rust, numerical, and full quality gates **not yet executed**.

## Why change the opt-in research policy

C1-H7's 162 runs included 36 rescues but the original C1-H6 policy only admitted
components that filled `max_local_region_size` **exactly**. This makes the
maximum an unintuitive target rather than a ceiling, missing viable regions
smaller than the configured cap. C1-H8 changes **only** the opt-in candidate
selection. All production defaults remain unchanged.

## Algorithm and fail-closed boundaries

- Start at the highest absolute Jacobi-scaled original-CSR residual DOF after
  the ordinary initial PCG probe.
- Traverse the `AbtmTopology` through numerical edges with normalized coupling
  `|A_ij| / sqrt(A_ii * A_jj) >= 0.1`. Do not stop at the configured size and
  claim a region has closed: if the BFS finds a *(capacity + 1)*-th reachable
  DOF, reject the entire candidate and preserve the live Jacobi-PCG recurrence.
- If traversal closes naturally, accept region sizes from 2 to the ceiling,
  **provided** that no strong edges leave the selected set, a positive weak
  edge crosses the boundary, and the conservative dense bound `8*n_region^2`
  fits the user-specified persistent local-factor budget.
- Actual local Cholesky creation continues through the existing
  `select_regions_with_factor_budget` and `HybridPreconditioner::from_csr32`
  path; restart PCG only after a successful preconditioner change. No new FFI,
  no new public option, no coarse-space or multi-region support, no change to
  defaults, and still no factor cache across RHS in this research mode.
- Sorted unique CSR columns, finite positive diagonals, residual finiteness,
  and an eligible single-region/zero-overlap opt-in configuration remain
  required. The threshold is **experimental**, not a physical FEM classifier.

## Positive / negative evidence

`C1H8_natural_region.py` creates known-reference SPD chains with natural local
strong-component sizes **64, 96, 128 DOF** and one global strong 160-DOF
negative control. All have the same 160x160 global operator size, with a weak
positive-semidefinite edge Laplacian at the local component boundary and a
strictly positive Dirichlet chain. Local cases use a positive 1e-4 bridge;
three manufactured references (`smooth`, `flat`, `oscillatory`) are evaluated
against capacities 64, 96, 128 at a common 12-iteration Jacobi probe and a
30-iteration total budget (1e-10 relative tolerance).

- `smoke` runs **5** configurations, including the old L96/cap96 reference,
  the new L96/cap128 acceptance, the L96/cap64 rejection, L64/cap128, and
  the global/cap128 negative.
- `full` runs **36** configurations. The strict gates require the canonical
  L96/cap96 **and** new L96/cap128 rescue, no promotion when a strong
  component exceeds the cap, and no promotion on the 160-DOF strong chain.
  Other outcomes are recorded as observations and must not be cherry-picked.
- On no-promotion arms, CLI statuses and iteration counts must agree with the
  ordinary Jacobi PCG baseline. Stronger bitwise baseline invariants are
  checked in existing `hybit-auto` unit tests. The Rust C1-H8 tests check
  region discovery for all three natural sizes and reject truncations.
- The existing C1-H6 C1-G corpus (5 matrices) and C1-H7 sensitivity sweep
  remain important external negative/positive regressions; run both after
  smoke/full before commit.

Run:

```bash
cd ~/hybit
bash benchmarks/scripts/C1H8_run_natural_region.sh smoke
bash benchmarks/scripts/C1H8_run_natural_region.sh full
bash benchmarks/scripts/C1H6_run_auto_route.sh
bash benchmarks/scripts/C1H7_run_sensitivity.sh full
cargo +1.99.1 test --release --workspace --locked
cargo +1.99.1 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.99.1 fmt --all -- --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
```

Results are written to `~/c1h8_natural_region/{smoke,full}/summary.csv` and
`summary.json`, with every raw CLI log saved under `logs/`.

## Limits

This only removes one brittle parameter coupling. It does not prove that a
numerically closed component is the right preconditioner, that all local SPD
modes can be corrected, or that dynamic region selection is beneficial on real
FEM meshes or in wall-clock cost. Compare original-CSR true residual and known
reference forward error; never claim rescue solely on a recursive Krylov norm.
This study is host-neutral; FrontISTR integration and external FEM evidence
remain later steps.
