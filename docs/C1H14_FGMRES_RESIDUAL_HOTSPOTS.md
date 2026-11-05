# C1-H14-B1: post-solve FGMRES residual-hotspot diagnostic (opt-in)

Status: **validated opt-in diagnostic** as of the user's 2026-11-04 WSL2
results. Committed and pushed on `develop/0.10.0` as `155ab59`. The user's
Rust 1.99.1 SHA, fmt, 7/7 example tests, strict example Clippy and Release
build passed. This does **not** validate automatic local-direct rescue.

## Evidence and hypothesis

The real FrontISTR `01_elastic_hinge` system is nonsymmetric (252,168 rows,
18,846,277 CSR nnz). With ABTM ILU(0), fixed-restart FGMRES did not converge
in 360 iterations at restarts 30, 70, or 140. The existing C1-H13 observer
showed 100% of the remaining true residual in one weakly connected component
of 250,080 rows, while 2,088 singleton components contributed no residual.
This alone **does not** establish that the residual is locally concentrated
*inside* that giant component.

Two smaller, physical-RHS FrontISTR systems serve as control cases:

- `03_hyperelastic_cylinder`: 1,887 rows; 1,575-row dominant component.
  ABTM ILU(0), restart 70 converged in 104 iterations at `1e-10`.
- `11_contact_2beam`: 756 rows; a 358-row residual-dominant component.
  ABTM ILU(0), restart 70 converged in 54 iterations at `1e-10`.

These are single-run observations and **not** an automatic local-direct
rescue, a general numerical certificate, or a timing speedup claim.

## New diagnostic mode

The existing `fem_fgmres_bench` example gains independent
`--hotspot-diagnostics`, which may be combined with
`--component-diagnostics`. Both run **after** the timed solve. Neither changes
the solver state, preconditioner, mathematical class, convergence check,
status, public API, or production routing.

1. Reuse C1-H13's original-CSR `r = b - Ax` and weak-component partition.
2. Pick the component with the largest stable L2 norm of `r`; on a tie,
   prefer more rows and then the smallest first-row index.
3. Rank component rows by descending `|r_i|`, with ascending row index
   as a deterministic tie-break. Record the top five 1-based DOF indices.
4. For 32/64/128/256/512 selected DOFs, report the **squared-L2 residual
   energy share inside that component**. Compute it as a squared ratio of
   stable L2 norms, not by directly squaring possibly huge entries.
5. Separately rank `|r_i| / max_j |A_ij|`. This is a **row-maximum magnitude
   heuristic**, not a physical-unit normalization or convergence criterion.
   A zero row with zero residual contributes zero; a nonzero residual on a
   zero row or an unrepresentable scaled value disables the entire scaled
   metric explicitly, without hiding the raw metric.

Machine-readable output prefixes:

- `C1H14_SUMMARY`: chosen weak component, row count, share of global
  residual **norm**, and component's raw residual norm.
- `C1H14_TOPK`: `metric=raw|row_max_scaled`, requested `k`, actual selected
  rows, and **within-component residual energy fraction**.
- `C1H14_ROW`: top-five ranked 1-based row indices and magnitudes.
- `C1H14_SCALED`: explicit unavailability reason when scaling is undefined.
- `C1H14_DIAGNOSTICS`: post-solve diagnostic elapsed milliseconds. The
  diagnostic is not included in the existing `total wall` solve timing.

If residual is exactly zero, the energy fractions are zero, not fabricated
100% shares. Values are finite-checked and the raw residual remains available
when row scaling cannot be performed. Ranking rows by residual magnitude does
**not** prove the selected rows form a connected or suitable local factor
region. No ABTM graph growth, local LU, automatic promotion, or parameter
selection is introduced by B1.

## Reproduction on the user's authoritative WSL checkout

Require branch `develop/0.10.0` at the verified C1-H13 baseline `6333c21`
before applying the patch, with no tracked changes. Preserve unrelated
untracked `__pycache__` and do not stage it.

```bash
cd ~/hybit
git apply --check /mnt/c/Users/michi/Downloads/C1H14_RESIDUAL_HOTSPOTS_20261104.patch
git apply /mnt/c/Users/michi/Downloads/C1H14_RESIDUAL_HOTSPOTS_20261104.patch

git diff --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit --example fem_fgmres_bench
cargo +1.99.1 clippy --release --locked -p hybit --example fem_fgmres_bench -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example fem_fgmres_bench
```

After these gates pass, use the original physical `*.mm` / `*.rhs` pairs with
`--component-diagnostics --hotspot-diagnostics`, starting with
`11_contact_2beam`, then `03_hyperelastic_cylinder`, and finally the hard
`01_elastic_hinge` restart-30/360 control. Save original logs, process exit
status, exact HEAD, input SHA256 and peak RSS outside the repository.

Acceptance: C1-H13 component and verified relative residual values remain
unchanged. The raw energy shares are finite, monotone, between zero and one,
and equal to one at `k >= component size` for a nonzero residual. On the
giant component of `01_elastic_hinge`, report whether top-k raw and row-scaled
energy suggests a narrow concentration; investigate the ABTM structural
neighborhood separately in C1-H14-B2.

## Actual 2026-11-04 WSL2 observations

- `11_contact_2beam`, ABTM ILU(0), fixed restart 30, 360 iterations:
  `MaxIterations`, independently verified relative residual `1.318012e-8`;
  the 358-row dominant component contains effectively all residual energy.
  Top 32/128/256 raw rows contain **70.06% / 94.34% / 99.71%** of its
  squared-L2 residual. These rows have **not** been shown connected.
- `01_elastic_hinge`, same restart and budget: `MaxIterations`, verified
  relative residual `1.890115e-1`. Of the 250,080-row dominant component,
  top 32/128/512 raw rows contain **3.24% / 8.53% / 20.19%** of squared-L2
  residual energy; the corresponding row-max-scaled values are
  **2.52% / 6.81% / 16.97%**. Thus substantial residual remains outside
  even the highest-ranked 512 rows.
- The top raw and scaled row identities differ, so a local region chosen
  using either ranking must be evaluated separately. No row index alone
  identifies a physical FEM neighborhood.

Next experiment: `docs/C1H14_FGMRES_GRAPH_REGIONS.md` (B2), comparing these
top-k unconstrained scores against *connected* ABTM-selected regions.
