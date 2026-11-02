# C1-H2 — ABTM residual topology observer (experimental)

Date: 2026-11-02. Scope: **FEM-host-neutral**, FrontISTR is the first intended
real-FEM host, not a dependency or product restriction.

## Motivation and evidence

C1-G2 / C1-H1 measured SPD P1 FEM systems (5,184 DOF, 1e-10 tolerance). A
monitor-triggered local Hybrid-PCG restart was unnecessary on `island_shell`
(320 Jacobi-PCG versus 588 monitored Hybrid-PCG iterations) and `multi_patch`
(336 versus 488). The two cases do **not** prove that local factors are
ineffective: they demonstrate that a scalar residual-rate trigger is too weak
as the *sole* basis for a factorization decision.

FGMRES+ILU(0) meets the iteration limit on `local_soft` (93 / 100) and
`rotated_anisotropy` (23 / 240). `island_shell` is an important caveat:
FGMRES+ILU(0), restart 70, relative residual 8.35e-14 still had forward
error 3.22e-5 at tolerance 1e-13, whereas Jacobi-PCG had forward error
5.12e-9 at tolerance 1e-10. **Residual accuracy is not forward accuracy**.

## Contract

The first implementation is an **example-only** probe:

`crates/hybit/examples/fem_abtm_residual_observer.rs`

- Reads a square Matrix Market coefficient matrix and an explicit RHS;
  optionally reads a known reference solution (`--xref`).
- Assumes an **externally established SPD matrix class**; positive diagonal
  alone is not an SPD certificate. The user must declare this only for a
  suitable SPD test corpus. Do NOT apply to unconstrained contact or
  symmetric-indefinite systems.
- Runs Jacobi-PCG in segments using the public `hybit_krylov`
  `pcg_start_with_workspace` and `pcg_continue_with_workspace` APIs. It never
  changes the matrix, the preconditioner, or the live PCG session.
- Constructs `AbtmTopology` **once** before the solve (it retains explicit
  structural zeros, unlike a value-packed ABTM matrix). At checkpoints,
  obtains original-CSR **true residual** `b - A*x` for diagnostics only.
- Defines hotspot seeds from the Jacobi-scaled magnitude
  `s_i = abs(r_i) / sqrt(A_ii)` at `s_i >= seed_fraction * max(s)`.
  `s_i` is a ranking heuristic, NOT a local condition-number estimate or
  a forward-error certificate.
- Uses ABTM topology-word mask OR for one-hop reachability and ABTM row-word
  enumeration to count connected components **restricted to hotspot DOFs**.
- Reports checkpoint true relative residual, number of hotspots, fraction of
  Jacobi-weighted residual energy in hotspots, intersection/union Jaccard
  against previous checkpoint, count persistent for 2 and 3 checkpoints,
  connected component count, largest component size, and one-hop frontier.
- Runs an uninterrupted Jacobi-PCG baseline on identical inputs. The command
  **fails** unless status, iteration count, and *all final x f64 bits* match.
- Reports status even on `MaxIterations`; process exit=0 is NOT a
  convergence certificate. No auto-switch, no factorization, no rescue claim.

Checkpoint descriptions are spatial *proxies*. Without numerical coupling
weights and spectral/low-mode information, topology alone cannot distinguish
a weak global coupling mode from genuinely localized stiffness problems.

## Run

```bash
cd ~/hybit
cargo build --release --locked -p hybit --example fem_abtm_residual_observer
bash ~/patch/c1h2/C1H2_run_spd_corpus.sh
```

For a single case:

```bash
M="$HOME/mtx/c1g/c1g_p1_island_shell_n72"
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  target/release/examples/fem_abtm_residual_observer \
  --matrix "$M.mtx" --rhs "$M.rhs.txt" --xref "$M.xref.txt" \
  --tol 1e-10 --max-iters 1200 --first-check 12 \
  --monitor-iters 64 --seed-fraction 0.2
```

Re-test at `--seed-fraction 0.05`, `0.2` and `0.5` to detect the dependence
of region statistics on the hotspot threshold. Do not tune an automatic
promote/no-promote rule from only these five pilot matrices.

## C1-H2 advancement gates

1. Rustfmt, `cargo test --release --locked -p hybit --example
   fem_abtm_residual_observer`, workspace tests and Clippy must pass.
2. All five C1-G SPD pilot cases must print `PCG invariant : PASS`.
3. Compare mask dynamics on `island_shell`, `multi_patch`, and
   `rotated_anisotropy` at the **same** residual checkpoint boundaries.
4. Repeat on actual *serial* FrontISTR and another FEM program's systems
   before generalizing any policy. MPI rank-local matrices alone are NOT
   complete global SPD systems.
5. No production routing, FFI/API change or commit until reviewed.

## Next experimental stage (not implemented)

- Add numerical coupling graph `w_ij = |a_ij| / sqrt(a_ii a_jj)` to distinguish
  weakly connected regions. The structure-only mask must NOT mistake a
  low-energy global error mode for a local issue.
- Test durable versus transient masks, region budget, overlap growth, coarse
  correction versus local Cholesky, and local ILU or flexible FGMRES without
  changing default PCG semantics. Local ILU requires care because it is in
  general **nonsymmetric** and cannot simply be inserted into PCG.
- Record factor benefit and setup/solve amortization, not just iterations.
- Compare with KrylovScope spectral probes to identify near-null modes.
