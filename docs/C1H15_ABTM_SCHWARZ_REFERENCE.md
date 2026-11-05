# C1-H15-A — ABTM graph-partitioned Restricted Additive Schwarz reference

Status: **experimental / unvalidated on WSL** (2026-11-05). This is a
small-matrix numerical reference, not a production solver integration, not a
PARDISO binding, and not evidence of convergence rescue yet.

## Why this experiment

The FrontISTR C1-H14 evidence establishes two distinct behaviors:

- `11_contact_2beam`: with ILU0-ABTM and fixed FGMRES restart 30, 360
  iterations stopped at true relative residual `1.318012270033e-8`, short of
  `1e-10`. In the dominant 358-row weak component, the 128 largest raw
  residual rows carried about 94.34% of its residual energy. A **connected**
  graph region requires an independent topology-based measurement.
- `01_elastic_hinge`: on 252,168 rows, fixed restart 30 / 360 stopped at
  true relative residual `1.890115052531e-1`. Residual is diffuse:
  the 512 largest raw residual rows held about 20.19% of dominant-component
  energy. A small local correction alone is unlikely to guarantee rescue.

Earlier PETSc additive Schwarz + MKL PARDISO experience motivates exploring
local direct solves, but does not establish that this RAS implementation will
work as well. The present comparison contains **no PETSc or PARDISO link**.

## Operator and graph construction

`crates/hybit/examples/c1h15_schwarz_probe.rs` implements one complete
reference preconditioner for FGMRES. It builds
`AbtmDualTopology::from_csr32`, partitions every row into disjoint core
sets by deterministic, undirected (outgoing + incoming) BFS, and optionally
adds a one-hop overlap to each core. Bitmap masks are used for visited-set
operations. For core `C_i` and extended region `R_i`, it computes a local
**partial-pivoted dense LU** of the principal submatrix `A[R_i,R_i]`.

The right-preconditioner is restricted additive Schwarz:

\[
M^{-1}_{\mathrm{RAS}}r
=\sum_i \widetilde R_{C_i}^{T}
  A_{R_i R_i}^{-1} R_{R_i}r .
\]

Only core degrees of freedom receive the local solution, so overlapping
regions do **not** sum duplicate corrections into an output row. This is
RAS rather than classical unrestricted overlapping Additive Schwarz.

The alternative `ilu0-abtm` arm invokes HyBIT's existing ABTM word-intersection
ILU(0) preparation. Both arms run the same `hybit::fgmres` with identical
initial guess, RHS, restart, tolerances and iteration cap.

## Hard safety limits

- RAS accepts complete square matrices of at most **10,000 rows**.
- Maximum local rows are explicit (`--max-local-rows`, default 512).
- Total resident dense numeric LU storage is checked **before** factor
  allocation (`--max-factor-bytes`, default 128 MiB). It is computed as
  `8 * sum_i |R_i|^2`; it excludes CSR, topology, index arrays and scratch,
  which must also be monitored via process peak RSS.
- `--overlap` may be only 0 or 1. Core rows must be nonzero and not exceed
  the maximum local rows. A singular/near-singular local factor, invalid
  pivot, NaN/Inf, or a capacity violation produces an explicit error, not
  silent fallback to Jacobi or identity.
- Exact true residual `||b-Ax||_2 / ||b||_2` is recomputed after every solve.
  For zero RHS the convention is absolute `||b-Ax||_2`. A reported
  `Converged` state that fails the independent tolerance gate is rejected.
  `MaxIterations` may exit with status zero: never infer convergence from
  process exit status alone.

**Performance caveat:** This reference stores each local factor densely and
constructs regions sequentially. A future production version should use
bounded sparse factorization, explicit symbolic reuse, locality-aware graph
partitioning and measured multi-RHS amortization. Do not apply this reference
to `01_elastic_hinge` (252,168 DOF).

## Required WSL 1.99.1 gates

```sh
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit --example c1h15_schwarz_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h15_schwarz_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h15_schwarz_probe
git diff --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
```

The four mathematical checks are: pivoted nonsymmetric local solve,
disconnected-block complete coverage, independent FGMRES convergence and
true residual, and overlap-only-core injection; additional guard tests
exercise singular-factor and factor-budget failures.

## Initial real-FEM comparison: no blanket speedup claim

First run `11_contact_2beam` and `03_hyperelastic_cylinder` with both
`--precond ilu0-abtm` and `--precond ras-lu`:

```sh
TUTORIAL_DIR='/mnt/c/Users/michi/FrontISTR/FrontISTR/tutorial'
RUN_DIR=$(mktemp -d ~/c1h11_evidence/c1h15_schwarz_XXXXXX)
git rev-parse HEAD > "$RUN_DIR/git_head.txt"
for case in 11_contact_2beam 03_hyperelastic_cylinder; do
  MATRIX="$TUTORIAL_DIR/$case/dump_matrix_1_0.mm"
  RHS="$TUTORIAL_DIR/$case/dump_matrix_1_0.rhs"
  sha256sum "$MATRIX" "$RHS" >> "$RUN_DIR/inputs.sha256"
  for arm in ilu0-abtm ras-lu; do
    prefix="$RUN_DIR/${case}_${arm}"
    /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
      -o "${prefix}.time" \
      timeout --signal=TERM --kill-after=10s 120s \
      target/release/examples/c1h15_schwarz_probe \
        --matrix "$MATRIX" --rhs "$RHS" \
        --precond "$arm" --tol 1e-10 --max-iters 360 --restart 30 \
        --core-rows 32 --overlap 0 --max-local-rows 256 \
        --max-factor-bytes 134217728 > "${prefix}.log" 2>&1
    echo "$case $arm exit=$?"
    rg '^C1H15_(INPUT|RESULT)' "${prefix}.log" || true
  done
done
echo "Results: $RUN_DIR"
```

If the non-overlapping RAS arm passes numerical gates but does not rescue
convergence, repeat with `--overlap 1`, possibly smaller core capacity,
and report **all** preparation costs and peak RSS. If preparation fails due
to a singular local system or budget, that is a legitimate negative outcome.
No parameters should be tuned based solely on a single successful arm.

For each pair, record: independent true residual (must be at or below
`1e-10` for a rescue), status, iterations, preparation and solve milliseconds,
number/max size of regions, numeric LU bytes, ABTM metadata bytes, peak RSS,
and SHA256 of the exact `.mm`/`.rhs` inputs. Additional restart 70 comparisons
are required before attributing rescue specifically to Schwarz: earlier
C1-H14 experiments showed restart length alone can change convergence.

## Next checkpoint

Only after WSL gates and real-FEM runs are reviewed should we decide whether
ABTM-enabled sparse local LU / RAS deserves a bounded experimental prepared
backend. Production auto-routing and stable public APIs remain untouched.
