# C1-G2: Independent forward-error reporting (opt-in benchmark instrumentation)

**Scope:** `fem_bench --rhs FILE --xref FILE` measures the full-vector relative
forward error `||x_computed - x_ref||_2 / ||x_ref||_2` for a nonzero reference;
when the reference is identically zero, it reports the absolute L2 error.
The output label is `relative x error` for consistency with the built-in
manufactured `b=A*ones` route. The benchmark also prints `reference residual`
for `||b - A*x_ref||_2 / ||b||_2`. The latter verifies that the supplied
reference vector corresponds to the actual RHS, but an approximate reference
may itself have nonzero residual.

The optional `--xref` is only accepted with an explicit `--rhs` (to avoid
ambiguity with the default `b=A*ones`). Both files are parsed with the
existing finite-vector and dimension checks. Invalid input is an error.
Forward-error reporting is **observational**: it changes no solver arithmetic,
preconditioning/escalation policy, C ABI, or convergence status. Importantly,
`MaxIterations` remains a **nonconverged** benchmark outcome even though the
`fem_bench` command may exit 0; downstream scripts must check the status and
original-CSR verified residual explicitly. A small true residual does not
establish a small forward error on an ill-conditioned system.

### Empirical reason for the diagnostic (2026-10-30, user WSL2)

For the manufactured SPD-like 2D P1 FEM case `c1g_p1_island_shell_n72`, one
WSL2 run with tol 1e-10 found:

| common max iterations | observe-only Jacobi | monitored Hybrid |
| --- | --- | --- |
| 240 | `MaxIterations`, 240 iters, rel true residual 3.161857e-7 | `MaxIterations`, 240 iters, no escalation |
| 1200 | `Converged`, 320 iters, rel true residual 8.934705e-11 | `Converged`, 588 iters, rel true residual 9.680579e-11, one escalation at iter 268 |

At iteration 268 the monitored *true residual* grew to `1.787744e-4` from
`1.708728e-7` at iteration 204, triggering the experimental rate threshold
0.99. The unmodified Jacobi recurrence nevertheless subsequently converged in
320 iterations. This is **not** an example of rescue; it is a guard/false
promotion case for future controller research. Neither branch reports the
forward error without an explicit reference, so no forward-error conclusion
about these two HyBIT runs should be inferred from residuals alone.

### Reproduce after `--xref` installation

```bash
cd ~/hybit
cargo build --release --locked -p hybit --example fem_bench
M="$HOME/mtx/c1g/c1g_p1_island_shell_n72"
for ARM in observe hybrid; do
  extra=()
  if [[ "$ARM" == observe ]]; then extra=(--monitor-only); fi
  OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    target/release/examples/fem_bench \
    --matrix "$M.mtx" --rhs "$M.rhs.txt" --xref "$M.xref.txt" \
    --tol 1e-10 --max-iters 1200 \
    --probe-iters 12 --monitor-iters 64 --monitor-rate-limit 0.99 \
    --max-region 128 --max-escalations 1 --skip-plain "${extra[@]}" \
    | tee "$HOME/mtx/c1g/c1g2_${ARM}.log"
done
```

Compare forward error and verified residual separately. Do not generalize the
performance of either arm from a single sample. `thermal1` is an already
convergent negative control, not a rescue case. Validate SPD analytically or
independently before using PCG corpus results as production evidence.

### Gates

```bash
cargo fmt --all -- --check
cargo test --workspace --locked
cargo test --release --workspace --locked
cargo test --release --locked -p hybit --example fem_bench
cargo clippy --workspace --all-targets --locked -- -D warnings
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
```

The 0.10.0 success criterion remains an *actual* local-direct escalation that
rescues a Jacobi baseline failure under the **same** tolerance, matrix/RHS,
initial guess, iteration budget, resource cap and numerical validity gates.
Do not present this reporting change as proof of rescue or full-system direct
solving.
