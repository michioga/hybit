# C1-H15-C1 — FGMRES true-residual restart-boundary trace

Status: **experimental / WSL Rust 1.99.1 acceptance PASS (2026-11-05)**. No new public API,
no changes to prepared FGMRES/PCG, and no change to automatic solver routing.

## Motivation and existing evidence

On the complete FrontISTR `01_elastic_hinge` matrix (252,168 rows), fixed
FGMRES + ILU0-ABTM at 360 iterations did not converge:

| Restart policy | True relative residual | FGMRES solver time | Status |
| --- | ---: | ---: | --- |
| Fixed 30 | 1.890115052531e-1 | 18,944.054 ms | MaxIterations |
| Fixed 70 | 1.434331785548e-1 | 23,657.627 ms | MaxIterations |
| BudgetAware 30→70 | 1.321462044752e-1 | 20,558.394 ms | MaxIterations |

Evidence: `~/c1h11_evidence/c1h15_hinge_restart_7YSUt1/`. Each run used
`--tol 1e-10 --max-iters 360 --precond ilu0-abtm` and the physical RHS.

BudgetAware improves the result, but without restart-boundary telemetry it
is unknown **when** the controller increased restart and whether the late
residual trajectory was genuinely stalling. The large active weak component
(250,080 rows), 2,088 inactive singletons and diffuse residual indicate a
possible global error problem; no spectral/coarse-mode diagnosis is proven.

## Implementation contract

`crates/hybit/examples/c1h15_restart_trace.rs` calls the **existing public
research hook** `fgmres_with_workspace_and_restart_controller`, which receives
`FgmresRestartProgress` for every completed, nonterminal restart cycle.
These fields are exact `||b-Ax||` norms computed by the solver, not the
smaller Hessenberg residual estimate. The example uses the existing
ILU0-ABTM factor constructor, a zero initial guess, and identical solver
options. It logs, for each callback:

- cycle number and cumulative/cycle iterations;
- current and next restart lengths;
- remaining iteration budget;
- true residual norm and relative residual;
- logarithmic decay per iteration.

It also reports numeric factor bytes, FGMRES workspace bytes, elapsed setup
and solver time, and an **independently recomputed** final true residual.
Terminal cycles are deliberately **not** emitted as `C1H15C1_BOUNDARY`:
the controller is not invoked after convergence, breakdown, or exhausting
the iteration budget. The final `C1H15C1_RESULT` closes that gap.

### BudgetAware mirror: explicit limitation

`BudgetAwareFgmresRestartController` in `hybit-auto` is private. Therefore,
the standalone example contains a verbatim-equivalent **research mirror**
of its current deterministic decisions (four-cycle windows, thresholds
1.10/1.40, two decision hits). This is **not production source reuse**.

An evidence run is accepted only if the final verified relative residual,
status and iterations match the authoritative production
`fem_fgmres_bench --restart-policy budget-aware` reference under the same
options. If results diverge, inspect and correct the mirror; do not infer
production controller behavior from an unverified replay. The mirror is
intentionally named `budget-aware-mirror` in logs.

## WSL acceptance gates

```sh
git diff --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit --example c1h15_restart_trace
cargo +1.99.1 clippy --release --locked -p hybit --example c1h15_restart_trace -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h15_restart_trace
```

## First real-FEM experiment

```sh
RUN_DIR=$(mktemp -d ~/c1h11_evidence/c1h15_restart_trace_XXXXXX)
MATRIX="$HOME/mtx/01_elastic_hinge/dump_matrix_1_0.mm"
RHS="$HOME/mtx/01_elastic_hinge/dump_matrix_1_0.rhs"
git rev-parse HEAD > "$RUN_DIR/git_head.txt"
sha256sum "$MATRIX" "$RHS" > "$RUN_DIR/input_sha256.txt"
sha256sum crates/hybit/examples/c1h15_restart_trace.rs > "$RUN_DIR/source_sha256.txt"
/usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
  -o "$RUN_DIR/trace.time" \
  timeout --signal=TERM --kill-after=10s 240s \
  target/release/examples/c1h15_restart_trace \
    --matrix "$MATRIX" --rhs "$RHS" \
    --policy budget-aware-mirror --tol 1e-10 --max-iters 360 \
    --restart 30 --max-restart 70 > "$RUN_DIR/trace.log" 2>&1
echo "Process exit code: $?"
rg '^C1H15C1_' "$RUN_DIR/trace.log" | tail -n 18
cat "$RUN_DIR/trace.time"
echo "Results: $RUN_DIR"
```

Check `C1H15C1_RESULT`: expected comparison against the earlier authoritative
BudgetAware run is `status=MaxIterations`, `iterations=360`, and
`verified_relative≈1.321462044752e-1`. A small floating-point difference
is possible; a material mismatch invalidates the mirror comparison.

After observing the trace, decide whether the principal limitation is late
restart expansion, persistent slow decay, or unexplained loss of progress.
Do not change the restart policy or add sparse local direct corrections
before inspecting the actual restart-boundary measurements.

## C1-H15-C1 WSL real-FEM trace acceptance (2026-11-05)

The user's WSL2 run is retained at
`~/c1h11_evidence/c1h15_c1_trace_ozIYpJ/` (`trace.log`, `trace.time`,
`git_head.txt`, input and source SHA-256 lists). This is a single real-FEM
replay at the uncommitted C1-H15-C1 checkpoint based on HEAD `8ebce3d`,
**not** a convergence-rescue demonstration or a performance distribution.

- Matrix: FrontISTR `01_elastic_hinge`; 252,168 rows; 18,846,277 canonical CSR
  entries; input 19,043,712 Matrix Market entries; physical RHS.
- Parameters: ILU0-ABTM, zero initial guess, relative tolerance `1e-10`,
  maximum 360 iterations, `budget-aware-mirror` initial restart 30 and
  maximum restart 70.
- WSL validation (reported): `git diff --check`, `sha256sum --check --strict
  --quiet SOURCE_SHA256.txt`, Rustfmt, the 3 example unit tests, Clippy
  `-D warnings`, and the release example build all passed.
- **Mirror parity gate passed for this input:** the independently reported
  final true relative residual `1.321462044752e-1`, `MaxIterations` status
  and 360 iterations agree with the earlier authoritative production
  `BudgetAware` replay to the digits printed. This does **not** prove the
  mirror tracks the private controller for all inputs.

| Completed boundary | Current restart | Next restart | True relative residual |
| ---: | ---: | ---: | ---: |
| 30 | 30 | 30 | 4.523356199266e-1 |
| 60 | 30 | 30 | 3.655149116822e-1 |
| 90 | 30 | 30 | 3.227694285396e-1 |
| 120 | 30 | 30 | 3.016425169503e-1 |
| 150 | 30 | 30 | 2.773439200701e-1 |
| 180 | 30 | 30 | 2.611710482145e-1 |
| 210 | 30 | 30 | 2.437914504186e-1 |
| 240 | 30 | 60 | 2.302754869844e-1 |
| 300 | 60 | 60 | 1.608321698251e-1 |
| 360 (terminal) | 60 | — | 1.321462044752e-1 |

`C1H15C1_BOUNDARY` is emitted only for nonterminal cycles; the last row
comes from `C1H15C1_RESULT`. The growth from 30 to 60 was decided after
240 iterations. The per-iteration log-decay metric rose from
`1.901226225105e-3` on the last 30-step cycle to `5.981916063841e-3`
on the first 60-step cycle; correlation does not establish causation.

Reported costs: ILU factor bytes `228172672`, FGMRES workspace bytes
`288522200`, setup `1586.780 ms`, FGMRES solve `20305.393 ms`,
`/usr/bin/time` elapsed `24.52 s`, peak RSS `929524 KiB`, exit 0.
The factor and workspace counts are **not** the complete live-memory peak;
the RSS measurement includes other allocations and process state.

Decision: keep the existing production controller unchanged. C1-H15-C2 is
a separate, controlled, research-only experiment that will vary restart
growth timing under the same 360-iteration budget, compare independently
verified final residuals, full setup/solve time and peak RSS, and avoid
claiming convergence until the declared tolerance is actually met.
