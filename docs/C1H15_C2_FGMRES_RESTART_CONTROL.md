# C1-H15-C2 — controlled FGMRES restart-growth experiment

Status: **research-only / WSL Rust 1.99.1 screening accepted (2026-11-06)**. This checkpoint
does not modify the production BudgetAware controller, solver API, numerical
preconditioners or the accepted C1-H15-C1 trace example.

## Hypothesis and scope

The accepted C1-H15-C1 true-residual trace on FrontISTR `01_elastic_hinge`
found that BudgetAware held restart 30 through iteration 240 and then grew
to 60. The verified relative residual at iteration 360 was
`1.321462044752e-1` (status `MaxIterations`; requested `1e-10`).

Hypothesis: earlier enlargement of the FGMRES Krylov subspace may improve
the true residual under the *same total 360-iteration limit*, but the
larger subspace costs memory and orthogonalization work. This is a
controlled numerical experiment, not a presumed convergence rescue.

## Implementation

`crates/hybit/examples/c1h15_restart_policy_probe.rs` is derived byte-for-byte
from the validated C1-H15-C1 example and then given narrowly scoped additions.
The installer requires exact SHA-256 of the committed C1-H15-C1 source.
The old example remains untouched, preserving its accepted evidence.

The new example uses exactly the same matrix reader, RHS parser,
`Ilu0Preconditioner::from_csr32_general_abtm`,
`fgmres_with_workspace_and_restart_controller`, zero initial guess and
independently recomputed final `||b-Ax||/||b||`.
Boundary/result output is prefixed `C1H15C2_`.

| Policy | Restart strategy | Research status |
| --- | --- | --- |
| `fixed` with `--restart 30` | 30 throughout | baseline |
| `fixed` with `--restart 70` | 70 throughout | baseline |
| `budget-aware-mirror` with 30/70 | same 4-cycle, 2-hit decisions as C1 | parity control |
| `early-growth` with 30/70 | identical mirror pressure thresholds, but **one** pressure hit instead of two triggers doubling | experimental |
| `scheduled-growth` with 30/70 | request 60 after completed iteration 120, 70 after completed iteration 240, subject to actual restart boundaries | experimental |

The scheduled policy is deliberately constrained to initial=30, max=70.
Actual boundary timing is logged and is not assumed to equal the request
if a cycle terminates early. Early growth still uses the same four-cycle
observation window and is not a predictive optimal restart algorithm.

The `fixed` arm allocates an FGMRES workspace for the fixed restart length;
adaptive arms allocate for max 70. This difference is intentional for
evaluating the real memory price of adaptivity; **do not call their peak RSS
identical-budget memory runs**. All arms use the same iteration count budget,
input and numerical factor policy, and report their workspace/factor bytes.

## Acceptance and numerical validity

Before a performance or convergence claim:

1. Check source integrity, Rustfmt, example tests and Clippy with Rust 1.99.1.
2. Confirm `fixed` 30 and 70 match previously recorded authoritative
   terminal residuals under the same conditions (tolerance, maxiters and RHS).
3. Confirm `budget-aware-mirror` matches the accepted C1-H15-C1 terminal
   status, iteration count and residual at printed precision.
4. Run each policy in an independent process with `/usr/bin/time` and
   `timeout`, retaining input and source SHA-256, HEAD, complete logs, peak
   RSS and elapsed wall time. Use identical CPU/thread environment.
5. Compare verified final relative residual **first**, then setup+solve
   time and RSS. Do not treat process exit 0 as convergence; only
   `status=Converged` with independent final residual at tolerance counts.
6. Report observed variance if timing claims are made; first single runs
   are diagnostic screening, not stable performance benchmarks.

## WSL gate

After `C1H15_C2_install.py --apply`, run `cargo +1.99.1 fmt --all` and
`--refresh-format-hash` before source integrity validation. This formatting
phase is necessary because the installer derives the new Rust example from
the exact already-formatted C1 code, and introduced syntax may be reformatted.

```sh
git diff --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit --example c1h15_restart_policy_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h15_restart_policy_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h15_restart_policy_probe
```

## Next decision

Classify early growth as helpful, neutral or harmful using independently
checked residuals under the same total iteration cap. If every restart
policy remains far from tolerance, prioritize a matrix-class-appropriate
global coarse correction and/or selective local direct reinforcement;
changing restart alone is not sufficient evidence for promotion to default.

## WSL screening evidence — 2026-11-06

Evidence directory: `~/c1h11_evidence/c1h15_c2_restart_tUBaCv/`.
Input SHA-256, example source SHA-256, commit HEAD, Rust version, per-policy
logs and `/usr/bin/time` records were saved. Five policies were executed in
separate processes on the same WSL host. Each arm ran once; these are
diagnostic screening results, **not variance-qualified performance claims**.

- FrontISTR `01_elastic_hinge`: 252168 rows, canonical CSR nnz 18846277.
- Physical RHS, zero initial guess, GeneralSquare ILU0-ABTM, `tol=1e-10`,
  `max-iters=360`, `max-restart=70`, independent recomputation of final
  `||b-Ax||/||b||` against the original CSR.
- WSL Rust 1.99.1 source-integrity check, Rustfmt, 5 example unit tests,
  Clippy `-D warnings` and release example build all reported PASS.
- Fixed30, Fixed70 and `budget-aware-mirror` independently reproduced the
  previously accepted final residuals to the printed digits.

| Policy | Verified relative residual | Solver ms | Elapsed s | Peak RSS KiB | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Fixed30 | 1.890115052531e-1 | 19451.310266 | 23.91 | 929544 | MaxIterations |
| Fixed70 | 1.434331785548e-1 | 24385.154174 | 28.48 | 929356 | MaxIterations |
| BudgetAware mirror | **1.321462044752e-1** | 20504.410838 | 24.53 | 929520 | MaxIterations |
| Early growth | 1.465369425088e-1 | 22370.749033 | 26.41 | 929416 | MaxIterations |
| Scheduled growth | 1.445092570687e-1 | 21983.573376 | 26.36 | 929396 | MaxIterations |

All five modes exited with process status 0 but **none converged** at the
requested tolerance. The minimum residual among them was BudgetAware.

| Policy | Observed growth boundary | Final reported factor bytes | FGMRES workspace bytes |
| --- | --- | ---: | ---: |
| Fixed30 | none | 228172672 | 127101080 |
| Fixed70 | none | 228172672 | 288522200 |
| BudgetAware mirror | 240: 30→60 | 228172672 | 288522200 |
| Early growth | 120: 30→60 | 228172672 | 288522200 |
| Scheduled growth | 120: 30→60; 240: 60→70 | 228172672 | 288522200 |

Early growth's verified final residual is about 10.9% larger and its solver
time about 9.1% longer than BudgetAware. Scheduled growth's residual is
about 9.4% larger and solver time about 7.2% longer. The result **rejects
these two early-growth schedules for this matrix and iteration budget**;
it does not establish that the production policy is globally optimal.
The approximately equal process peak RSS does not imply equal workspace
cost: Fixed30 allocated about 161 MB less workspace than adaptive modes.
The overall RSS high-water mark also contains matrix input and other
allocations, and need not be attained during the Krylov solve.

Decision: no production restart-policy change. A one-problem, 360-iteration
restart experiment has not rescued the large diffuse active component.
The next primary validation milestone is the chartered C1-G1 hard-SPD
Jacobi failure → Hybrid rescue corpus; a **separate** GeneralSquare global
coarse/local-direct correction experiment may follow. The current
non-symmetric FrontISTR matrix must not be routed into SPD-only Cholesky.
