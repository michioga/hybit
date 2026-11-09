# C1-H12-A10A: tolerance ladder and exact-error diagnosis

Status: **unaccepted experimental numerical diagnostic**. The package adds
isolated tests and a new Rust example; it does not alter production PCG,
public APIs, A8 or A9 code, or the original matrix/RHS evidence.

## Motivation

A9 completed 72/72 original-CSR residual and RHS-identity gates. Over 36
paired inputs, native multilevel AMG reduced total PCG iterations 7,538 to
2,782, and solve-only time 3,110 to 1,335 ms. However, on `island_shell`
with seeded-random manufactured exact x, multilevel
StrongGraph+JacobiSmoothed had a relative x-error around **2.05e-2** despite
`||b-Ax||/||b||` near `6.38e-11`. The two-level control's exact-x error was
around `2.24e-5` with similar true residual.

This discrepancy may reflect severe conditioning, near-null error directions,
round-off or an implementation issue. A small true residual alone is not a
reliable bound on forward x-error without conditioning information.

## Experimental protocol

Use the **original** `island_shell` 20,736-DOF matrix with three fixed
aggregation/basis arms (`graph_piecewise`, `strong_graph_piecewise`,
`strong_graph_smoothed`) and three independent manufactured exact solutions:
`ones`, `index_smooth`, `seeded_random`. Run both frozen A9 preconditioners
against three tolerances (`1e-10`, `1e-12`, `1e-13`) using a common global
budget of **1,200 PCG iterations**. Total: **54 processes**.

At `1e-10`, each A10-A run must reproduce the corresponding frozen A9 status,
iteration count, hierarchy, exact-x error, RHS fingerprint and true relative
residual. The original A9 600-iteration budget is not needed for replay
because all 18 selected historical controls converged within 600.

For each tol, record:

- Original-CSR residual `||b-Ax||/||b||`, used to certify convergence.
- An **independently accumulated** CSR residual using per-row Kahan summation.
- Exact solution error `||x-x*||/||x*||`, available for all A10-A RHS modes.
- Error response `||A(x-x*)||/||b||`, which should be compared with residual.
- Error-direction Rayleigh quotient `(x-x*)^T A (x-x*) / ||x-x*||²`.
- Error/residual amplification, two-level/multilevel iteration count, restarts,
  hierarchy stats and setup/solve/full time.

The Rayleigh quotient is **not a full estimate or proof of lambda_min**, and
the residual/amplification ratio is **not an independently computed condition
number**. Kahan summation provides a cross-check, not arbitrary precision.
Both residuals use binary64 matrix coefficients and RHS. At `1e-13`, roundoff
may prevent convergence; an honest `MaxIterations`/`Breakdown` is retained.
The gate prints any independently computed compensated residual disagreement
and does not hide it under a false success label.

## Owner WSL commands

After checking and applying the companion installation script:

```bash
cd ~/hybit
BASE="$HOME/c1h11_evidence/c1h12_a9_rhs_Xg80GE"

bash benchmarks/scripts/C1H12_A10A_accuracy_pilot.sh --dry-run "$BASE"
bash benchmarks/scripts/C1H12_A10A_accuracy_pilot.sh --run "$BASE"
```

The script checks baseline source/matrix provenance and runs Python negative
tests, Rust tests, Clippy `-D warnings`, release build, 54 native processes,
and numeric/evidence gates. It prints the raw evidence location and saves
`accuracy_tolerance_comparison.csv` with both preconditioners retained.

## Decision after results

Inspect each tolerance and RHS separately. Confirm whether x-error falls with
tolerance on `island_shell`, whether the error Rayleigh quotient indicates a
near-null direction, and whether residual measurements agree. If a 2% forward
error remains at tighter tolerances, investigate the matrix spectrum, data
scaling, and coarse-space construction **before** production integration.

Do not stage, commit, push or claim a production AMG or proven speedup until
the owner reviews the actual A10-A measurements.
