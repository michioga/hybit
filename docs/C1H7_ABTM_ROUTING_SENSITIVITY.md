# C1-H7 — Sensitivity and falsification study for experimental ABTM routing

**Base:** `867cc8e7418e6683b0a6215a939bb9083ec03a6b` (`develop/0.10.0`).

## Research question

C1-H6 showed a 160-DOF localized SPD rescue after 12 Jacobi-PCG iterations,
and KEEP_PCG on a global chain, with no opt-in promotion on five C1-G pilot
matrices. However, its strong-edge threshold (`0.1`), exact-fill condition,
maximum region size (96) and RHS/probe selection were demonstrated at just one
positive configuration. They must **not** be presented as a general policy.

C1-H7 deliberately perturbs those inputs **without modifying the solver**. It
calls the actual `fem_bench --abtm-weighted-routing` HyBIT Auto implementation
on generated known-reference linear systems. Both arms share matrix, RHS,
initial guess, stopping tolerance and 30-iteration budget. All systems have
SPD-by-construction energy: a Dirichlet 96-DOF chain, positive independent
exterior diagonals, plus a positive-semidefinite bridge Laplacian. For a
global negative control, the same length-160 Dirichlet chain is used.

## Sweep and evidence

Run:

```bash
cd ~/hybit
bash benchmarks/scripts/C1H7_run_sensitivity.sh smoke
bash benchmarks/scripts/C1H7_run_sensitivity.sh full
```

The `smoke` grid has 5 experiments including the two C1-H6 controls. The
`full` grid has **162** measured arms (5 localized bridge strengths and one
global chain; 3 RHS references; 3 region capacities; 3 probe lengths).
The parameters are:

- Local bridge: `1e-6, 1e-4, 1e-2, 1e-1, 0.5`. Strong-edge threshold in
  `hybit-auto` is fixed at `0.1`. Note the **normalized** bridge weight is
  `bridge / (2 + bridge)`; a raw bridge value of `0.1` remains weak.
- Region capacities: `64, 96, 128`; probe lengths: `8, 12, 20`.
- Manufactured known solutions: `smooth` (matches C1-H6 at bridge=1e-4),
  `flat`, `oscillatory`.
- Fixed 30-iteration budget, `1e-10` relative tolerance, maximum one
  local-direct promotion, one factor region, 1 MiB factor budget, CSR backend.

Output: `$HOME/c1h7_abtm_sensitivity/{smoke,full}/summary.csv`,
`summary.json`, generated Matrix Market inputs and individual raw logs.

A **rescue** requires baseline `MaxIterations` and original-CSR verified
residual above tolerance, versus Auto `Converged` with independently verified
true residual at/below tolerance, forward error at/below `1e-6`, and an actual
escalation. An **unnecessary promotion** is Auto escalation despite a
baseline that already converged within the same 30-iteration budget; this is
a diagnostic classification, not by itself proof of poor runtime.

`KEEP_PCG` runs must preserve status and iteration count in CLI results;
C1-H6 contains a stronger bitwise final-x regression for the canonical
global case. Parsed CLI metrics have limited printed precision, so C1-H7
does not make bitwise equivalence claims for all variants.

## Acceptance vs research observations

**Hard gate**: C1-H6 canonical positive rescue and global KEEP_PCG control
must remain valid, generated input/metrics must parse, and status/iteration
counts must agree on all non-promoted arms. All other grid outcomes are
observations to classify, *not predetermined wins* or failures that may be
removed to create a favorable summary.

The study is **not** an Auto production-policy launch. It does not modify
`hybit-auto`, `hybit-ffi`, solver defaults, or the public API, and does not
claim real-FEM generality, acceleration, or soundness on indefinite systems.
A next step is real SPD matrices and physical RHS from external FEM hosts,
then held-out policy thresholds and cost amortization.
