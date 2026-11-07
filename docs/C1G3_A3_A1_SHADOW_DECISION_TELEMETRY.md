# C1-G3-A3-A1 — Shadow-trial admission telemetry

Status: **experimental opt-in diagnostic; WSL acceptance pending.**
Date: 2026-11-07 (JST). Base: `develop/0.10.0` commit `fe23aa0`.

## Motivation

The C1-G3-A2 horizon sweep found non-monotonic local-Cholesky admissions.
In particular, `island_shell` trial length 128 rejected a candidate with a
smaller end-of-trial residual than the baseline. The A2 log omitted the
checkpoint residual and therefore could not explain which additional
admission gate was responsible. It would be incorrect to infer a bug solely
from those two residuals.

This change records **why** each A2 comparison accepted, rejected or failed.
It does not change the A2 admission predicate, candidate/baseline execution,
PCG state cloning, or final CSR residual certification. No rollback or
production policy is introduced. The shadow feature remains **disabled by
default** (`shadow_trial_iterations == 0`).

## New report fields

`HybridShadowTrialReport` now additionally exposes:

| Field | Meaning |
| --- | --- |
| `decision_reason` | First failed gate, or successful admission |
| `checkpoint_true_relative` | Original-CSR residual at the shared trial fork, divided by `||b||` (or the unscaled norm if `b == 0`) |
| `baseline_status` | Baseline PCG branch status, `None` if no comparison ran |
| `candidate_status` | Candidate PCG branch status, `None` if no candidate status was returned |

The diagnostic reason is a public `HybridShadowDecisionReason` enum, re-exported
through the `hybit` facade. Possible values:

| Reason | Interpretation |
| --- | --- |
| `Accepted` | All A2 numerical admission tests pass |
| `CandidateExecutionFailed` | Candidate returned an error before an outcome could be recorded |
| `CandidateNonFiniteResidual` | Independent original-CSR candidate norm was non-finite |
| `CandidateFalseConvergence` | Candidate reported convergence despite its true norm exceeding the target |
| `CandidateBreakdown` | Candidate PCG returned breakdown |
| `BaselineAlreadyAtTarget` | Baseline true residual already reached the solver target |
| `CandidateNotBelowCheckpoint` | Candidate did not improve on the fork's true residual |
| `CandidateRatioNotMet` | Candidate true residual did not reach `accept_ratio * baseline_true` |
| `CloneBudgetExceeded` | Shadow candidate skipped before allocation because its clone estimate exceeded the cap |

The last three numerical gates are ordered exactly as the A2 acceptance
comparison. Earlier candidate-failure checks take precedence. If trial setup
does not reach the shadow comparison (for example, no eligible local region),
the benchmark continues to print `C1G3A2_SHADOW|decision=NotTriggered`.

## Log compatibility and interpretation

The existing single-line `C1G3A2_SHADOW|...` record keeps its **prefix and all
old key/value fields in their original order**. Four suffixes are appended:

```text
|reason=<enum>|checkpoint_true_relative=<f64>|base_status=<Option<SolveStatus>>|candidate_status=<Option<SolveStatus>>
```

Iteration counts continue to describe the selected PCG path; their sum in
`executed_trial_iters` measures extra speculative branch work. Added telemetry
does not make a short-horizon admission predictive of final accuracy or runtime.

## Required WSL 1.99.1 acceptance

1. Guarded installer `--dry-run`, then `--apply`; `cargo +1.99.1 fmt --all`,
   followed by installer `--refresh-format-hash`.
2. `git diff --check`, `sha256sum --check --strict --quiet SOURCE_SHA256.txt`,
   `cargo +1.99.1 fmt --all -- --check`.
3. `cargo +1.99.1 test --release --locked -p hybit-auto` and strict Clippy
   for `hybit-auto` and `hybit --example fem_bench`, plus release example build.
4. Replay the A2 FEM cases with unchanged CLI options and compare **status,
   iterations, verified residual, relative x error, and escalation decision**
   with historical A2 logs. Only the diagnostic suffix and negligible profiling
   overhead should change. Do not claim bitwise identity from rounded printed
   numbers alone.
5. Verify `island_shell` and `multi_patch`, horizon lengths 32 and 128, and
   `rotated_anisotropy` with `NotTriggered`. Record fresh input and source
   checksums and uncommitted source diff alongside logs before committing.

Data and previous A2 logs are described in
`docs/C1G3_A2_SHADOW_HORIZON_EVIDENCE.md`. Real FEM has not been executed
against this A3-A1 code at the time this package was prepared. The separate
rollback/watchdog proposal remains a future stage, not part of this patch.
