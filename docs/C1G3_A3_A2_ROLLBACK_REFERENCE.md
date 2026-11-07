# C1-G3-A3-A2 — PCG shadow rollback reference

Status: **isolated reference for WSL acceptance; not production routing**.
Date: 2026-11-07 (JST). Base HEAD: `5888e9f` on `develop/0.10.0`.

## Why isolate this step?

The A3-A1 real-FEM test demonstrated that a short `island_shell` shadow trial
could admit a local-Cholesky candidate and yet produce a much worse final
relative residual and forward error. The A3-A1 decision telemetry correctly
explained the 128-iteration rejection: the candidate was worse than the
**fork checkpoint**, despite outperforming the baseline at that horizon.

An autonomous controller therefore needs a safe *post-admission* correction.
The necessary invariant is stronger than saving `x` alone: one must restore
the **original live PCG search direction, preconditioned residual scalar,
and scratch vectors**. Starting a new Jacobi-PCG session does not reproduce
an uninterrupted baseline recurrence.

This checkpoint is a separate `hybit-krylov` example and changes **no
existing Rust implementation, production API, default or experimental
HyBIT routing policy**. Its synthetic SPD operator has twelve diagonal
entries spanning powers of two. It is not a production FEM validation.

## Reference algorithm

1. During a live baseline PCG solve, capture the fork `x`, copied `PcgSession`
   and cloned `PcgWorkspace`. Reject the capture if the estimated `6 * n * 8`
   bytes exceed an explicit cap. It is a small reference estimate, not a
   peak-RSS or factorization-memory certification.
2. Start an independent fixed-preconditioner candidate at the same fork,
   then advance it through the admitted trial steps. The harness assumes the
   candidate was admitted by the preceding A2 policy; it does **not** modify
   that policy.
3. Continue the candidate through a bounded watchdog window. Independently
   verify the original-operator residual.
4. Replay the *unchanged baseline* from the saved state for precisely the
   candidate's selected logical iteration budget (`trial + watchdog`).
   Do **not** call `pcg_start_with_workspace` on that baseline.
5. Compare independently computed true residuals. If the candidate breaks,
   produces non-finite values, falsely reports convergence, or exceeds the
   baseline residual multiplied by a user-supplied ratio (at least 1), and
   the baseline is valid, restore its advanced `x/session/workspace` exactly.
6. The selected PCG iteration count is the actual chosen branch count. The
   discarded branch's work is measured separately; rollback cannot claim a
   speedup without charging that extra work.

Candidate and baseline execution remain fixed-preconditioner PCG. A true
change of preconditioner still requires a new candidate PCG session. This
example does not allow negative budgets, unconstrained checkpoint memory,
or a total iteration budget overrun.

## Synthetic numerical test coverage

- `delayed_candidate_deterioration_rolls_back_to_exact_baseline`: a weak
  diagonal candidate (`A^{-0.1}`) wins a two-step 0.95-ratio admission from
  the fork but loses after another two steps. Watchdog rollback restores the
  baseline *bit-for-bit in `x`*, and continuation matches the original PCG
  residual scalar and subsequent `x`.
- `useful_candidate_is_kept_after_watchdog`: a stronger candidate (`A^{-0.9}`)
  retains its state because it remains numerically preferable.
- `clone_memory_limit_rejects_before_state_mutation`: fails before cloning
  when one byte below the estimated memory bound.
- `invalid_watchdog_budget_is_fail_closed`: fails before candidate progress
  when proposed trial + watchdog iterations exceed `max_iterations`.

These tests are performed against the standalone reference and do **not**
prove safety of the production FEM integration or of a particular runtime
watchdog horizon. The exact final decision may still be horizon-dependent.

## Requirements before moving into the FEM controller

- Verify WSL Rust 1.99.1 example release tests, strict Clippy, release run,
  source SHA256 and formatting.
- Decide memory accounting for the fork snapshot **plus** the A2 speculative
  candidate allocation and local Cholesky factors. No double counting or
  ignored temporary peak is permitted.
- Handle unequal candidate/baseline early termination, breakdown and
  recursive false convergence under original-CSR certification.
- Preserve a single selected-path iteration budget while charging all
  speculative branch iterations and wall time in independent diagnostics.
- Define post-rollback `SolveReport` metrics and factor-cache semantics.
- Reproduce the `island_shell` 32-step false admission, then test a bounded
  watchdog with no worse verified final residual and explicitly measured
  extra memory/time. Hold out FEM cases beyond the tuning corpus.
- Keep both shadow admission and rollback opt-in until validated.

## Acceptance

Use `cargo +1.99.1 test --release --locked -p hybit-krylov --example
c1g3_shadow_rollback_reference`, then corresponding strict Clippy and release
run. The run prints `C1G3A3A2_RESULT|...`; expected `rolled_back=true` for
the deteriorating synthetic candidate. Do not interpret this as an actual
production rollback having occurred.
