# C1-G3-A4-A1 — Shadow watchdog phase-cost decomposition

Status: **WSL Rust gate and 45/45 real-FEM evidence PASS; experimental instrumentation only**.
Baseline: `develop/0.10.0` commit `f66f76c`, the A3-A3 real-FEM
rollback checkpoint, whose GitHub CI run 42362605133 passed 8/8 jobs.

## Hypothesis and constraints

The full-horizon A3-A3 watchdog spends additional wall time on (a) candidate
continuation, (b) exact Jacobi replay, and (c) independent CSR residual
verification. Before changing the controller, measure these costs independently
to identify a viable optimization target.

**No change** to SPD prerequisites, shadow candidate admission, watchdog
comparison, rollback selection, PCG iteration accounting, clone-memory budget,
or the solver's default behavior. Shadow remains experimental opt-in.
No early abort, horizon shortening, memoized baseline, or admission-policy
change is included in A4-A1.

## New diagnostics

`HybridShadowTrialReport.phase_cost: HybridShadowPhaseCost` has eight additive
timing fields, in seconds:

| Field | Scope |
| --- | --- |
| `checkpoint_capture_seconds` | Saved rollback solution, session and workspace; zero without watchdog |
| `trial_clone_seconds` | Admission candidate solution/workspace clone and phase transition |
| `trial_candidate_seconds` | Candidate PCG setup and iterations, excluding independent CSR check |
| `trial_baseline_seconds` | Actual live Jacobi-PCG iterations |
| `trial_verification_seconds` | CSR checks of checkpoint, candidate and Jacobi |
| `watchdog_candidate_seconds` | Post-admission candidate PCG iterations |
| `watchdog_baseline_seconds` | Exact Jacobi replay from saved checkpoint |
| `watchdog_verification_seconds` | CSR checks of watchdog candidate and Jacobi replay |

`fem_bench` emits an additional `C1G3A4_COST|...` line when a shadow trial
is actually triggered, including separate `trial_other_ms` and
`watchdog_other_ms` (nonnegative remainder of the inclusive measured total).
The earlier `C1G3A2_SHADOW` and `C1G3A3_WATCHDOG` output fields and their
order are unchanged. `report.local_factor_seconds` supplies the existing
factor-setup measurement, printed separately as `factor_ms`.

Timing is diagnostic only. Instant granularity, nested measurement overhead,
allocator and CPU variation can cause sums to differ slightly from inclusive
totals. These timings are neither CPU instruction counts nor proof of speedup.

## WSL acceptance (Rust 1.99.1)

1. Guarded installer `--dry-run`, `--apply`, Rustfmt, then
   `--refresh-format-hash`.
2. `git diff --check`, `sha256sum --check --strict --quiet SOURCE_SHA256.txt`,
   `cargo +1.99.1 fmt --all -- --check`.
3. `cargo +1.99.1 test --release --locked -p hybit-auto`;
   `cargo +1.99.1 clippy --release --locked -p hybit-auto -- -D warnings`;
   `cargo +1.99.1 clippy --release --locked -p hybit --example fem_bench -- -D warnings`;
   `cargo +1.99.1 build --release --locked -p hybit --example fem_bench`.
4. Run `C1G3_A4_A1_replay.sh` (3 cases × 3 arms × 5 repeats), with source,
   corpus hashes, verified residuals, x errors, decisions, iteration/status,
   overall wall and RSS, and the new phase breakdown recorded under one
   `mktemp` evidence directory.
5. **Numerical gate:** for each case, `observe`, `a2`, and `watchdog` must
   reproduce A3-A3 status, decision, selected iterations and final verified
   residual/x error within the documented n144 tolerance. The rollback case
   (`island_shell`) must exactly match the Jacobi reference at printed
   precision; `multi_patch` must retain the useful candidate; anisotropy
   must remain NotTriggered. Reject any false Converged claim.
6. **Telemetry gate:** triggered runs have exactly one `C1G3A4_COST` record;
   all reported phase durations are finite/nonnegative; the previous A3
   decision/reason fields retain their values. No timing speedup acceptance
   is asserted from this measurement-only change.

After the verified 45/45 WSL evidence checkpoint, commit/push as a separate checkpoint.
See `docs/C1G3_A4_A1_PHASE_COST_EVIDENCE.md` for measured medians,
source/corpus integrity, and the replay provenance.
Keep the untracked `benchmarks/scripts/__pycache__/` untouched.
Next stage A4-A2 may compare bounded early-warning policies, but only after
an explicit numerical-safety and memory gate independent of this telemetry.
