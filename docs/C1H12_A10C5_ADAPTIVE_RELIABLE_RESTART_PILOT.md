# C1-H12-A10C5: Adaptive Reliable Residual Restart and Stagnation Pilot

## Motivation from frozen A10C4

Frozen A10C4 verified 108 original-CSR residual replacement/tolerance
outcomes. The 25- and 50-period schemes each certified 18/36 convergences,
but 18/36 cases reached 1,200 iterations. The most extreme frozen case
recomputed/restarted 1,029 times in 1,200 iterations because false-convergence
events bypassed the nominal 50-step periodic interval. The resulting repeated
loss of conjugate directions can waste work and obscure the observed numerical
precision or preconditioner limitations.

## Independent A10C5 policy

An experimental textbook left-PCG is prepared with the same independent AMG
hierarchies as A10C4. Production PCG, API and defaults are unchanged.

- Case: original `multi_patch` (n=20,736, nnz=103,104), single thread.
- Three ABTM aggregation/basis arms × two two-level/multilevel policies.
- Two RHS: manufactured `Ones` and external `PointLoad`.
- Relative tolerances: 1e-10, 1e-11, 1e-12; 1,200 iteration budget.
- Two cooldown controls: 25 or 50 iterations **minimum between all residual
  replacements**, including false-convergence-triggered replacements.
- Original CSR, compensated original CSR and recursive residuals are checked at
  iteration 0, 1, every 25 steps, and at most every 5 steps after a recursive
  tolerance crossing. The latter throttles repeated *verification*, too.
- A replacement requires a scheduled 25-step checkpoint or a verified
  recursive/true discrepancy, plus an expired cooldown, plus failure to meet
  all three tolerance certificates. Replacement restarts the direction
  `p=M^-1(b-Ax)` and refreshes the preconditioned inner product.
- Progress is defined by a **10% decrease in the best joint independent
  CSR/Kahan residual certificate**. If no such progress occurs over 200
  iterations after at least one replacement, report `Stagnated` and stop early.
  This is a heuristic computational-budget decision, **not** a proof that
  the desired solution or residual is mathematically unattainable.
- A `Converged` status still requires the recursive, original-CSR and Kahan
  certificates to meet the user's tolerance simultaneously.

There are 6 OS processes × (2 RHS × 3 tolerances × 2 cooldowns) =
**72 A10C5 outcomes**. Every outcome is paired with frozen A10C4 at the same
RHS/tolerance/arm/policy and corresponding 25- or 50-period setting.

## Evidence and fail-closed conditions

`C1H12_A10C5_adaptive_gate.py` verifies:

- All 72 identities and original FEM, RHS, hierarchy, ABTM provenance.
- Frozen A10C4 108-row comparison and checkpoint SHA-256 fingerprints.
- Complete 25-step original CSR checkpoints, additional threshold-check
  spacing, recursive and compensated residual values and gap ratios.
- Exact cooldown exclusion, suppressed restarts, replacements, progress
  markers and false-event counts reconstructed independently from records.
- `Converged` and `Stagnated` statuses against original CSR and heuristic
  criteria; fail-closed GNU time exit, completeness and provenance.

Accepted evidence is saved separately as `adaptive_update_comparison.csv`
and `adaptive_residual_trace.csv`, with six preserved `.log` and `.time`
files. The A10C4 raw results and all prior evidence remain unchanged.

## Owner WSL commands

```bash
cd ~/hybit
A10C4="$HOME/c1h11_evidence/c1h12_a10c4_reliable_wdamI6"
A10B="$HOME/c1h11_evidence/c1h12_a10b_prepared_oBIupI"

bash benchmarks/scripts/C1H12_A10C5_adaptive_pilot.sh --dry-run "$A10C4" "$A10B"
bash benchmarks/scripts/C1H12_A10C5_adaptive_pilot.sh --run "$A10C4" "$A10B"
```

Compare matched convergence statuses, original-CSR best and final residuals,
iteration counts, replacement and verification work; do not infer policy
speedup from GNU time because each process contains both cooldown conditions.
The A10C5 stopping heuristic must not be released into production without
independent tests on diverse FEM systems and an explicitly documented
accuracy-vs-work policy.
