# C1-H12-A10C6 — Observe-only stagnation continuation

## Purpose and research question

A10C5 achieved a large reduction in original-CSR residual replacements, but
each cooldown lost one `1e-10` convergence that the matched A10C4 control
had certified. Its early-stop heuristic (`Stagnated`) and the altered
cooldown/restart trajectory must be disentangled before production changes.

This A10C6 experiment preserves the existing cooldowns (25/50), progress
factor (0.9), stagnation observation window (200 iterations), one-thread
execution, original FEM CSR input, RHS generation, hierarchy configuration,
maximum 1200 iterations, and original-CSR/Kahan certificates.

The diagnostic PCG runs **exactly as A10C5 up through its first stagnation
candidate checkpoint**. At that checkpoint it records the candidate, defers
the restart (matching A10C5's decision), but **does not stop**. After that,
normal cooldown-governed residual replacements may resume and PCG continues
until certified convergence, the 1200-iteration budget, or breakdown.

`StagnationCandidate` is observational. It is neither a numerical failure
verdict nor proof that the requested tolerance is unattainable.

## Frozen controls and isolation

Inputs (not modified):

- A10B original-CSR certified 720-sample comparison and matrix SHA.
- A10C4 108-outcome reliable-update comparison and raw logs.
- A10C5 72-outcome adaptive comparison, 2146 original-CSR checkpoints,
  all raw logs, source provenance, and original controls.

The preflight requires historical committed HEAD `60104c9` on
`develop/0.10.0`, no staged changes, source-manifest SHA verification, and
identical frozen matrix and commit provenance. The installer creates **five
new research files only**, updating MANIFEST and SOURCE_SHA256. Production
PCG, public API/ABI, defaults, and frozen evidence are unchanged. No
commit/push is performed.

## Experimental grid and comparison

The same original `multi_patch` FEM matrix, three ABTM hierarchy arms,
two policy arms, two difficult RHS modes, three tolerances (1e-10/1e-11/1e-12),
and two cooldowns (25/50) produce **six independent OS processes and 72
outcomes**. Each is inspected with original-CSR and Kahan true residuals,
recursive residual and 25-iteration checkpoints (plus the original threshold
verification cadence). Rust tests and Clippy are performed in owner WSL.

The independent evidence gate:

1. Validates frozen A10C5 raw logs and its two CSVs against the frozen
   A10C4/A10B provenance; it never rewrites them.
2. Requires exact numerical checkpoint prefix equality to A10C5 through
   its terminal checkpoint, including the *first* stagnation candidate.
3. Requires every frozen A10C5 certified convergence to remain certified
   with an identical number of iterations.
4. For A10C5 `Stagnated` cases, requires the recorded candidate to be at
   the frozen stop point and A10C6 to run beyond it.
5. Recomputes the entire cooldown, replacement, suppression and progress
   accounting from independently logged original-CSR certificates.
6. Rejects fabricated success, premature `MaxIterations`, `Stagnated` as
   a terminal outcome, missing/duplicate checkpoints, provenance changes,
   nonzero GNU time exit status, and mismatched first-candidate markers.
7. **Explicitly compares convergence against A10C4**. Any regression is
   reported per cooldown; an integrity PASS is not an accuracy-regression
   PASS. A10C4 has 18/36 convergences at each matched 25/50 period, versus
   17/36 for the frozen A10C5 experiment.

Outputs are `shadow_comparison.csv` and `shadow_residual_trace.csv` in a
new immutable-by-convention evidence directory
`~/c1h11_evidence/c1h12_a10c6_shadow_XXXXXX`.

## Interpretation

- Recovery after the first candidate would directly demonstrate that the
  A10C5 early-stop decision could discard subsequent useful convergence.
- No recovery in an A10C4-convergent case would favor a cooldown/recurrence
  sensitivity hypothesis, though it would not itself prove causation.
- One-run whole-process elapsed times include setup and all 12 outcomes per
  process. Do not infer per-condition speedup or a production-level guarantee.
- Even if all regressions disappear, further controls and independent FEM
  validation are required before touching the real PCG implementation.
