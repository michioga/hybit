# C1-H12-A10C3: original-CSR residual-gap trace (standalone research)

## Question

A10B's 720 samples preserved 300/360 convergences under both fresh and reused
prepared hierarchies, with a consistent `multi_patch` failure for `Ones` and
`PointLoad`. A10C1 showed severe cancellation on `multi_patch/Ones`; A10C2
measured low Ritz candidates and ABTM aggregate-space projections. These
results do **not** identify why PCG stalls. A10C3 checks how far a recursive
PCG residual can drift from independently reconstructed original-CSR residuals.

## Experimental isolation

One original `multi_patch` FEM matrix (20736 rows, 103104 CSR nonzeros), three
frozen A10B aggregate/basis arms (`graph_piecewise`, `strong_graph_piecewise`,
`strong_graph_smoothed`), both A10B preconditioners (additive two-level and
native multilevel symmetric V-cycle) and four deterministic RHS modes yield
**6 processes / 24 paired solves**. Every solve begins from zero, uses a fixed
relative tolerance of `1e-12`, a maximum of 1200 iterations, and one thread.

For each RHS, a **separate textbook left-PCG implementation** with no residual
replacement emits the relative recursive residual `||r_k||/||b||` and a fresh
`||b-Ax_k||/||b||` from the original CSR at steps 0, 1, multiples of 25 and
termination. The second certificate accumulates each original CSR row with
Kahan compensation. The ratio of conventional true to recursive relative
residuals is included to expose gaps.

The same prepared preconditioner is then used for a **different** solve through
the A10B-copied `solve_certified`, which calls HyBIT's existing production PCG
with final true-residual checks and possible restarts. Its status, RHS
fingerprint and iteration count must reproduce the frozen A10B samples. These
two PCG implementations do not have identical recurrences, summation order or
restart logic; trace checkpoints cannot be attributed to production PCG.

## Required gates

1. Historical HEAD provenance: installed checkpoint `60104c9`, frozen A10B
   HEAD `1f3eaa5` (full hashes), frozen A10C1 and A10C2 HEAD `60104c9`.
2. Original FEM matrix SHA manifests agree byte for byte, and frozen A10C2
   Ritz/projection evidence and A10C1 precision evidence are present.
3. Exactly six process logs and time records contain 24 results; four RHS
   fingerprints, outcomes and iteration counts match the accepted frozen A10B
   CSV for each policy and arm.
4. Checkpoints contain a complete ordered cadence, valid work counts, finite
   original-CSR/Kahan measurements, consistent residual-gap ratios and no
   uncertified convergence. Invalid timing, missing entries and provenance
   edits fail closed.
5. Separate `residual_trace.csv` (all checkpoints) and `trace_comparison.csv`
   (24 outcomes) are written **only after** validation of every process.

The gate records `FalseConvergence` as a diagnostic state and does not silently
change it into success. A breakdown likewise requires a final original-CSR
checkpoint. The purpose is to understand the precision and conditioning
problem before changing the preconditioner or production stopping rule.

## Owner-machine commands (WSL2 Ubuntu)

After installing the ZIP payload using its apply script:

```bash
cd ~/hybit
A10B="$HOME/c1h11_evidence/c1h12_a10b_prepared_oBIupI"
A10C1="$HOME/c1h11_evidence/c1h12_a10c1_precision_sbnkOd"
A10C2="$HOME/c1h11_evidence/c1h12_a10c2_spectral_8SEN1T"

bash benchmarks/scripts/C1H12_A10C3_trace_pilot.sh --dry-run "$A10C2" "$A10C1" "$A10B"
bash benchmarks/scripts/C1H12_A10C3_trace_pilot.sh --run "$A10C2" "$A10C1" "$A10B"
```

Do not commit or push until the 24/24 independent trace gate and Rust build,
tests and Clippy all pass. This prototype is neither proof of a production
precision floor nor a claim of multilevel or ABTM-only speedup.
