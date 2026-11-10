# C1-H12-A10C4: Original-CSR Reliable Residual Replacement / Restart Pilot

## Research question

A10C1 found a large dot-product cancellation envelope for `multi_patch/ones`.
A10C2 reported deterministic low-energy Ritz and ABTM coarse-space diagnostics
(not globally certified spectral bounds). A10C3 separately found that the
independent textbook PCG reached a small recursive residual in twelve difficult
RHS/arm combinations, even though an original-CSR residual recomputation did
not meet `1e-12`. Production certified PCG did not silently accept these
false convergences.

A10C4 isolates whether **periodic reliable residual replacement accompanied
by a PCG restart** changes the outcome under controlled tolerance changes.
It does **not** update production PCG, its API, any AMG setup, or earlier evidence.

## Experimental design

- Matrix: frozen `c1g_p1_multi_patch_n144.mtx` (20,736 DOFs, 103,104 CSR NNZ).
- Hierarchies: `graph_piecewise`, `strong_graph_piecewise`,
  `strong_graph_smoothed`, each with two-level and native multilevel AMG.
- RHS: exactly `Ones` and `PointLoad`, the two A10C3 difficult modes.
- Relative tolerances: `1e-10`, `1e-11`, `1e-12`; 1,200 maximum iterations.
- Residual replacement period: `0` (no replacement), `25`, or `50` iterations.
- Six fresh OS processes, one per hierarchy/policy; 18 solves per process,
  **108 independent tolerance/replacement/RHS records** in total.
- Single thread. Original CSR `b-Ax` and separately compensated (Kahan)
  CSR residual are checked at iteration 0, 1, every 25 iterations, at any
  recursive tolerance crossing, and at the final iteration.
- For period 25/50: when a scheduled replacement occurs or recursive
  convergence is contradicted by original-CSR evidence, recompute `r=b-Ax`,
  then restart the PCG direction with `p=M^-1 r` and fresh `r·M^-1r`.
  The 0-period control retains the original A10C3 no-replacement recurrence
  and flags any false convergence instead of claiming success.
- A solution is `Converged` only if recursive, original-CSR, and Kahan
  relative residuals all meet the requested tolerance.

## Reproducibility and fail-closed gate

`C1H12_A10C4_reliable_gate.py` checks all 108 record identities, RHS hashes,
original FEM dimensions, hierarchy provenance, 25-step checkpoints,
preconditioner call counts, replacement and false-convergence event counts,
the three residual certificates and the entire GNU time schema. At `1e-12`
with replacement period `0`, it also requires agreement with all twelve
corresponding frozen A10C3 outcomes and residuals. The archived A10C3 CSV
hashes and both historical Git HEADs are checked independently.

Accepted output files, alongside immutable per-process `.log` and `.time`:

- `reliable_update_comparison.csv` — 108 outcomes, status, iteration count,
  true and recursive residuals, replacement work, gap and timing.
- `reliable_residual_trace.csv` — original-CSR checkpoints and replacement flags.
- `FROZEN_A10C3_SHA256.txt` — original comparison and checkpoint file hashes.

Execute after installing the five new research sources:

```bash
cd ~/hybit
A10C3="$HOME/c1h11_evidence/c1h12_a10c3_trace_4Mx3f0"
A10B="$HOME/c1h11_evidence/c1h12_a10b_prepared_oBIupI"

bash benchmarks/scripts/C1H12_A10C4_reliable_pilot.sh --dry-run "$A10C3" "$A10B"
bash benchmarks/scripts/C1H12_A10C4_reliable_pilot.sh --run "$A10C3" "$A10B"
```

## Interpretation boundaries

Passing the evidence gate proves data integrity and internal consistency of
this experiment, **not** convergence for every arm/RHS, production PCG
reliability, a global spectral bound, an IEEE precision floor, or an ABTM-only
speedup. Periodic restarts change the PCG recurrence and may substantially
increase iterations. A falsifiable improvement requires lower independently
certified residuals at comparable total cost, followed by repeat experiments
on larger real FEM matrices, a baseline comparison and solver-level design
review before any production changes.
