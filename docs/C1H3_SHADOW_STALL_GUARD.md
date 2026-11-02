# C1-H3 — ABTM residual shadow stall guard (offline, experimental)

Date: 2026-11-02. Host-neutral FEM scope; FrontISTR is the first real-FEM integration test, not a required dependency.

## Status / non-goals

C1-H2 observation-only PCG (commit `e1425b7`) already passed 15/15 bitwise-PCG invariance checks on five manufactured SPD P1 FEM pilot cases, each with `seed_fraction` 0.05, 0.2, and 0.5. In these data, `island_shell` at iterations 204 to 268, and `multi_patch` at 140 to 204, show true-residual spikes **followed by spontaneous recovery**. A scalar one-window threshold can make a false local-factor promotion.

This checkpoint implements an **offline shadow decision** on C1-H2 logs. It does **not** change the Rust library, PCG, ABTM structures, factorization, controller, FFI, or public API. It does **not** prove a new local-Hybrid rescue or make a production decision. No per-region localization or numerical coupling strength is inferred from aggregate mask counts.

## Shadow criteria (test thresholds, not tuned production defaults)

For consecutive checkpoints `i-1` and `i`, compute the geometric per-iteration residual rate `q = exp((log(r_i)-log(r_{i-1})) / (iteration_i-iteration_{i-1}))` from independently checked true residuals. A window is *weak* if `q >= 0.995`. An exactly zero residual is treated as good progress. Require **at least 2 consecutive weak windows**, not just a spike.

A weak second window is only a `shadow_candidate` when the current ABTM-derived hotspot summary also satisfies:

- fraction of Jacobi-scaled residual energy in hotspots >= 0.50;
- intersection/union Jaccard with preceding checkpoint >= 0.20;
- 3-checkpoint persistent DOFs / current hotspot DOFs >= 0.15.

The thresholds are illustrative and deliberately independent of the running solver. `shadow_candidate` means *review for possible more expensive diagnostics*, **not** "factorize now", "false alarm proven", or "rescue achieved". Cases with high residual but no localized persistent mask may still need coarse-space work; this detector is not a universal convergence classifier.

On the provided 15 C1-H2 logs, this offline rule observed **87 checkpoints**, **6 isolated weak windows**, **6 recoveries**, and **zero shadow candidates**. This is solely a negative-control outcome on 5 unique systems. To assess false negatives, the next step must include known, persistent, **unrescued** cases and local-Hybrid rescue-positive cases; evaluate both quality and setup/memory/solve time. The current five pilot cases all converge under uninterrupted Jacobi-PCG and cannot establish positive recall.

## Run

```bash
cd ~/hybit
python3 -B -m unittest discover -s benchmarks/scripts -p 'test_C1H3_shadow_policy.py' -v
python3 -B benchmarks/scripts/C1H3_shadow_policy.py \
  --logs "$HOME/c1h2_abtm_observer" --out "$HOME/c1h3_shadow" \
  --rate-limit 0.995 --require-corpus
# Or: bash benchmarks/scripts/C1H3_run_shadow.sh
```

`summary.csv`, `events.csv`, and `summary.json` are emitted under the output directory **outside the git checkout**. Inputs must report `PCG invariant : PASS` and have valid C1-H2 text headers, finite nonnegative residuals, coherent ABTM mask counts, and strictly increasing checkpoint iteration numbers; malformed data are rejected. Audit the residual trajectories before extrapolating any policy.

## Next experiment

Run the unchanged C1-H2 observer on more demanding FEM systems, including a deliberately difficult but **SPD-certified** real-FEM case. A successful candidate requires a paired trial: Jacobi-PCG fails under the same memory/iteration budget, local Cholesky Hybrid converges with externally recomputed residual and acceptable forward error where a reference exists. Compare the shadow decision against the actual rescued core region, factor memory, and time-to-solution. Separately, evaluate numerical coupling weights `|a_ij|/sqrt(a_ii a_jj)` and low-energy global modes (coarse correction), without assuming the unweighted ABTM graph can infer them.
