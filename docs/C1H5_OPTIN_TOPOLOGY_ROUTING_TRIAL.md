# C1-H5 — Opt-in topology-aware routing trial (independent Rust example)

2026-11-02. **FEM-host-neutral.** This is **not** `HybitSolver` production Auto,
not a new public API, not a default change, and not a demonstrated real-FEM
speedup. Source checkpoint expected: `develop/0.10.0` at `457eb1f`.

## Motivation and evidence

- C1-H3: 15 independent *runs* on 5 related FEM pilot matrices × 3 hotspot
  thresholds; all ultimately converged with plain Jacobi-PCG. Shadow stall
  guard recorded 0 hypothetical promotions; the six temporary bad intervals
  arise from only two underlying matrices, each at three thresholds.
- C1-H4: manual intervention on a *constructed* 160 DOF SPD example. Under
  30-iteration budget, Jacobi-PCG was `MaxIterations`, true residual
  `2.557405006e-7`; a 12-step probe plus the selected 96-DOF local packed
  Cholesky preconditioner converged at a combined 14 iterations, true residual
  `2.079880944e-14`, relative forward error `1.456794450e-13`.
  A globally coupled 160-DOF control did **not** benefit sufficiently from
  the same size-limited local factor.

## Experimental controller (new example only)

`crates/hybit/examples/c1h5_topology_routing_trial.rs` copies the fixed
construction/graph exploration of the C1-H4 positive and negative controls.
The binary is parameter-free and deterministic, but timer values fluctuate.

1. Run a resumable Jacobi-PCG probe of exactly 12 steps, preserving the live
   recurrence using `hybit_krylov::pcg_start_with_workspace` and
   `pcg_continue_with_workspace`.
2. Compute the original CSR true residual. Seed at the maximum
   `abs(r_i)/sqrt(A_ii)`. Construct ABTM topology. Expand the connected
   numerical strong-coupling neighborhood (threshold **0.1**) capped at
   **96 DOFs**.
3. Record numerical boundary cuts between the selected region and its
   complement. A **synthetic-specific conservative early gate** admits local
   factorization only if: region contains exactly 96 DOFs; **no strong
   cut**; at least **one weak nonzero cut**; conservative dense `8n²`
   memory estimate is at most **128 KiB**. These conditions are empirical,
   not necessary/sufficient criteria for a correct decision on general FEM.
4. **Promote**: factor the selected region with the existing
   `HybridPreconditioner::from_csr32` and PCG-restart because the
   preconditioner changed. Allow at most 18 extra Krylov steps. Count all
   12+stage iterations and factor bytes; verify original-CSR relative
   residual <=1e-10 and reference forward error <=1e-6.
5. **Keep PCG**: perform no factorization or restart. Continue the SAME
   live `PcgSession` for the remaining 18 steps. Require bitwise equality
   of all final `f64` solution entries, identical iteration count and status
   compared to an independent uninterrupted 30-step Jacobi-PCG reference.

The program fails nonzero unless the *constructed* local and global cases
pass their two separate gates. The C1-H3 healthy-case logs are tested by a
**separate** Python gate; that corpus does not provide weighted cuts and
therefore is not a substitute for actual `HybitSolver` default-invariance
validation on the same matrices.

## Run

```bash
cd ~/hybit
bash benchmarks/scripts/C1H5_run_routing_trial.sh
```

Prerequisites: prior C1-H2 observation logs at
`~/c1h2_abtm_observer/*_fraction*.log` (15) and current repository with
C1-H3 scripts available.

## Acceptance and limitations

- Local rescue: baseline above tolerance after 30 iterations, routed case
  converged within 30 with **true residual** and **known-reference error**
  both acceptable; actual local factor bytes >0.
- Global control: routing chooses `KEEP_PCG`, factors zero regions, exact
  bitwise final solution and status/iteration equivalence.
- Healthy C1-H3 negative corpus: 15/15 read, 0 shadow candidates.
- Standard `cargo fmt --all -- --check`, Release workspace tests, Clippy
  `-D warnings`, SHA manifest and `git diff --check` must pass before commit.
- **Not demonstrated:** in-core Auto routing, robust decision on real FEM,
  differential benefit of ABTM versus CSR graph construction, predictive
  classification on held-out difficult matrices, amortized wall-clock gain,
  SPD class validation for imported matrices.
- The positive outcome is a *synthetic selection proof-of-concept*, not a
  generally valid online classifier. In particular, region cardinality 96,
  the 0.1 coupling threshold and weak-boundary signature are properties of
  the constructed example; they must not be hardcoded as default Auto rules.

## Following experiments

C1-H6 should expose an opt-in, inert-by-default gate in `hybit-auto` once
independent held-out positive/negative examples exist, preserving the exact
`PcgSession` without promotion, and restarting PCG only on changed
preconditioners. Compare with `HybridOptions::algebraic_coarse`, 128/256/512
local caps, factor bytes, and total latency. FrontISTR remains only the
first intended host for imported real-FEM comparisons.
