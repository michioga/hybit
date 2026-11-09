# C1-H12-A6: periodic original-CSR residual certification (candidate)

Status: **owner WSL pilot NOT RUN**. This is an opt-in scientific comparison,
not a production solver policy or a claim of speedup.

## Motivation and constrained change

The A2/A3 ABTM symbolic operations matched CSR, but their timings depend on
graph shape and representation setup. A4's native two-level PCG rescued
challenging FEM systems, yet it reported six false recursive convergences on
`multi_patch` at `tol=1e-10`. A5 corrected all six through independently
recomputed original-CSR residuals and restart, with 21/21 provenance and
numerical comparisons accepted (18 certified convergences, 3 MaxIterations).

A5 checked the true residual only after a PCG segment finished. A6 asks:

> Can checking every 16 or 32 global iterations and *selectively* restarting
> an inaccurate recurrence improve the number of certified iterations or
> reduce the expensive A5 late restarts, without degrading other systems?

A6 uses existing `hybit-krylov` stateful PCG start/continue functions from
the **example only**. Between true-residual checks it retains the PCG search
direction and scalar state (no hidden restart). It runs the same original
CSR and same Jacobi/two-level preconditioner, starting with `b=A*ones` and
`x=0`. Its recursive residual is compared with independently recomputed
`||b-A*x||/||b||`. If the ratio `true/recursive >=2` while true relative
residual is above `1e-10`, A6 safely starts a new PCG recurrence at the
current iterate. It also rejects any recursive convergence claim whose
true residual is too high. Every restart spends the *remaining* global
600-iteration budget, never a new budget. A zero-progress false recurrence
is reported as Breakdown, not an infinite loop.

Only `--certified-restart --cert-periodic-check {16,32}` enables A6. With
`--certified-restart` alone the A5 code remains unchanged. Neither flag
touches HyBIT's production default PCG, public API, C ABI or PETSc/hypre
dependencies. This is **not** a full multilevel AMG implementation.

## Owner WSL reproduction

Base repository must be `develop/0.10.0` HEAD `1f3eaa5` with A2–A5/FMT-FIX1
accepted and pushed. Apply the accompanying `C1H12_A6_apply.sh` candidate
with `--dry-run` before `--apply`; do not commit until numerical results
have been reviewed. Previous *unchanged* A5 raw evidence lives at
`~/c1h11_evidence/c1h12_a5_certified_QGyKD6`.

```bash
cd ~/hybit
BASE="$HOME/c1h11_evidence/c1h12_a5_certified_QGyKD6"

bash benchmarks/scripts/C1H12_A6_periodic_pcg.sh --dry-run "$BASE"
bash benchmarks/scripts/C1H12_A6_periodic_pcg.sh --run "$BASE"
```

The runner performs the Python negative tests, Rust example tests,
strict Clippy, rustfmt check and source SHA validation on the owner WSL;
then runs the same native Release executable for 3 matrices × 7 arms ×
3 policies = 63 processes:

- `a5`: original certified-restart control;
- `p16`: periodic original-CSR certificate every 16 iterations;
- `p32`: periodic original-CSR certificate every 32 iterations.

One thread (`RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), F64 transfer,
coarse dimension target 512, `tol=1e-10`, maximum 600 global PCG iterations.
Fresh evidence directory contains raw `.log` and GNU `.time` for all 63
processes, original matrix SHA manifest, source HEAD, configuration and
`periodic_pcg_comparison.csv`. **Historical A5 and new A6 HEADs differ by
design.** The runner requires identical original input checksums, not a
false same-HEAD equality between distinct snapshots.

`C1H12A4|` carries identical configuration and independent true residual
for all arms; A5 control uses `C1H12A5|`; A6 variants use `C1H12A6|`
including count/cause of restarts, gap ratio and time spent computing true
residuals. The Python gate rejects duplicated/missing rows, bad processes,
non-finite costs, provenance changes, over-budget iterations and any
`Converged` status exceeding the unchanged original-CSR 1e-10 gate.

**A negative result is evidence.** Compare certified convergence, total
iterations, proactive/rejected restarts, setup/solve ms and RSS, not merely
recursively reported tolerance. A single process repeat does **not**
support a speedup claim; repeat randomized paired orders after an
encouraging numerical outcome. Even successful A6 certification would
remain a two-level preconditioner experiment, not PETSc/GAMG equivalence.

## Decision gates

1. `cargo +1.99.1 fmt --all -- --check`, strict example Clippy and example
   test suite PASS on owner WSL.
2. All 63 log/time pairs present and complete, original matrix SHA unchanged.
3. Every convergence certified independently; no false convergence accepted.
4. Report new `multi_patch/graph_smoothed` and other hard-arm results against
   same-binary A5 control, including any regression or Breakdown.
5. Only after seeing owner data, decide whether to refine the threshold,
   check interval or underlying multilevel hierarchy. No automatic commit.
