# C1-H12-A10C1 — independent CSR residual-precision diagnostic

Status: standalone research only; accept after owner WSL run and review.

## Why this trial exists

The frozen A10B 720-sample gate passed at commit `60104c9`, with exactly
300/360 converged samples per lifecycle. Every MaxIterations outcome was on
`multi_patch`, RHS `Ones` or `PointLoad`, at tolerance `1e-12` and 1200 steps.
For manufactured Ones, final forward error was approximately `1e-11` while
the independently verified relative residual remained around `1e-10`. This
may indicate an attainable-precision limitation from cancellation in `Ax`,
but it is not yet a proven explanation.

## What A10C1 measures

For each of 3 original FEM matrices and 4 A9 RHS modes, without new PCG runs:

1. Reconstruct canonical original CSR directly from Matrix Market, rejecting
   nonfinite or structurally asymmetric matrices and mismatching SHA-256.
2. Recreate the manufactured exact vector `x*` for Ones, IndexSmooth and
   SeededRandom; compare sequential CSR SpMV against per-row `math.fsum`.
3. Report `|| |A| |x*| ||_2 / ||A x*||_2` as a cancellation indicator and
   `gamma_(2k) * || |A| |x*| ||_2 / ||A x*||_2` as a conservative *model-based*
   rounding envelope, where `k` is maximum row length and `u=2^-53`.
4. Report Rayleigh quotient `x*^T A x* / (x*^T x*)` and diagonal-scaled
   quotient `x*^T A x* / (x*^T D x*)`. These describe selected vectors,
   **not** extremal eigenvalues or an AMG-preconditioned spectrum.
5. For PointLoad, report only `||b||_2` and frozen A10B solver outcomes;
   manufactured-error diagnostics are `n/a` because `x*` is unavailable.
6. Preserve frozen A10B case/RHS fingerprints and count exact lifecycle
   convergence outcomes. No frozen log or original matrix is modified.

The independent Python reconstruction can differ in the last bits from Rust
`sin/cos`, row reduction and canonical floating-point summation. Only the
trivial PointLoad RHS fingerprint is required to match bitwise. This trial
does not claim the model-based bound is a certified lower bound for attainable
PCG residual or prove ill conditioning. A later A10C2 numerical spectral
experiment will investigate low modes with explicit Ritz residual checks.

## Owner WSL execution

```bash
cd ~/hybit
A9="$HOME/c1h11_evidence/c1h12_a9_rhs_Xg80GE"
A10A="$HOME/c1h11_evidence/c1h12_a10a_accuracy_GH52NS"
A10B="$HOME/c1h11_evidence/c1h12_a10b_prepared_oBIupI"
bash benchmarks/scripts/C1H12_A10C1_precision_pilot.sh --dry-run "$A10B" "$A10A" "$A9"
bash benchmarks/scripts/C1H12_A10C1_precision_pilot.sh --run "$A10B" "$A10A" "$A9"
```

The runner requires current committed HEAD `60104c9`; frozen evidence HEAD
remains `1f3eaa5` by design. It produces 12 records in
`precision_floor.csv`, with an independent schema/provenance gate. Interpret
values before starting A10C2. No production PCG, public ABI, solver defaults,
or third-party libraries are touched. Do not commit/push this experiment until
the owner reviews the output.
