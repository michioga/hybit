# HyBIT 0.10.0 — C1-G5C library-resident selection-shadow boundary

**Scope:** opt-in Rust implementation in `hybit-auto`, surfaced through the `hybit` facade. This is NOT a trained selector, NOT a production default, and NOT a claim of numerical superiority. Source base: committed `develop/0.10.0` HEAD `e6e96a7`.

## Architectural correction

The corpus is development/validation evidence, not a mandatory set of runtime CSV/JSON/Python files. **Selection and switching must happen in HyBIT Rust library code**, from online-observable features. Packaging, Python and shell scripts are permitted for offline evidence generation, build/test gates and source integration only; they must never run within `hybit::solve`.

This checkpoint adds a feature-gated library API `HybitSolver::shadow_select_csr32` and `HybitSolver::solve_csr32_with_selection_shadow`, with an exported `ShadowSelectionObservation`. It extracts input features from the real CSR/RHS in the library (not from a matrix name/hash), keeps `expected_rhs_reuse` and an explicit local-factor memory budget, and returns a reason for abstention. The second API executes the existing solve method unchanged and provides the shadow observation separately. No modification of `SolveReport`, default routing, PCG arithmetic, restart logic, ABI or dependency tree.

`C1G5C_EVIDENCE_20261114` compiles the presently established corpus **qualification state**: 72 A9 measurements, 24 Jacobi/monitored-Hybrid measurements, one measured matrix-generator family, two demonstrated Hybrid regressions, zero local rescues, no independent family-held-out pass. That evidence is *actually read by the library API* and included in its output; it is intentionally **not converted into a guessed performance model**. Until validated numerical, memory and total-cost gates exist, `recommended=None` for every case.

### API example (with `c1g5c-shadow` feature)

```rust
use hybit::{Csr32Matrix, HybitSolver};
let a = Csr32Matrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1],
                        vec![3.0, -1.0, -1.0, 3.0])?;
let b = [1.0, 1.0];
let mut x = [0.0; 2];
let (solve_report, shadow) = HybitSolver::new()
    .solve_csr32_with_selection_shadow(&a, &b, &mut x, 1)?;
assert!(shadow.recommended.is_none());
# let _ = solve_report;
# Ok::<(), hybit::HybitError>(())
```

The real library `HybitSolver::solve_csr32`, facade `hybit::solve`, default Cargo features and FFI stay unchanged. **This is the first in-library observation boundary**, not yet automatic selection from a qualified corpus; it does not support a trained cost model or dynamic promotion.

## Gate and next steps

- Both feature-disabled and feature-enabled builds/tests must pass. Feature-enabled strict Rust Clippy must pass. Compare numerical results with/without the shadow API on the same deterministic SPD matrix.
- A future corpus builder must freeze **runtime-extractable** feature vectors and numerical/memory outcomes for independent matrix *families*, not merely RHS variants; training labels cannot be passed to inference.
- Only after held-out family tests, repeated common-harness end-to-end cost comparisons, and memory/accuracy checks may a frozen Rust selector table/model return `Some(candidate)`. The model shall be included in the Rust crate, not loaded from a script at runtime. Production switching requires separate bounded PCG-restart and rollback tests.
- The right place for a future selector is the high-level `HybitSolver` analyze/prepare/solve flow; do not make examples or shell scripts responsible for application-level routing.

## Original limitations retained

SPD remains a caller-declared contract; a symmetric header or positive diagonal does not prove SPD. Manufactured forward error is offline-only. G4 and G5B used distinct timing scopes and cannot furnish an unbiased four-way end-to-end oracle. Native multilevel is currently research-only, not a supported production route.
