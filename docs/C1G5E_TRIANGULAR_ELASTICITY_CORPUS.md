# C1-G5E: independent triangular elasticity FEM research corpus and four-arm control

## Scope / source

- Checkpoint parent: `develop/0.10.0`, `04851d7317ae984e8ec15c81e6d7537d0b7144ec` (2026-11-14).
- This is a **synthetic finite-element** family from a *different governing PDE* and a different discretization than the previous C1-G4 scalar-diffusion constructions, **not external, user-submitted, or physically observed FEM**. It is still only one independent generator family.
- Host-runtime model selection remains **in `hybit-auto`**; `C1-G5C/D` embedded model continues to abstain and the original `hybit::solve()` is unchanged. Python and shell scripts here generate research data and audit measurements; they are never invoked in the numerical library runtime.

## Matrix construction and SPD argument

Three deterministic variants (`elastic_uniform`, `elastic_stiff_lens`, `elastic_shear_layers`) use a unit-square 41×41-node conforming triangular P1 mesh with alternating diagonals and controlled interior perturbations. Two displacement unknowns `(u_x,u_y)` per node yield 3,280 free DOFs after **both displacement components are clamped on the left boundary**. Each triangle uses a positive area and a plane-strain isotropic elasticity tensor with `E>0` and `-1<nu<0.5`.

\[
K = \sum_{e\in\mathcal T} A_e B_e^T D_e B_e,\qquad
u\in(-1,0.5),\; E_e>0.
\]

The associated strain-energy form is nonnegative and, on a conforming connected mesh with a fully fixed left edge, the only zero-strain displacement is zero (rigid-body modes are removed). Therefore the **assembled free-DOF stiffness is SPD by its FEM construction**, under the stated nondegeneracy and boundary conditions. This is a construction argument, not a general SPD test on arbitrary sparse matrices or proof that a PCG implementation is correct. The generator checks positive element areas, material admissibility, finite assembled entries and diagonal positivity. Numerical CSR/Kahan residual verification remains mandatory.

The three variants differ in material contrast and Poisson ratio; they are **not three independent training families**. The previous C1-G4 corpus plus this generator provides only two construction families; this is still inadequate for a production learned selector and no family-held-out promotion is authorized.

## Four-way measurement

A single generated Rust example, `crates/hybit/examples/c1g5e_four_arm_probe.rs`, contains a pinned copy of the **existing repository A9 native research Multilevel support**, plus four candidates using identical matrix loading, RHS generation, `tol=1e-10`, `max_iterations=600`, zero initial solution, one thread, original-CSR residual verification and independent process sampling:

| Candidate | Calculation | Notes |
|---|---|---|
| Jacobi-PCG | `JacobiPreconditioner` + A9 certified PCG wrapper | Baseline |
| Monitored Hybrid | `HybitSolver` with G5B monitor configuration | Can remain JacobiOnly; local factor usage recorded |
| Two-Level | A9-style `Graph/PiecewiseConstant` two-level preconditioner | Research control; coarse setup included |
| Native Multilevel | Original A9 3+ level V-cycle code | Not a production API or trained selector |

Four RHS per matrix: ones, deterministic index-smooth, xorshift seeded-random manufactured solutions, and sparse point-load (no manufactured reference). Four candidates each. **48 independent processes, 12 (matrix,RHS) groups**. Persist `.log`, `.time`, `.rc` files, original matrix SHA256 and RHS hashes, stdout and process errors. The same Rust experiment executable handles all four candidates. Process times/peak RSS use GNU `time` and candidate-specific `arm_work_ms` uses a consistently bounded Rust clock; internal setup/solve timing differs by API and must not be substituted for end-to-end measurements. Each arm is only sampled once. No ranking/learning is promoted.

## Numerical integrity and limitations

- `Converged` requires original CSR relative residual **and** independent compensated CSR residual <= requested tolerance. External forward-error labels are not provided for point-load.
- A failed setup or crashed child is recorded `PROCESS_ERROR`; it must not be called nonconvergence or silently omitted. The group is not complete for four-arm oracle use.
- Symmetric header / positive diagonal alone do not prove SPD. The FEM energy argument depends on the exact generator and constraints.
- **No user-provided physical load**, no mixed-field geometry, no independent industrial FEM mesh, no MPI, no cold/warm repeated trials, and no external hardware performance validity.
- Existing `C1-G4/G5B` timings are not appended to this new four-arm dataset as comparable data. Do not claim production speedup, generalization, or a trained predictor.
- Production `hybit::solve` / FFI / `hybit-auto` feature defaults are not modified.

## Gate and next step

1. Verify file hashes and build/run research-only tests with `C1G5E_install.sh --gate`.
2. Generate the pinned FEM matrices, run 48 experiments, audit all statuses with `C1G5E_run.sh`.
3. Inspect and freeze the owner-generated evidence before any Git commit. Keep missing/failed candidates visible.
4. Only then expand to **independently sourced external SPD-eligible FEM families with physical RHS**, establish matched reuse/cold/hot cost boundaries and family-held-out validation, and refine the Rust library's embedded/optional model on evidence. No automatic promotion from this single family.

## 2026-11-14 owner four-arm evidence acceptance

- The owner executed 48/48 candidate processes; 32 original-CSR/Kahan certified Converged, 16 not converged; 12 complete groups; original evidence and raw hashes frozen in `C1G5E_FOUR_ARM_EVIDENCE_20261114.md`. No production promotion.
