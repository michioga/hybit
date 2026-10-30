# C1-G: Difficult SPD FEM corpus for convergence rescue / 収束救済検証

**Status:** 2026-10-30 design and acceptance plan; generator and full corpus **not yet implemented**.
**Priority:** numerical correctness > convergence rescue > bounded memory > runtime speed.
**Authority:** `docs/PROJECT_CHARTER.md` and `docs/HYBIT_DECISION_LOG.md` (ADR-010).

## 1. What HyBIT must prove

HyBIT is not primarily a faster-than-Jacobi solver on already-easy matrices.
Its distinctive objective is to **rescue an SPD sparse system for which a
memory-efficient iterative baseline fails to meet an independently checked
tolerance within the same disclosed iteration/resource budget**, by selectively
using bounded local direct corrections while retaining the global Krylov solve.

The strongest demonstrated case has:

1. A documented, independently defensible SPD matrix and fixed, nontrivial RHS.
2. Baseline Jacobi-PCG: `MaxIterations`, with original-CSR true residual above
   tolerance at the prescribed budget (or a separately classified breakdown).
3. Hybrid-PCG: `Converged`, independently verified true residual at or below
   the same tolerance, with actual local Cholesky regions and factor bytes.
4. The same initial guess, precision, stopping criteria, iteration ceiling,
   matrix, RHS, software/host environment and memory limit for both arms.
5. Transparent results across several predeclared iteration ceilings and a
   held-out matrix set, rather than cherry-picking a single low ceiling.

`MaxIterations` establishes failure **within a budget**, not mathematical
nonconvergence. A local direct correction is a **preconditioner**, not a direct
factorization of the global matrix; it does not guarantee convergence.
A converged recursive Krylov norm without an acceptable original-CSR residual
is never a rescue success.

## 2. Corpus families, built from easiest to hardest

| Family | Controlled difficulty | Question |
| --- | --- | --- |
| Local stiffness contrasts | Contrast 1e4, 1e6, 1e8 in isolated material/element patches | Can bounded local Cholesky cure a localized hard subspace? |
| Strongly anisotropic diffusion/elasticity | Direction-dependent coefficients and oriented meshes | Does region identification align with hard directions? |
| Thin/slender constrained FEM structures | Large element aspect ratios; properly constrained rigid-body modes | What is genuinely local and what is a global low-energy mode? |
| Multiple separated hard patches | Several distant difficult regions and varied overlap | Do selected regions, memory budgets and multistage escalation work? |
| Global low-frequency difficulty | Long-range smooth modes / weakly coupled substructures | Where are local factors insufficient and coarse correction necessary? |
| Easy/control and adversarial cases | `thermal1`-type benign solves; potentially invalid SPD input | Does Auto avoid unnecessary promotion and reject unsupported classes? |

Begin with **constructed SPD matrices whose positivity follows from their
construction**, for example a constrained SPD energy discretization with
strictly positive material coefficients and sufficiently fixed boundary DOFs.
A mere positive diagonal or successful PCG run **does not certify SPD**.
Store matrix generation parameters, boundary/constraint policy and a separate
class-validation record. Use independent symmetric checks and suitable
reference Cholesky/eigensolver checks on tractable cases, explicitly noting
that incomplete numerical spectral sampling is not a formal proof.

Next add actual FEM matrices, but distinguish synthetic/constructed systems
from physical FEM assemblies and manufactured loads from real application RHS.
Use at least a manufactured `b=A*x_ref` with nonconstant, localized and smooth
`x_ref` choices, plus physical RHS when available. Do not use only `x_ref=1`:
it can mask difficult modes. Keep zero-RHS trivial convergence outside rescue
statistics. Record matrix, RHS and generator/solver hashes.

## 3. Primary and secondary outcomes

**Primary outcome: convergence rescue.** Report the full 2x2 status table:

| Jacobi-PCG | HyBIT Hybrid-PCG | Interpretation |
| --- | --- | --- |
| Fails tolerance by declared budget | Converges with certified original-CSR residual | **Rescue success**, if local factors were actually used |
| Fails | Fails | Unrescued: inspect localized/global difficulty and failure type |
| Converges | Converges | Both valid; compare memory/time as secondary evidence |
| Converges | Fails | **Reliability regression** |

Count all attempted cases, timeout/error/breakdown and rejected SPD assumptions;
do not remove failed cases from the denominator. On a predeclared eligible
baseline-failure cohort, report rescue rate and its numerator/denominator,
while also publishing the separate baseline-success cohort's regression rate.
Do not merge an SPD numerical breakdown with exhaustion of iterations.

**Secondary:** additional factor and peak RSS bytes, number and distribution
of covered DOFs, escalation timing and stages, repeated RHS behavior, total
setup/solve cost and SpMV/prec timing. A slower rescue is still valuable if it
is the only correct solve within the available memory and iteration budget.
Time-to-solution determines whether a rescue policy is practical; speed alone
does not redefine the primary objective.

## 4. Fair protocol and gates

- Pre-register matrix families, seeds, RHS vectors, tolerances, budgets, region
  sizes `64/128/256/512`, caps and at least three iteration ceilings; record
  hardware, compiler, parallel settings and exact commit/dirty state.
- Check symmetry and the SPD contract **independently**; use PCG only for
  supported SPD cases. Classify suspected singular, indefinite or non-symmetric
  matrices separately rather than counting them as hard-SPD rescue failures.
- Require both status **and** `||b-Ax||_2 / ||b||_2` (proper zero-RHS handling).
  For manufactured RHS, also record forward error against `x_ref`; a small
  residual does not always imply a small forward error.
- Compare Jacobi-only, Hybrid, and an explicitly enabled algebraic/structural
  coarse option where mathematically appropriate. Include external AMG or
  small sparse-direct references only when installed, run and documented.
- Distinguish limited-iteration failure from non-finite arithmetic, invalid
  preconditioner, factorization failure, timeout and memory exhaustion.
- Keep a strict local-factor budget and record **peak process RSS**, not only
  persistent factor bytes. Use paired repetitions for speed but do not discard
  rescue cases merely because Hybrid is slower.
- Retain the current `thermal1` results as a **negative-control** example:
  `b=A*ones`, 82,654 DOF, 574,458 CSR nnz, Jacobi 1,286 iterations, opt-in
  one-stage Hybrid 1,305 iterations, both converged at `1e-8`. The C1-D nine-pair
  median relative cost of Hybrid was +19.7375%; the C1-F single-run profiler
  attributed much of the solver-time difference to preconditioner application.
  These measurements do **not** establish an SPD certificate, a physical load
  or a universal speed/routing rule.

## 5. Implementation sequence (proposed, not completed)

- **C1-G1 — construct and certify corpus:** parameterized FEM-like SPD systems,
  original matrix/RHS data and repeatable generation/validation metadata.
- **C1-G2 — rescue comparison:** same-budget Jacobi vs local-Hybrid vs optional
  coarse correction; record numerical truth and the complete contingency table.
- **C1-G3 — controller robustness:** evaluate genuine sustained difficulty vs
  transient slowdown, escalation timing, and bounded memory. `0.99` per-step
  progress limit remains an *experimental hypothesis*, not a production gate.
- **C1-G4 — held-out physical FEM:** independently verified real structural/
  thermal matrices and application RHS; audit rescue rate and failure modes.

**Production gate:** A candidate autonomy policy must demonstrate reproducible
rescue success on distinct held-out hard-SPD families, avoid unacceptable
regressions on solvable controls, preserve the original-CSR residual and class
contracts, respect memory caps, and allow conservative refusal to escalate.
Until then, retain existing behavior and label probes/thresholds experimental.

This document defines **how to assess future work**. It does not claim that
C1-G1--G4, global sparse-direct fallback, automatic MINRES, or a universal
robustness guarantee are implemented.
