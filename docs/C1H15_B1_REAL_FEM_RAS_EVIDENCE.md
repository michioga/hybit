# C1-H15-B1 — Real FEM RAS evidence and class-aware local direct strategy

Date: 2026-11-05. **Experimental evidence / design decision**, not a
production performance claim. No changes to HyBIT public API, automatic
routing, or existing SPD local Cholesky implementation are authorized by this
document.

## Provenance and gates

- User-side WSL2 repository: `develop/0.10.0` at base HEAD `dc49e95`, with
  uncommitted C1-H15-A example and integrity metadata changes.
- Example: `crates/hybit/examples/c1h15_schwarz_probe.rs`.
- Verified post-rustfmt example SHA256:
  `7df59796785df17e1ab721f233d98eba318ca238e7402f514344c8ca70ccd54a`.
- Rust 1.99.1: `fmt --check`, five example tests, release Clippy with
  `-D warnings`, release build, `git diff --check`, and `SOURCE_SHA256.txt`
  integrity all PASS in the user's WSL environment.
- `11_contact_2beam` logs and input SHA256 files:
  `~/c1h11_evidence/c1h15_b1_contact_afTqJx/`.
- `07_viscoelastic_cylinder` / `08_creep_cylinder` logs and input SHA256:
  `~/c1h11_evidence/c1h15_screen_WwY5De/`.
- Input matrix/RHS SHA256 values are retained in those evidence directories;
  they were not copied into this document without checking the full files.

## Contact case: measured results

Matrix `11_contact_2beam`, n=756, CSR nnz=28,960. Common FGMRES settings:
`--tol 1e-10 --max-iters 360`; all independently verified true residuals.

| Preconditioner | Restart | Status | Iterations | Verified relative residual | Setup ms | Solver ms | Factor bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ILU0-ABTM | 30 | MaxIterations | 360 | 1.318012270033e-8 | 1.493775 | 23.660969 | 353,572 |
| RAS-LU, core32, overlap0 | 30 | MaxIterations | 360 | 1.893700512284e-5 | 0.563130 | 31.565940 | 181,216 |
| RAS-LU, core32, overlap1 | 30 | MaxIterations | 360 | 1.121944120252e-8 | 3.092811 | 157.114263 | 1,504,208 |
| ILU0-ABTM | 70 | Converged | 54 | 3.068967556521e-11 | 1.481645 | 4.848572 | 353,572 |
| RAS-LU, core32, overlap1 | 70 | Converged | 58 | 6.690243885406e-11 | 3.021041 | 27.624995 | 1,504,208 |

RAS-LU overlap0: 63 regions, maximum 32 rows, ABTM dual topology 94,184
metadata bytes. RAS-LU overlap1: also 63 regions, maximum 108 rows, same
topology metadata. Dense LU bytes exclude graph, matrix, indices and scratch;
therefore they **must not** be interpreted as total process memory.

Interpretation:

1. One-hop overlap improved RAS-LU's restart30 true residual by about 1,688
   times, showing interface coupling matters. It did **not** attain `1e-10`.
2. Changing restart from 30 to 70 enabled **both** ILU0-ABTM and RAS-LU to
   converge. For this case, a restart increase is a much cheaper successful
   rescue than global dense local direct reinforcement.
3. At restart70, ILU0-ABTM needed 54 rather than 58 iterations, had about
   5.7x lower solver time, and about 4.25x lower factor storage than RAS-LU.
4. A small 756-DOF example and single wall-time samples are not sufficient
   for general speedup or memory claims. Sparse local LU may perform very
   differently from this bounded dense reference.

## Further small-case screening

Both cases use n=1,887, CSR nnz=94,379, ILU0-ABTM preconditioner.

| Case | Restart | Status | Iterations | Verified relative residual | Solver ms |
| --- | ---: | --- | ---: | ---: | ---: |
| 07_viscoelastic_cylinder | 30 | Converged | 31 | 8.977928364399e-11 | 6.201445 |
| 07_viscoelastic_cylinder | 70 | Converged | 31 | 6.688261734456e-11 | 7.263345 |
| 08_creep_cylinder | 30 | Converged | 31 | 7.629234637198e-11 | 6.302796 |
| 08_creep_cylinder | 70 | Converged | 31 | 5.551152503835e-11 | 6.458153 |

Both are **already solvable** under the original restart budget, so neither
provides a genuine convergence-rescue case for a new preconditioner.

## Design decision: do not replace Cholesky with LU

The FrontISTR 18-matrix audit classified 17 as non-symmetric and one as
approximately symmetric but **SPD unverified**. Positive diagonal entries
are not proof of SPD, and a principal submatrix of a non-symmetric matrix
cannot automatically be Cholesky-factorized.

Keep factorization selected by the **declared mathematical class**:

| Class | Krylov family | Local numeric factor | Contract |
| --- | --- | --- | --- |
| SPD | PCG + existing hybrid | Cholesky | Preserve symmetric positive preconditioner and existing weighted overlap contract. |
| GeneralSquare | FGMRES | Robust pivoted LU or ILU | Nonsymmetric corrections allowed; verify true residual. |
| SymmetricIndefinite | Future MINRES or suitable alternative | Pivoted LDL^T candidate | Not yet implemented or validated. |

ABTM graph/bitmap topology can be shared for region discovery, overlap,
local index mapping, and symbolic preparation. Factorization backend depends
on class. No automatic, generic local Cholesky assumption is permitted for
the sampled FrontISTR GeneralSquare matrices.

## Proposed next experiment: selective local correction, not uniform RAS

1. Retain class-appropriate low-memory baseline preconditioner. Compare
   fixed restart against a **bounded** restart expansion before promoting
   to direct reinforcement; include its Krylov-basis memory cost.
2. Verify true residual at restart boundaries. Only on a documented stall,
   construct residual-weighted **connected** graph regions using ABTM and
   finite row/byte budgets; top-k rows alone are not a connected region.
3. GeneralSquare reference candidate: ILU0-ABTM base correction `z0`, then
   a restricted local LU solve of the remaining defect
   `d = r - A z0`, giving
   `z = z0 + R_core^T A_region,region^(-1) R_region d`.
   This is a **proposed** multiplicative correction, not yet implemented or
   shown to accelerate convergence. The extra SpMV/work must be measured.
4. If correction regions or strengths vary by iteration, FGMRES can support
   a flexible preconditioner; avoid silently changing PCG's fixed-SPD
   preconditioner contract. For SPD, reuse the existing symmetric local
   Cholesky/hybrid formulation rather than transplanting nonsymmetric RAS.
5. Compare against ILU0-ABTM with the **same total iteration, restart,
   time and peak-memory budgets**. Count only true verified convergence
   at tolerance as a rescue. Keep preparation/reuse and total RSS visible.

`01_elastic_hinge` is a difficult large candidate (252,168 DOF, prior
restart30/360 ILU0-ABTM true residual about 0.189), but its residual is
diffuse and requires a separate bounded-design review. The dense C1-H15-A
reference has a hard 10,000-row cap and must not be used for that matrix.

**Next prioritized action:** archive this verified checkpoint, then screen
harder real-FEM cases under bounded adaptive restart *before* investing in
new local-direct production code.
