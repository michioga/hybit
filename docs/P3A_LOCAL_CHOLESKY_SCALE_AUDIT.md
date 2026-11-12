# P3-A: Local Cholesky scale-invariance audit (research; not production change)

**Source checkpoint:** `develop/0.10.0` at `fdb047e`. Apply after P2-B FIX1 on WSL2. The existing three tracked P2-B FIX1 modifications and unrelated untracked paths are preserved. P2-B paired owner pilot at `~/c1h11_evidence/p2b_paired_ayLJ1D` reported 3/3 OFF and ON certified convergence, no natural feature-triggered rescue, and no convergence regression; zero RHS + nonzero initial `x` + relative-only stopping remained `MaxIterations` at 96 steps. These observations do **not** establish that the P2-A retry was exercised naturally.

## Question

`LocalCholeskyRegion::from_csr32` in `hybit-precond` currently computes `symmetry_tol = 1e-11 * scale.max(1.0)` and `pivot_tol = 1e-14 * scale.max(1.0)`, where scale is the maximum local matrix entry magnitude. Consequently the tolerance is **not homogeneous** in a matrix-wide positive scalar and can be too permissive for symmetry, or too restrictive for positive Cholesky pivots, at small/heterogeneous scales.

P3-A adds only a research example, a machine-readable gate, unit tests and this document. No changes to `LocalCholeskyRegion`, C/Fortran ABI, PCG, ABTM, policies or defaults. The data-integrity gate validates records; *unfavorable numerical observations remain findings*, not a reason to discard a case.

## Mathematically classified cases

For the symmetric cases, `A = s D T D`, with `s>0`, diagonal `D` strictly positive, and `T = tridiag(-1,2,-1)` (4 x 4). `T` has positive eigenvalues, hence these test matrices are SPD by congruence. Their SPD classification does not rely on the results of HyBIT. Five cases: unit; uniform scales `1e-12`, `1e-16`, `1e16`; heterogeneous diagonal congruence `D = diag(1e-8,1,1,1e8)`. For nonsymmetric controls, only `A(0,1)` is perturbed by 50%: at scale 1 and at scale `1e-12`. A local symmetric Cholesky factor must not silently accept the nonsymmetric input as SPD.

The manufactured `x_ref=[1,-0.75,1.25,-0.5]` and `b=A*x_ref` are common within each case. A full-region `HybridPreconditioner` has one region, so its apply approximates `A^{-1} b`; separately recomputed scaled-norm relative residual and forward error check accepted results. The audit distinguishes `observed=accept/reject` and the **analytical** `expected=accept/reject`, with all cases retained. No global FEM rescue claim is made from 4-DOF tests.

## WSL2 qualification (after applying package)

```bash
cd ~/hybit
bash /mnt/d/Work/HyBIT_0_10_P3A_SCALE_AUDIT_20261112/P3A_validate.sh "$PWD"
```

The script checks repository provenance, full Source SHA-256, Python data-integrity tests, Rust 1.99.1 example unit tests, strict Clippy/rustfmt, builds and executes the probe, stores full logs and hashes at `~/c1h11_evidence/p3a_local_scale_XXXXXX`, and parses the results. It does not commit/push, modify production numerical code, rerun P2-B, or touch the two unrelated untracked paths.

## Acceptance and next decision

- **Measurement gate PASS:** exact seven records, finite numeric observations, proper treatment of rejects, and summary agreement. A PASS here **does not mean all seven numerical contracts PASS**.
- **Risk finding:** an SPD case rejected solely due to scale/conditioning implies the current local-direct path is not scale robust; a materially nonsymmetric case accepted implies an inadequate symmetry guard. Either finding motivates a narrowly scoped P3-B fix and a held-out FEM regression study.
- **P3-B design candidates:** a scale-relative symmetry check (not an absolute floor of 1), diagonal equilibration for local factorization (additional persistent O(n) scaling/memory and apply accounting), finite-pivot and backward-error checks, and compatibility with overlapping Schwarz SPD. Do not blindly remove pivot guards or declare arbitrary ill-conditioned matrices safely factorable.
- Do not change production defaults or publish release before actual WSL numerical results and an independent P3-B acceptance test.
