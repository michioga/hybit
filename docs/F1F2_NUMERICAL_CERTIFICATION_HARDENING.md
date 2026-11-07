# F1/F2 — Resident PCG breakdown and CSR certification hardening

Status: **reviewed patch candidate for WSL acceptance, not yet built or run**.
Source baseline: `develop/0.10.0` at `7fb48f3` (C1-G3-A2 accepted).

## Scope

This focused change applies the third-party *v1* F1/F2 patch with source
anchors against the committed A2 checkpoint. The broader v2 patch is **not**
applied: it introduces scaling changes in several Krylov algorithms, with a
confirmed reciprocal-overflow edge case for subnormal floating-point values.

### F1 — Resident PCG mid-iteration non-finite residual

The default CPU `KrylovExecutionBackend::l2_norm` previously returned
`Err(NumericalBreakdown)` as soon as `dot(x, x)` overflowed to infinity.
This could bypass the caller's `Ok(Breakdown)` handling and the download of
the updated solution, unlike Rayon. The minimal fix passes the non-finite
norm as a value to the resident PCG controller, which can classify breakdown
and synchronize the solution. Initial non-finite norms still fail in
`checked_pcg_target`; dimensional and other backend errors remain errors.

Two regression tests cover fused update norms on CPU/Rayon and a
fault-injection operator that causes an overflow mid-iteration.

### F2 — Original-CSR final certification threshold

The certifier computes a target from absolute and relative tolerances and
the right-hand-side norm. If `relative_tolerance * norm_b` overflows, an
infinite threshold must not let a falsely converged report pass. Add
`!target.is_finite()` to the final demotion predicate. Two tests check the
overflow case and a valid `relative_tolerance > 1` control.

## Invariants and acceptance gates

- Exact source anchors: auto SHA256
  `9d61312ae9b2b62071fe061307144ad386a1753c12a65432539b2c275099eead`
  and resident execution SHA256
  `bade8469547478d87f2d1f2cfd45ef5b152d98c28af2579ab35281573d790309`.
- Exactly two Rust files change, plus this document, session handoff,
  MANIFEST and SOURCE_SHA256; no changes to solver default configuration.
- Preserve `benchmarks/scripts/__pycache__/` untracked and untouched.
- WSL Rust 1.99.1: `fmt --check`, strict SHA256, `git diff --check`,
  `hybit-auto`/`hybit-krylov` release tests, strict Clippy for those two
  crates and the facade example, build the facade benchmark.
- The reported original-CSR certification and real-FEM A2 behavior remain
  unchanged outside the two abnormal numerical conditions. Any discrepancy
  requires investigation before commit.

## Limits

- This fixes the reviewer-named F1/F2 cases only. It does not claim a
  completely scaled PCG norm for subnormal or supernormal inputs.
- Do not claim this patch was built until the developer's WSL output passes.
- The third-party review's CI/MPI/Python coverage recommendations remain
  separate work, as do A2 horizon instability and automatic rescue policy.
