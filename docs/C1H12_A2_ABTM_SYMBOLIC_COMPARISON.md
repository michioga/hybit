# C1-H12-A2: Native HyBIT ABTM symbolic comparison (experimental)

Date: 2026-11-09 (JST). Baseline `develop/0.10.0` at `4593c20`.

## Purpose and scope

Investigate whether ABTM bitwise topology unions accelerate repeated symbolic
work required by a future **native Rust AMG hierarchy**. PETSc/hypre supply
external comparison data only: **neither is linked or called** by HyBIT.

The experiment is isolated in `crates/hybit/examples/c1h12_a2_abtm_symbolic_probe.rs`.
It does **not** alter production PCG, aggregation, preconditioner selection,
the public API, or numerical solve behavior.

For each scalar FEM CSR matrix, the probe computes a shared normalized
off-diagonal strength graph. For each row `i`, an edge `i -> j` is selected
when `abs(A_ij) / sqrt(A_ii A_jj)` reaches at least 0.25 of that row's
maximum strength. The threshold is experimental, not a universal AMG
strength rule. Positive diagonal is required but is **not** an SPD proof.

The two arms compute the exact same two-hop support:

`N2(i) = union { N(j) : j in N(i) }`.

- **CSR/list arm:** concatenate adjacent neighbor lists; sort and deduplicate.
- **ABTM arm:** iterate compressed topology words and OR 64-bit masks;
  enumerate their set bits in ascending order.

The two-hop support models a relevant *symbolic* graph-union primitive but is
**not equivalent to the exact nonzero pattern of `P^T A P`** (that needs a
specific interpolation `P`, cancellations and numerical assembly). This
experiment must not be cited as a Galerkin speedup or solver convergence gain.

## Verification and measurements

The probe rejects mismatched strong graph structure, two-hop row supports,
or non-finite strengths. Synthetic Rust tests cover 64-bit boundaries,
disconnected vertices and rows with no strong neighbors.

For each of three C1-G hard SPD P1 diffusion matrices (n=20,736), run
five measurements per arm and one warmup, printing medians:

- Shared numeric `strength_ms`, once per matrix.
- ABTM staging `graph_csr_prepare_ms` (temporary structural CSR) and
  `abtm_conversion_ms` (CSR-to-ABTM topology), explicitly separate.
- `csr_symbolic_ms` and `abtm_symbolic_ms` for the same two-hop union.
- CSR adjacency list capacity bytes, ABTM metadata bytes, nonempty words,
  exact output nnz and a deterministic output checksum.
- Process elapsed wall and peak RSS (including Matrix Market input).

The ABTM candidate pays **both** temporary CSR preparation and topology
conversion costs. Its
break-even number of repeated uses is only meaningful if `abtm_symbolic_ms`
is consistently below `csr_symbolic_ms`; otherwise there is no break-even.
The reported CSR and ABTM storage bytes have different ownership/lifetime
and are **not directly comparable peak RSS figures**.

## WSL acceptance (Ubuntu 26.04, Rust 1.99.1)

```bash
cd ~/hybit
bash benchmarks/scripts/C1H12_A2_abtm_symbolic.sh --dry-run
bash benchmarks/scripts/C1H12_A2_abtm_symbolic.sh --run
```

Run on the exact `4593c20` base with only this uncommitted comparison patch
applied. It invokes source SHA, rustfmt, example tests, strict Clippy, release
build and then the three real corpus cases, storing raw logs and timings in a
new `~/c1h11_evidence/c1h12_a2_symbolic_XXXXXX/` directory.

## Decision rule / rollback

Correctness must pass in all three cases. Compare median kernel time **and**
conversion/setup overhead before recommending integration. If ABTM is slower
or has unfavorable peak memory, retain the existing CSR symbolic route.
The production code remains unchanged, so rollback is simply leaving this
experimental example disabled or removing its isolated files in a subsequent
reviewed change. No commit, push, release or automatic AMG selection is
authorized by this note.
