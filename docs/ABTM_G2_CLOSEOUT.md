# ABTM G2 closeout: metadata-first product pruning

Status: **validated / closed for the 0.8 development checkpoint**

G2 asked whether ABTM topology can eliminate numerical work before value access,
whether row/column topology should be prepared in both orientations, and which
numerical physical layout is appropriate for CPU sparse row-by-column products.

The answer is not one universal storage format. The evidence supports a
metadata-first topology layer, adaptive numerical execution, and an explicit
fallback for structures where ABTM does not pay for itself.

## Experiment sequence

| Checkpoint | Question | Result |
| --- | --- | --- |
| G2a | Can topology prune products before numerical value loads? | Yes. Pruning follows active support, but fixed bitmap metadata is not universally faster. |
| G2c | Does row + column topology accelerate structural intersection? | Yes, strongly on high-occupancy cases; low-occupancy cases remain near explicit-index performance. |
| G2d | Can numerical values remain single-copy through a column-to-row source map? | Correct, but the extra indirection can be expensive. Duplicated hot column values are faster where transpose-oriented numerical work matters. |
| G2e | Does adaptive Sparse/Bitmap/Dense execution recover low/intermediate-occupancy performance? | Yes. This is the strongest tested CPU numerical representation. |
| G2f | Can byte-packed variable payloads reduce metadata further? | Storage improves, but hot-loop decoding is too expensive. Rejected for numerical execution. |
| G2g | Can an 8-byte typed compact descriptor retain compression without the byte decoder? | Faster than G2f, but slower than G2e on every tested matrix. Rejected as the G2e replacement. |

## G2e ten-matrix numerical corpus

All timings are median scalar CPU row-by-column sparse-dot measurements with
eight deterministic pairs per row and five repeats. `time ratio` and
`storage ratio` are relative to an explicit CSR + transpose-CSR baseline.

| Matrix | CSR nnz | avg nnz/row | Sparse tiles | Bitmap tiles | Dense tiles | time ratio | storage ratio | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| thermal1 | 574,458 | 6.95 | 272,688 | 8 | 0 | 1.642 | 1.286 | explicit |
| sherman5 | 20,793 | 6.28 | 6,547 | 548 | 0 | 1.032 | 1.138 | explicit |
| cfd1 | 1,825,580 | 25.84 | 392,861 | 37,245 | 0 | 1.067 | 0.981 | explicit |
| venkat25 | 1,717,763 | 27.52 | 321,608 | 10,020 | 0 | 0.990 | 0.925 | near parity |
| boneS01 | 5,516,602 | 43.36 | 921,644 | 13,187 | 0 | 0.782 | 0.893 | ABTM |
| s3dkq4m2 | 4,427,725 | 48.95 | 75,331 | 265,813 | 0 | 0.718 | 0.771 | ABTM |
| cant | 4,007,383 | 64.17 | 116,483 | 230,484 | 0 | 0.759 | 0.783 | ABTM |
| raefsky3 | 1,488,768 | 70.22 | 15,648 | 62,304 | 0 | 0.704 | 0.738 | ABTM |
| x104 | 8,713,602 | 80.40 | 316,872 | 431,021 | 1,717 | 0.850 | 0.785 | ABTM |
| nd3k | 3,279,690 | 364.41 | 43,102 | 101,724 | 18,689 | 0.597 | 0.796 | ABTM |

The corpus does **not** justify selecting only from 64-column word occupancy.
`boneS01` is the counterexample: about 98.6% of its tiles are Sparse, yet the
adaptive ABTM path is about 21.8% faster than explicit and uses about 10.7%
less storage. Row structural length and the amount of row/column merge work are
therefore material predictors in addition to local word occupancy.

Within this corpus, average row length separates the clear outcomes:
matrices at or below about 28 nnz/row are explicit or near parity, while all
measured matrices at or above about 43 nnz/row favor G2e. The unmeasured region
between those groups is too large to promote a production threshold. An
`avg_nnz_per_row ~= 32` rule is useful only as an experimental screening
heuristic, not a stable API or automatic policy.

## Numerical agreement

The adaptive and explicit kernels accumulate the same sparse products in
different orders. The initial six-matrix corpus had scaled errors from exact
agreement through approximately `1.5e-13`. The larger held-out matrices exposed
more cancellation:

| Matrix | max scaled error |
| --- | ---: |
| cant | 9.159e-10 |
| s3dkq4m2 | 4.846e-11 |
| boneS01 | 4.955e-9 |
| x104 | 9.313e-9 |

The performance-harness validation tolerance is therefore `1e-8`. This is a
benchmark-equivalence gate, not a solver convergence criterion. Any production
promotion must additionally validate operator-level relative error and
end-to-end residual behavior.

## Architecture decision

G2 closes with the following CPU direction:

```text
logical/symbolic layer:
    dual row/column topology
    metadata-first intersection and pruning

hot numerical layer:
    fixed direct descriptor
    adaptive Sparse / Bitmap / Dense execution
    prepared row values
    duplicated prepared column values when transpose-oriented reuse justifies it

fallback:
    explicit CSR / CSC-like path
```

The following are explicitly **not** promoted:

- universal bitmap-only numerical storage;
- source-index mapped single-copy column values as the hot default;
- byte-decoded variable payloads in the numerical hot loop;
- typed compact 8-byte descriptors as a replacement for the faster G2e layout;
- a production automatic ABTM/explicit threshold derived from this corpus.

## Preparation cost

G2e prepares both row and column orientations. On the larger corpus matrices
this is on the order of tens to a few hundreds of milliseconds per orientation.
That cost must be amortized by repeated row/column work. A production selector
therefore needs a solve/work horizon or an application-provided reuse signal;
kernel time alone is not sufficient.

## Next checkpoint

G3 moves from generic sparse-dot experiments to ABTM's intended topology-heavy
operations:

1. region growth;
2. overlap and multiplicity;
3. local submatrix extraction;
4. preparation/reuse metrics for those operations.

G3 should keep CSR as the numerical reference/fallback and should not assume
that the G2e physical layout must be reused unchanged for every region kernel.
