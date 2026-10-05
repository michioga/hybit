# ABTM G4 closeout: symbolic/numeric ILU(0)

## Status

G4 is closed for the HyBIT 0.8 development checkpoint.

The validated production outcome is an **explicit** ABTM-assisted ILU(0)
constructor and GeneralSquare policy. Automatic matrix-level routing is not
promoted in G4.

## Architecture

The production ABTM ILU(0) path preserves the existing canonical ILU(0)
semantics:

- CSR rows are canonicalized by sorting columns, summing duplicates, and
  dropping exact-zero off-diagonal entries.
- Missing diagonal entries remain a strict applicability boundary.
- The row-relative pivot floor remains unchanged.
- Numerical factors are stored in the same canonical CSR `row_ptr`, `col_idx`,
  `lu`, and `diag_pos` layout as the established ILU(0).
- Triangular application is unchanged.
- ABTM topology, word-prefix metadata, and direct rank LUTs are constructor
  scratch and are released after factorization.

The public opt-in path is:

```text
Ilu0Preconditioner::from_csr32_general_abtm(...)
GeneralSquarePreconditionerPolicy::Ilu0Abtm
```

`Jacobi` remains the GeneralSquare default. Existing `Ilu0` and
`Ilu0Fallback` behavior is unchanged.

## G4a: symbolic intersection

G4a replaced CSR upper-candidate traversal plus per-candidate binary search
with ABTM row-word intersection. The six-matrix development corpus produced
exact symbolic agreement (`mismatched_pivots=0` throughout).

ABTM/CSR symbolic time ratios:

| Matrix | Ratio |
| --- | ---: |
| sherman5 | 0.915 |
| raefsky3 | 0.364 |
| venkat25 | 0.655 |
| cfd1 | 0.602 |
| thermal1 | 1.022 |
| nd3k | 0.175 |

The evidence established that topology-word compression can remove substantial
symbolic search work, especially for long/high-overlap rows, but does not
guarantee a win for sparse short rows.

## G4b: numeric intersection with rank-by-popcount

G4b drove actual ILU(0) updates from the same intersections while retaining
canonical CSR numerical factor storage.

All six matrices produced bitwise-identical factors and exact triangular-apply
agreement with the CSR reference.

ABTM/CSR numeric ratios:

| Matrix | Ratio |
| --- | ---: |
| sherman5 | 1.277 |
| raefsky3 | 0.741 |
| venkat25 | 1.169 |
| cfd1 | 0.983 |
| thermal1 | 1.332 |
| nd3k | 0.318 |

This isolated rank/address recovery as a significant remaining cost.

## G4c: direct all-word rank LUT

G4c replaced two rank-by-popcount operations per executed update with direct
`u8[64]` word-rank tables. Factor and apply results remained exact.

All-word LUT/CSR numeric ratios:

| Matrix | Ratio |
| --- | ---: |
| sherman5 | 0.916 |
| raefsky3 | 0.289 |
| venkat25 | 0.609 |
| cfd1 | 0.627 |
| thermal1 | 1.054 |
| nd3k | 0.137 |

The LUT therefore removed a real hot-loop bottleneck.

## G4d: adaptive per-word LUT rejected

An occupancy-threshold LUT experiment was correct for all tested thresholds,
but the per-word adaptive dispatch itself introduced enough inner-loop overhead
that even threshold=1 was materially slower than the direct G4c path on the
important matrices.

G4 therefore rejects per-word runtime LUT selection for the CPU scalar path.

## G4e: held-out matrix-level evidence

A held-out set tested the symbolic metric `word_steps_over_candidates` against
direct all-word LUT numeric timing.

Strict ILU(0) was applicable to five held-out matrices. All five clear
predictions matched measured G4c numeric winners:

- `apache2`: CSR
- `boneS01`: ABTM-LUT
- `cant`: ABTM-LUT
- `s3dkq4m2`: ABTM-LUT
- `x104`: ABTM-LUT

`Goodwin_010` remained outside strict ILU(0) applicability because a diagonal
entry is missing.

This evidence supported continuing to production implementation, but not an
automatic selector.

## G4f: production constructor corpus

The production benchmark includes canonicalization, ABTM topology construction,
prepared word metadata, rank-LUT construction, numeric factorization, and
returned-factor construction.

All eleven supported corpus matrices produced:

- identical persistent factor byte counts between CSR and ABTM constructors;
- exact triangular-apply agreement (`apply_max_scaled_error = 0`);
- no validation mismatches.

Production constructor timings:

| Matrix | CSR ms | ABTM ms | ABTM/CSR | Result |
| --- | ---: | ---: | ---: | --- |
| sherman5 | 0.327 | 0.847 | 2.592 | CSR |
| raefsky3 | 124.953 | 60.514 | 0.484 | ABTM |
| venkat25 | 51.144 | 60.076 | 1.175 | CSR |
| cfd1 | 58.270 | 71.628 | 1.229 | CSR |
| thermal1 | 10.093 | 26.541 | 2.630 | CSR |
| nd3k | 2850.031 | 540.225 | 0.190 | ABTM |
| apache2 | 41.851 | 276.767 | 6.613 | CSR |
| boneS01 | 458.298 | 252.592 | 0.551 | ABTM |
| cant | 386.524 | 165.916 | 0.429 | ABTM |
| s3dkq4m2 | 241.254 | 154.889 | 0.642 | ABTM |
| x104 | 1110.776 | 474.460 | 0.427 | ABTM |

Six matrices favored ABTM and five favored CSR.

The strongest production wins were:

- `nd3k`: about 5.28x faster constructor;
- `x104`: about 2.34x;
- `cant`: about 2.33x;
- `raefsky3`: about 2.06x;
- `boneS01`: about 1.81x;
- `s3dkq4m2`: about 1.56x.

The production corpus also invalidated promotion of the earlier permissive
`word_steps_over_candidates <= 0.40` rule: `venkat25` and `cfd1` have favorable
symbolic compression but lose once full ABTM preparation cost is included.

Across the eleven measured matrices, the arithmetic sum of constructor times
was 5333.52 ms for CSR and 2084.45 ms for ABTM. This aggregate is dominated by
the largest matrices and is descriptive only; it is not a universal performance
claim.

## Decisions

G4 closes with the following decisions:

1. Keep canonical CSR factor storage and triangular application.
2. Keep ABTM topology/rank metadata temporary to factor construction.
3. Keep the direct all-word rank LUT for the explicit ABTM constructor.
4. Reject per-word adaptive LUT dispatch on the validated CPU path.
5. Expose ABTM ILU(0) explicitly; do not change the GeneralSquare default.
6. Do not promote an automatic CSR/ABTM selector from the current corpus.
7. Treat missing diagonals as the existing strict ILU(0) applicability boundary.
8. Move the ABTM roadmap to G5 block-ABTM experiments.

A future automatic selector should use a cheap predictor measured against full
constructor cost, not only the isolated numeric kernel.
