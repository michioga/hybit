# HyBIT 0.9 H1-C3K5-A — One-node MPI-PCG repeatability and Allreduce results (2026-10-20)

**Status: experimental single-node validation PASSED; not production PCG, distributed-input, multi-node, or full-FEM coefficient evidence.**

- Baseline branch: `develop/0.9.0`, HEAD before K5-A: `047a2bb` (K4). No K5-A commit/push in this checkpoint.
- Windows 11 / MSVC / native **MS-MPI**; every rank ran on **one physical node**. **No network-connected multi-node measurement.**
- Matrix source: `D:\Work\thermal1.mtx`; **only the sparsity pattern** is used to generate a synthetic strictly diagonally dominant SPD matrix (shift `0.01`). This does not test original physical FEM coefficients.
- 82,654 DOFs, 574,458 surrogate nnz; Jacobi-PCG, `tol=1e-10`, max 3000 iterations, synthetic known solution, 125 iterations / 126 SpMV / 377 dot Allreduces in every measured policy.
- Process launches: 4 ranks × 5 launches and 8 ranks × 5 launches, 5 partition policies each = **50 independent-launch policy records**. Each process launch ran 2 warmups + 5 timed solves per policy. One extra initial 4-rank smoke run was excluded from the 5-launch summary.
- Sources: 2026-10-20 user-supplied PowerShell transcript (all 10 independent launches, including the five 4-rank runs) plus five 8-rank summary CSVs and five 8-rank `.samples.csv` files. **4-rank raw sample CSVs were not supplied** and have not been independently reviewed here.
- K5-A apply gates: Rustfmt PASS, strict MPI Clippy PASS, 85 distributed library tests PASS, 3 K5 example tests PASS, release example build PASS, source integrity **318 manifest / 316 SHA-256** PASS, staged whitespace PASS; six files staged, not committed.

## Run-level timing summary

Each cell below is the **median of five per-launch medians** (each launch itself contains five timed samples). Values in milliseconds. Do not interpret 25 nested trials as 25 independent process launches.

| Rank count | Equal | Graph | ABTM Region Growth | ABTM Multi-source | ABTM Multilevel |
|---:|---:|---:|---:|---:|---:|
| 4 | 29.3218 | 28.4541 | 26.5532 | 27.0779 | 27.0042 |
| 8 | 21.0646 | 21.0118 | 20.7704 | 17.9319 | 19.9258 |

## Paired within-launch improvement vs Equal

`improvement% = 100 * (Equal median solve − candidate median solve) / Equal median solve`. The policy arms within each launch are measured sequentially with rotated per-trial order; this is a useful but nonrandomized comparison.

| Ranks | Method | Per-launch improvement (%) | Median (%) | Faster launches |
|---:|---|---|---:|---:|
| 4 | `graph_contiguous` | +3.10, +9.36, -1.26, -0.48, -1.60 | -0.48 | 2/5 |
| 4 | `abtm_region_growth` | +8.55, +14.86, +5.38, +8.45, -0.73 | +8.45 | 4/5 |
| 4 | `abtm_multisource` | -4.51, +13.17, +6.92, +6.86, +7.29 | +6.92 | 4/5 |
| 4 | `abtm_multilevel` | +6.30, +11.73, +4.55, +9.82, +7.90 | +7.90 | 5/5 |
| 8 | `graph_contiguous` | -2.19, +2.79, -0.01, +4.04, +2.85 | +2.79 | 3/5 |
| 8 | `abtm_region_growth` | -0.57, +5.62, +5.54, +5.14, +10.21 | +5.54 | 4/5 |
| 8 | `abtm_multisource` | +16.20, +14.04, +13.39, +12.78, +20.43 | +14.04 | 5/5 |
| 8 | `abtm_multilevel` | +6.85, +0.33, +7.24, +9.00, +9.37 | +7.24 | 5/5 |

**Key observation:** 8-rank ABTM Multi-source improved solve time in all five independent starts; paired improvement median **14.04%**, range **12.78–20.43%**. At 4 ranks Region Growth had 4/5 wins and Multilevel 5/5; ranking is not sufficiently stable to pick one universal partition policy. No significance claim from just five launches.

## Allreduce, Halo, and numeric breakdown (8 ranks)

Median across the five independent launch-level medians, ms. Each component is first max-reduced across ranks and **then independently median-reduced**. The columns are **not additive** and must not be subtracted from total to claim a causal attribution.

| Method | Solve | Halo exchange sum | Numeric SpMV sum | Local dot sum | Dot Allreduce sum |
|---|---:|---:|---:|---:|---:|
| `equal` | 21.0646 | 3.8301 | 10.4327 | 2.6597 | 6.9164 |
| `graph_contiguous` | 21.0118 | 3.7134 | 9.5653 | 2.5780 | 7.2009 |
| `abtm_region_growth` | 20.7704 | 3.2195 | 10.0013 | 2.5485 | 7.2256 |
| `abtm_multisource` | 17.9319 | 1.5380 | 10.0727 | 2.5459 | 6.9598 |
| `abtm_multilevel` | 19.9258 | 3.3051 | 10.3221 | 2.5777 | 6.9154 |

- At 8 ranks, Multi-source reduced median max-rank Halo exchange accumulation from 3.8301 to 1.5380 ms, approximately **59.8% lower** than Equal. This is in-process MPI Halo time including packing, not network-only time.
- Allreduce consumed ~6.96 ms of the Multi-source ~17.93 ms solve at 8 ranks, approximately **38.8% of the independently measured total**, but this percentage is descriptive, **not a proven additive critical-path attribution**. Its raw timing includes waiting/synchronization and clock-read overhead.
- Max-rank local dot sum at 8 ranks was 2.546 ms for Multi-source. There are 377 dot Allreduces per solve. This motivates studying collective overhead **and load synchronization**, not automatically replacing the PCG algorithm.
- Ghost counts at 8 ranks: Equal 23,085; Region Growth 3,227; Multi-source 5,841; Multilevel 3,687. Despite fewer ghosts, Region Growth was not the fastest PCG policy. The policy requires end-to-end evidence.

## Setup cost and amortization

The 8-rank summary CSV provides partition, owned-row extraction, Halo setup and RHS setup stage maxima. The **median of each method’s sum of reported stage maxima and solve median** was approximately: Equal 28.57 ms, Region Growth 60.18 ms, Multi-source 96.89 ms, Multilevel 241.29 ms. These are **illustrative sums, not independently timed end-to-end wall clocks**; they exclude file read / SPD surrogate construction, which replicated across all ranks.

At 8 ranks, Multi-source’s median additional preparation against Equal was about 72.26 ms. Paired per-launch median save per PCG solve was about 3.01 ms, so break-even is **roughly 24 solves**, *only if the partition/Halo/CSR and operator remain reusable and solve time remains similar*. K5-A repeatedly solves the same manufactured RHS for timing; actual solve-many with **distinct RHS** is not yet validated.

## Accuracy and remaining limits

- All 50 method/launch records report `K5 NATIVE PASS`, 125 iterations, `true_rel=4.423e-11` and `max_solution_err=1.873e-9` at console precision; 8-rank CSV full-precision residuals and iteration counts were stable across launches.
- The experimental example still loads and retains the **global CSR on every rank**. `rank_operator_halo_sum_bytes` excludes that replication, so it is **not** a true process peak RSS or proof of scalable distributed memory behavior.
- Every launch shares one physical host; neither the OS scheduling environment nor inter-node network links were varied. Benchmark policies are not randomized across process launches, and only five independent launches were measured.
- Instrumentation adds clock reads, so K5-A absolute timings should be compared within K5-A, not attributed to algorithm changes relative to K4.
- There is no evidence here for convergence on the original `thermal1.mtx` coefficients, friction/contact systems, multi-node runs, or production solver auto-routing.

## Checkpoint judgment / proposed K5-B

**K5-A experimental validation is eligible for a documentation-preserving commit after the staging integrity gate.** Preserve old production MPI-PCG and K4 example. Next: K5-B should measure `solve-many` using **several truly distinct manufactured known solutions / RHS**, reuse the prepared ownership, Halo and Jacobi state, validate each solve independently, include full startup / memory, and investigate process/NUMA affinity plus reproducibility. Do not promote any ABTM partition policy automatically based on the current single-node evidence.
