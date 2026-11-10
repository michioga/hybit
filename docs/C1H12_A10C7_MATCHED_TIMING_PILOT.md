# C1-H12-A10C7 — matched A10C4/A10C6 diagnostic PCG timing

## Question

Does A10C6's cooldown + shadow continuation reduce residual replacement work and
measured diagnostic PCG time relative to frozen A10C4, **when the two algorithms
see exactly the same original FEM, RHS, tolerance, hierarchy and machine run**?

A10C4's 108-case process elapsed times **cannot** be compared directly to
A10C6's 72-case elapsed times: A10C4 also ran a period-zero arm. A10C7
reconstructs a common comparison grid; the prior CSVs are **numerical controls
only**, never timing baselines.

## Frozen inputs / safety

- Committed research base: `develop/0.10.0`, `60104c9`.
- A10B `1f3eaa5` frozen original FEM, RHS, ABTM hierarchy and matrix hash.
- A10C4 108 original-CSR reliable-update outcomes; compare periods 25 and 50.
- A10C6 72 shadow-continuation outcomes; compare cooldown 25 and 50.
- Production PCG, `hybit` API/ABI, defaults, A10B–A10C6 source/evidence unchanged.
- A10C7 is **another standalone research example**, not a solver policy change.

## Owner WSL experiment

Six independent OS processes: `multi_patch` × three ABTM aggregate/basis arms
(`graph_piecewise`, `strong_graph_piecewise`, `strong_graph_smoothed`) × two
policies (`TwoLevel`, `Multilevel`). Each builds **one** prepared hierarchy and
reuses it for both diagnostic solvers.

Each process covers two RHS (`Ones`, `PointLoad`) × three tolerances (`1e-10`,
`1e-11`, `1e-12`) × replacement interval 25/50: 12 paired conditions.
For each condition: one excluded paired warmup followed by six measured pairs,
three `A10C4 → A10C6` and three `A10C6 → A10C4`. Thus **864 measured solves**
plus **144 excluded warmup solves** across all processes.

Timing decomposition:

- `setup_ns`: one shared hierarchy setup per OS process, not attributed twice.
- `solve_ns`: wall time around the complete diagnostic PCG, including its
  original-CSR and Kahan residual certificates and replacement work.
- `certificate_ns`: cumulative time inside original CSR true-residual and
  independent Kahan residual computations during the PCG run.
- `replacement_ns`: cumulative CSR SpMV + `r = b − Ax` refresh time; excludes
  the subsequent preconditioner apply, which remains in `solve_ns`.
- Per-condition medians and median of six **paired** C4/C6 solve ratios.

Checkpoint printing is disabled **only in the new timed example**, for both
algorithms. Preconditioner setup, RHS creation, file I/O, warmups and CSV
generation are excluded from the individual solve wall times. Their per-process
cost is independently recorded by GNU time for auditing.

## Fail-closed numerical and provenance gate

`C1H12_A10C7_matched_gate.py` checks:

1. Exact full 864 records, chronological balanced AB/BA order, no duplicate
   samples; six processes and all 72 paired conditions.
2. Frozen A10C4/A10C6 CSV schemas, all 108/72 reference identities, consistent
   original RHS hashes, and unchanged SHA-256 fingerprints.
3. Every repeated status, iteration count, preconditioner/check/replacement/
   false-event count, original CSR residual, compensated residual, best true
   residual, and (A10C6) first stagnation candidate against its original control.
4. Every `Converged` sample is certified by *recursive*, *original CSR*, and
   *Kahan* relative residuals at the specified tolerance.
5. Finite positive time, valid nonoverlapping timing decomposition, valid
   replacement counts, original matrix SHA and historical HEAD provenance.

It writes `matched_timing_samples.csv` (864) and `matched_timing_summary.csv`
(72) into a new `~/c1h11_evidence/c1h12_a10c7_matched_XXXXXX` directory.

**PASS means the measurement and numerical provenance are valid.** It does not
mean A10C6 is faster, nor does it prove a production speedup. Performance can
lose, be indistinguishable, depend on RHS/tolerance/iteration count, or change
with the environment. Interpret the paired ratios together with total iterations,
replacement counts, setup time, and convergence status. The timing experiment
is conducted on one matrix and one machine, with limited repeated sampling;
thermal drift, cache effects and scheduling remain possible.

## Commit boundary

Install the five new research files, execute the WSL owner `--dry-run` and
`--run`, and inspect the resulting summary. Only after the numerical/timing
gate passes and `MANIFEST.txt` / `SOURCE_SHA256.txt` / clean staged diff checks
pass should the owner stage **explicit** A10C1–A10C7 research sources and
manifests, commit, then push `develop/0.10.0`. Previously untracked unrelated
tile-kernel files and Python bytecode are outside this checkpoint.
