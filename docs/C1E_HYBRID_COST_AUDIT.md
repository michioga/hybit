# C1-E — Independent C1-D cost-evidence audit (offline)

The C1-D crossover measures Observe/Jacobi vs opt-in local Hybrid from fresh
processes. The C1-E auditor **does not modify the solver, its policy, its API,
or its numeric recurrence**. It reads a **full C1-D raw evidence directory**
and independently checks each measured pair against its recorded numeric gate.

## Input and usage

Only `schema=hybit-c1d-paired-crossover-v1` is accepted. C1-A and
`hybit-c1-fem-region-sweep-v2` files are **not** interchangeable; do not combine
the separate two-pair sweep with the nine-pair C1-D run.

After the C1-D `thermal1` crossover has produced its evidence directory:

```bash
python3 -B benchmarks/scripts/fem_hybrid_cost_audit.py \
  --evidence-dir "$HOME/hybit-c1d-thermal1-YYYYMMDD-HHMMSS" \
  --output "$HOME/hybit-c1e-thermal1-audit.json"
```

Replace the timestamp with the exact nine-pair directory. **Do not use the
latest directory automatically**: another session may have created a different
run. The auditor does not overwrite an existing output file.

## Validation semantics

For every measured pair, both arms must be present exactly once, with distinct
positions. Both must have a successful exit, no timeout, status `Converged`,
true original-CSR relative residual within tolerance, and local factor bytes
within the stated budget. Observe must report zero escalations; Hybrid must
report at least one. Numeric, time, iteration and RSS values must be finite
and physically meaningful. Missing/duplicate pairs and C1-D summary
inconsistencies are **fail-closed**; a nonzero exit code is returned.

The audit calculates medians of **paired** Hybrid-minus-Observe differences:
HyBIT total wall, solver time, diagnostic time, local factor time, iteration
count, peak process RSS and persistent factor MiB. Component durations are
reported **separately**; their instrumentation scopes may overlap and they
must not be summed as if disjoint. In particular, this dataset cannot isolate
hybrid preconditioner *apply* time or PCG-restart loss.

For at least five complete informative pairs, the auditor independently
reproduces C1-D's 4,000-resample bootstrap percentile interval (seeded from
metadata), classifies only matrix-specific evidence, and compares the computed
classification and median against `summary.json`. A two-sided exact sign
statistic is included as **exploratory**: OS effects, thermal drift and serial
run dependencies may violate independence assumptions.

The output also fingerprints the input metadata, results and summary by SHA-256
for traceability. SHA-256 records identity, **not** independent SPD proof or
measurement validity. The existing `thermal1` RHS `b=A*1` is manufactured;
real physical load vectors and independently justified SPD-class cases remain
necessary before any production policy decision.

## Acceptance gates

```bash
python3 -B -m unittest discover -s benchmarks/scripts \
  -p 'test_fem_hybrid_cost_audit.py' -v
sha256sum --check --strict --quiet SOURCE_SHA256.txt
```

The audit must be run on the original nine-pair `results.csv`, `metadata.json`,
and `summary.json` created by the crossover; terminal-output transcription is
not a substitute for raw evidence. The existing C1-D nine-pair `thermal1`
result showed a **19.7375% paired median slowdown** with one Hybrid
escalation. It is a valuable *negative* regression fixture but does not
justify a universal policy. Positive real-FEM cases, cold/warm controls and
prepared multi-RHS amortization remain future experiments.
