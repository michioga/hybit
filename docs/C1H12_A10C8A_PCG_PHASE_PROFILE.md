# C1-H12-A10C8-A: Independent PCG phase attribution

Status: **research only**, prepared for owner WSL measurement on `develop/0.10.0` at `2c5cbc6`.

## Question

The frozen A10C7 matched pilot showed 864 certified measurements with no
meaningful global speedup: at cooldown 25/50, C4/C6 solve-time medians were
0.9976/0.9904, respectively. C6 reduced residual replacements, but some
`Ones`/`1e-10` conditions took substantially more PCG iterations. A10C7
did not separate preconditioner, main SpMV, dot, or vector time.

This phase-profile pilot does **not** modify a production solver. It records
where time goes in exactly the same original-CSR diagnostic C4/C6 algorithms
as A10C7, preserving the 72 conditions, six timed alternating pairs and shared
prepared hierarchy. It is neither a speedup experiment nor an optimization.

## Exclusive clocks

| Field | Exact timed scope |
| --- | --- |
| `csr_ns` | Original-CSR true residual check including its SpMV/norm |
| `kahan_ns` | Independent sequential CSR row accumulation with Kahan correction |
| `precond_ns` | Whole prepared preconditioner apply (includes internal AMG work) |
| `spmv_ns` | Main PCG matrix-vector product `q = A p` only |
| `dot_ns` | PCG `r⋅z`, `p⋅q`, and updated `r⋅z` inner products |
| `vector_ns` | RHS/recursive norms, `x,r,p` updates |
| `replace_spmv_ns` | Original CSR `A x` for residual replacement |
| `replace_vector_ns` | Replacement residual `r = b − A x` |
| `direction_ns` | Initial and restart copies `p = z` |
| `other_ns` | `solve_ns − sum(all above)`: uninstrumented loop/control/allocator, timer overhead, etc. |

Scopes are non-nested; an internal SpMV in `precond_ns` or `csr_ns` is **not**
again included in `spmv_ns`. Total accounting is checked for every sample.
These timers have non-negligible overhead, particularly for short vector/dot
operations. Do **not** directly compare absolute A10C8-A wall times against
A10C7 to claim an improvement or regression.

## Controls and acceptance

- Frozen A10C7, A10C6, A10C4, A10B evidence directories remain unchanged.
- The gate first independently revalidates A10C7 raw logs against A10C4/A10C6,
  then verifies its stored two CSV files exactly match the rederived evidence.
- For each of 864 new records, exact status, iteration, RHS, work counts,
  staggered order and numerical control metadata must match A10C7. Residual
  fields must match under A10C7's tight floating-point comparison tolerance.
- Timed phase counters must equal solver work: preconditioner calls, main
  SpMVs, original-CSR/Kahan checks, and replacements. No false convergence.
- Exactly six process logs, 864 samples, 144 solver/condition medians, and
  successful GNU time exits are required.
- Timings remain descriptive. No speedup/percentage threshold is a gate.

## Owner WSL commands

After downloading and placing the extracted package where you prefer:

```bash
cd ~/hybit
PACKAGE='/mnt/c/Users/michi/Downloads/hybit_c1h12_a10c8a_profile_20261111'
bash "$PACKAGE/C1H12_A10C8A_apply.sh" --dry-run "$PWD"
bash "$PACKAGE/C1H12_A10C8A_apply.sh" --apply "$PWD"

A10C7="$HOME/c1h11_evidence/c1h12_a10c7_matched_cWmQ7D"
A10C6="$HOME/c1h11_evidence/c1h12_a10c6_shadow_paXN77"
A10C4="$HOME/c1h11_evidence/c1h12_a10c4_reliable_wdamI6"
A10B="$HOME/c1h11_evidence/c1h12_a10b_prepared_oBIupI"

bash benchmarks/scripts/C1H12_A10C8A_profile_pilot.sh \
  --dry-run "$A10C7" "$A10C6" "$A10C4" "$A10B"

bash benchmarks/scripts/C1H12_A10C8A_profile_pilot.sh \
  --run "$A10C7" "$A10C6" "$A10C4" "$A10B"
```

The `--run` command executes Python checks, Rust release unit tests,
Clippy `-D warnings` and build **before** creating an evidence directory.
The gate emits `phase_profile_samples.csv` and `phase_profile_summary.csv`.

The original untracked tile-kernel experiment and Python cache in the owner
worktree are not modified, staged, or deleted. **Do not commit or push until**
the WSL owner run and numerical provenance gate pass; review the resulting
phase attribution and then decide the A10C8-B direction.
