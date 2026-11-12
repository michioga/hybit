# P2-B: production PCG certified-restart paired evidence — candidate

P2-B source baseline: `develop/0.10.0` at `fdb047e` (2026-11-12); toolchain MSRV 1.99.1. **Owner WSL P2-B paired numerical gate PASS (2026-11-12):** 3 difficult SPD-FEM cases converged in both feature OFF and ON, with 0 natural rescues and 0 certified-convergence regressions. Evidence: `~/c1h11_evidence/p2b_paired_ayLJ1D`. Zero-RHS relative-only nonzero-initial-x remained `MaxIterations` (observation, not success). This work adds only isolated evidence harness files; it does not alter production PCG, default options, numerical backends, ABTM, ABI, or any frozen A10C raw data.

## Experimental question and scope

P2-A (`p2-certified-restart`) fires **only** when a recursive PCG report says `Converged`, independent original-CSR certification downgrades it to `Breakdown`, and time/iteration/preconditioner guards permit one restart. Ordinary `MaxIterations` cannot trigger P2-A. A10C4 periodic reliable updates use a *different research PCG algorithm and preconditioner* and are **not** a fair direct P2-A speed/control comparison; keep that comparison in a separately labeled follow-on study.

P2-B measures production `fem_bench` (same source example compiled feature OFF and ON) on the **same** original CSR, original RHS, reference solution, `tol=1e-10`, `max_iterations=600`, and default production hybrid policy with `--skip-plain --backend csr`. All raw stdout, `/usr/bin/time` elapsed and peak RSS, input SHA, source/tool SHA and binary SHA are preserved in one owner directory. Default case file specifies three independently manufactured difficult SPD FEM classes: island_shell, multi_patch and rotated_anisotropy. It can be changed with a tab-separated manifest listing `case, matrix, rhs, xref`; input files are never overwritten.

A separate 1D SPD Laplacian `p2b_zero_rhs_probe` uses **no hybrid escalation** to isolate the fixed Jacobi-PCG and independently computes scaled `||b-Ax||₂`. Four cases: zero RHS + zero x, zero RHS + nonzero initial x + relative-only tolerance, zero RHS + nonzero x + absolute tolerance, and a nonzero manufactured RHS. There is no assertion that the known zero-RHS relative-only scenario must converge. Non-convergence, if seen, is an observed numerical-contract defect and not a P2-A rescue or a pass for robustness; any falsely certified convergence and any iteration-cap violation are hard errors.

## Acceptance and interpretation

1. `cargo +1.99.1 test --release --locked -p hybit --example p2b_zero_rhs_probe` passes feature OFF and ON; strict Clippy and full rustfmt pass.
2. Python gate rejects false original-CSR convergence, nonfinite residual/forward error, iteration-cap overshoot, duplicate/missing zero-RHS cases, unmatched input data, or OFF-converged/ON-not-converged regressions.
3. Natural rescue is counted **only** as OFF `MaxIterations`/`Breakdown` to ON **independently certified** `Converged` under the fixed matched contract. An equal or unchanged result is recorded, not presented as a rescue.
4. A lack of natural drift is not evidence the P2-A restart improves convergence. The output should then report `natural_rescues: 0` transparently. Conversely one positive pilot is not a broad held-out success rate.
5. `fem_bench` prints independent verified relative residual to six decimal digits and computes it using an ordinary (not scaled) norm. That method is suitable here only because the selected manufactured FEM RHS are normal-scale; for extreme RHS, the separate scaled-norm probe is the required check. Do not generalize this pilot's printed residual handling to extreme dynamic ranges.
6. `elapsed_s` and `peak_rss_kib` are whole-process measurements; they include Matrix Market input parsing, setup and solve, not an isolated solver-kernel speedup. Repeated timings and a larger held-out corpus are needed before any performance claim.

## Owner WSL procedure

```bash
cd ~/hybit
# Source harness is already committed in fdb047e. Apply P2B FIX1 separately
# before running the validation commands below; do not reapply an old package.

cargo +1.99.1 fmt --all -- --check
sha256sum --check --strict --quiet SOURCE_SHA256.txt
python3 -B -m unittest -v benchmarks.scripts.test_P2B_gate
cargo +1.99.1 test --release --locked -p hybit --example p2b_zero_rhs_probe
cargo +1.99.1 test --release --locked -p hybit --example p2b_zero_rhs_probe --features p2-certified-restart
cargo +1.99.1 clippy --release --locked -p hybit --example p2b_zero_rhs_probe -- -D warnings
cargo +1.99.1 clippy --release --locked -p hybit --example p2b_zero_rhs_probe --features p2-certified-restart -- -D warnings

a=benchmarks/scripts/P2B_pilot.sh
bash "$a" --dry-run
bash "$a" --run
```

By default, the case paths are below `$HOME/mtx/c1g`. If these exact files are not present, prepare a **separate** TSV manifest matching the four columns of `benchmarks/scripts/P2B_cases.tsv` and pass that file as the last argument to `--dry-run` and `--run`. Do not silently substitute non-SPD or different numerical class. The pilot saves `p2b_comparison.csv`, `p2b_summary.json`, raw logs/time, binary/input hashes and the exact manifest under `$HOME/c1h11_evidence/p2b_paired_*`.

After the owner results are reviewed, extend the test to difficult FEM RHS diversity and compare against the separately frozen A10C4 pilot with an explicit warning that AMG/coarse operators differ. **Do not commit/push until all owner gates are accepted.** Neither a verified natural rescue nor P2-A default enablement is established by this package.

## Third-party-review note

This test responds to the review-1 numerical complaint about `b=0` with nonzero initial x while preserving the old solver contract for observation. Review-1 also raised solve-many signature hashes, local Cholesky scale, Rust `Ok + MaxIterations` semantics, and MPI host communicator ownership; they are **out of scope** for P2-B and remain separate priorities. Review-2's C1-G corpus and PETSc baseline status were snapshots prior to the subsequent research evidence; never remove or re-run those results unnecessarily. ABTM remains a topology/Boolean/symbolic helper where measured useful, not a PCG preconditioner.
