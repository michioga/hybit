#!/usr/bin/env bash
# A10B: native repeated RHS solve with fresh vs once-prepared A8 ABTM AMG hierarchy.
set -euo pipefail

mode="${1:---dry-run}"
a10a="${2:-}"
a9="${3:-}"
corpus="${4:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$a10a" || -z "$a9" ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A10B_prepared_pilot.sh --dry-run|--run A10A_RUN_DIR A9_RUN_DIR [corpus_dir]' >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '1f3eaa5' ]] || {
    echo 'Unexpected frozen HEAD; review owner commits before rerunning' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes must be reviewed first' >&2; exit 1;
}
for previous in "$a9" "$a10a"; do
    [[ -s "$previous/HEAD.txt" && -s "$previous/matrix_sha256.txt" ]] || {
        echo "Missing frozen A9/A10A provenance: $previous" >&2; exit 1;
    }
    [[ "$(<"$previous/HEAD.txt")" == "$(git rev-parse HEAD)" ]] || {
        echo "Frozen source HEAD mismatch in $previous" >&2; exit 1;
    }
done
cmp "$a9/matrix_sha256.txt" "$a10a/matrix_sha256.txt" || {
    echo 'A9 and A10A matrix provenance mismatch' >&2; exit 1;
}
sha256sum --check --strict --quiet "$a9/matrix_sha256.txt"
[[ -s "$a10a/accuracy_tolerance_comparison.csv" ]] || {
    echo 'Missing frozen A10A tolerance CSV' >&2; exit 1;
}
for case in island_shell multi_patch rotated_anisotropy; do
    [[ -s "$corpus/c1g_p1_${case}_n144.mtx" ]] || {
        echo "Missing original FEM $case" >&2; exit 1;
    }
    for rhs in ones index_smooth seeded_random point_load; do
        [[ -s "$a9/${case}_graph_piecewise_${rhs}_two_level_1.log" ]] || {
            echo "Missing A9 fingerprint: $case/$rhs" >&2; exit 1;
        }
    done
done
[[ -s crates/hybit/examples/c1h12_a10a_accuracy_probe.rs ]] || {
    echo 'A10A baseline probe missing' >&2; exit 1;
}
[[ -s crates/hybit/examples/c1h12_a10b_prepared_probe.rs ]] || {
    echo 'A10B prepared probe missing' >&2; exit 1;
}
sha256sum --check --strict --quiet SOURCE_SHA256.txt
echo 'C1-H12-A10B prepared hierarchy reuse preflight PASS'
echo "  repo=$repo_root; HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A10A=$a10a; A9=$a9; FEM=$corpus"
echo '  3 FEM x 3 ABTM hierarchy arms x 2 preconditioners x 2 lifecycle modes = 36 processes'
echo '  each process: 4 identical RHS x 5 measured repeats, plus 1 unmeasured warmup batch'
echo '  tol=1e-12; max_iterations=1200; one thread; original-CSR and Kahan certificates'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or raw evidence files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU time required' >&2; exit 1; }
git diff --check
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -v benchmarks.scripts.test_C1H12_A10A_accuracy_gate benchmarks.scripts.test_C1H12_A10B_prepared_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a10b_prepared_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a10b_prepared_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a10b_prepared_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10b_prepared_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cmp "$a9/matrix_sha256.txt" "$run_dir/matrix_sha256.txt" || {
    echo "A10B matrix SHA mismatch; raw evidence at $run_dir" >&2; exit 1;
}
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1' > "$run_dir/thread_config.txt"
echo 'A10B fresh/reuse same A8 ABTM hierarchy; 5 repeats x 4 RHS; per-process warmup excluded; no ABTM-only speedup attribution' > "$run_dir/policy.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

arms=(graph_piecewise strong_graph_piecewise strong_graph_smoothed)
fails=0
for case in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case}_n144.mtx"
    for arm in "${arms[@]}"; do
        case "$arm" in
            graph_piecewise) aggregation=graph; basis=piecewise ;;
            strong_graph_piecewise) aggregation=strong-graph; basis=piecewise ;;
            strong_graph_smoothed) aggregation=strong-graph; basis=smoothed ;;
        esac
        for policy in two_level multilevel; do
            for reuse_mode in fresh reuse; do
                stem="${case}_${arm}_${policy}_${reuse_mode}_1"
                if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' -o "$run_dir/${stem}.time" target/release/examples/c1h12_a10b_prepared_probe --matrix "$matrix" --case "$case" --aggregation "$aggregation" --basis "$basis" --policy "${policy//_/-}" --reuse-mode "$reuse_mode" --repeats 5 --tol 1e-12 > "$run_dir/${stem}.log" 2>&1; then
                    echo "RUN $stem: exit 0"
                else
                    echo "RUN $stem: nonzero exit, evidence retained" >&2
                    ((fails += 1))
                fi
            done
        done
    done
done
echo "Raw C1-H12-A10B results: $run_dir"
if ((fails != 0)); then
    echo "A10B failed $fails process(es); review raw evidence before accepting" >&2
    exit 1
fi
python3 -B benchmarks/scripts/C1H12_A10B_prepared_gate.py "$a9" "$a10a" "$run_dir"
echo 'Scope: prepared ABTM-backed hierarchy reuse vs fresh; no ABTM-only, multithread, or general speedup claim.'
