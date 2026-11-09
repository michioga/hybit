#!/usr/bin/env bash
# Native HyBIT convergence pilot: reuse existing two-level SPD preconditioner.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:---dry-run}"
corpus="${2:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
repeats="${3:-1}"
if [[ "$mode" != '--dry-run' && "$mode" != '--run' ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A4_native_coarse_pilot.sh [--dry-run|--run] [corpus_dir] [repeats]' >&2
    exit 2
fi
if [[ ! "$repeats" =~ ^[1-9][0-9]*$ ]] || (( repeats > 10 )); then
    echo 'Repeats must be an integer from 1 through 10' >&2
    exit 2
fi

cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '4593c20' ]] || {
    echo 'Expected exact baseline commit 4593c20 (A2/A3/A4 uncommitted)' >&2; exit 1;
}
[[ -f crates/hybit/examples/c1h12_a3_galerkin_symbolic_probe.rs ]] || {
    echo 'C1-H12-A3 comparison files are required' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes are not expected during this pilot' >&2; exit 1;
}
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$matrix" ]] || { echo "Missing matrix: $matrix" >&2; exit 1; }
done

echo "C1-H12-A4 native coarse pilot preflight PASS; HEAD=$(git rev-parse --short HEAD)"
echo "  corpus=$corpus"
echo "  3 cases x 7 native PCG arms x $repeats process repeats; tol=1e-10 max_iters=600"
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU /usr/bin/time required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -v benchmarks.scripts.test_C1H12_A4_numerical_gate
cargo +1.99.1 clippy --release --locked -p hybit --example fem_coarse_crosscheck -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example fem_coarse_crosscheck

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a4_native_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cargo +1.99.1 -V > "$run_dir/cargo_version.txt"
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1' > "$run_dir/thread_config.txt"
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1

arms=(jacobi contiguous_piecewise contiguous_smoothed graph_piecewise graph_smoothed strong_graph_piecewise strong_graph_smoothed)
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    for arm in "${arms[@]}"; do
        for ((repeat=1; repeat<=repeats; repeat++)); do
            stem="${case_name}_${arm}_${repeat}"
            opts=(--mode direct --coarse-aggregation graph --coarse-basis smoothed)
            case "$arm" in
                jacobi) opts=(--mode jacobi) ;;
                contiguous_piecewise) opts=(--mode direct --coarse-aggregation contiguous --coarse-basis piecewise) ;;
                contiguous_smoothed) opts=(--mode direct --coarse-aggregation contiguous --coarse-basis smoothed) ;;
                graph_piecewise) opts=(--mode direct --coarse-aggregation graph --coarse-basis piecewise) ;;
                graph_smoothed) opts=(--mode direct --coarse-aggregation graph --coarse-basis smoothed) ;;
                strong_graph_piecewise) opts=(--mode direct --coarse-aggregation strong-graph --coarse-basis piecewise) ;;
                strong_graph_smoothed) opts=(--mode direct --coarse-aggregation strong-graph --coarse-basis smoothed) ;;
            esac
            if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
                -o "$run_dir/${stem}.time" \
                target/release/examples/fem_coarse_crosscheck \
                --matrix "$matrix" --coarse-dofs 1 --coarse-target 512 \
                --coarse-values f64 --tol 1e-10 --max-iters 600 \
                --report-c1h12a4 "${opts[@]}" > "$run_dir/${stem}.log" 2>&1; then
                echo "RUN $stem: process exit 0"
            else
                echo "RUN $stem: NONZERO PROCESS EXIT (preserved for diagnosis)" >&2
            fi
        done
    done
done

echo "Raw C1-H12-A4 results: $run_dir"
python3 -B benchmarks/scripts/C1H12_A4_numerical_gate.py "$run_dir" --repeats "$repeats"
echo 'Scope: native HyBIT two-level PCG, not multilevel AMG; no PETSc or hypre linkage.'
