#!/usr/bin/env bash
# C1-H12-A6: paired A5/periodic-PCG controls at a single native source HEAD.
set -euo pipefail

mode="${1:---dry-run}"
previous="${2:-}"
corpus="${3:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$previous" ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A6_periodic_pcg.sh [--dry-run|--run] PREVIOUS_A5_DIR [corpus_dir]' >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0 branch' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '1f3eaa5' ]] || {
    echo 'Expected owner checkpoint HEAD 1f3eaa5; do not run on a moving baseline' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes are not expected' >&2; exit 1;
}
[[ -f "$previous/HEAD.txt" && -f "$previous/matrix_sha256.txt" ]] || {
    echo 'Missing historical A5 run provenance' >&2; exit 1;
}
sha256sum --check --strict --quiet "$previous/matrix_sha256.txt"
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$matrix" ]] || { echo "Missing matrix: $matrix" >&2; exit 1; }
    [[ -f "$previous/${case_name}_jacobi_1.log" ]] || {
        echo "Incomplete A5 baseline for $case_name" >&2; exit 1;
    }
done

echo 'C1-H12-A6 native periodic-PCG preflight PASS'
echo "  repo=$repo_root; HEAD=$(git rev-parse --short HEAD)"
echo "  historical A5=$previous; corpus=$corpus"
echo '  3 FEM matrices x 7 identical preconditioner arms x 3 policies = 63 processes'
echo '  policies: original A5, periodic 16 iterations, periodic 32 iterations; gap=2'
echo '  tol=1e-10, max_iters=600, one thread; original-CSR true-residual gate'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU /usr/bin/time required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -q \
    benchmarks.scripts.test_C1H12_A4_numerical_gate \
    benchmarks.scripts.test_C1H12_A5_certification_gate \
    benchmarks.scripts.test_C1H12_A6_periodic_gate
cargo +1.99.1 test --release --locked -p hybit --example fem_coarse_crosscheck
cargo +1.99.1 clippy --release --locked -p hybit --example fem_coarse_crosscheck -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example fem_coarse_crosscheck

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a6_periodic_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cmp "$previous/matrix_sha256.txt" "$run_dir/matrix_sha256.txt" || {
    echo "Input SHA mismatch, raw evidence retained in $run_dir" >&2; exit 1;
}
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1' > "$run_dir/thread_config.txt"
printf '%s\n' 'a5=--certified-restart, p16=--certified-restart --cert-periodic-check 16 --cert-gap-ratio 2, p32=...32' > "$run_dir/policy.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1

arms=(jacobi contiguous_piecewise contiguous_smoothed graph_piecewise graph_smoothed strong_graph_piecewise strong_graph_smoothed)
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    for arm in "${arms[@]}"; do
        opts=(--mode jacobi)
        case "$arm" in
            jacobi) ;;
            contiguous_piecewise) opts=(--mode direct --coarse-aggregation contiguous --coarse-basis piecewise) ;;
            contiguous_smoothed) opts=(--mode direct --coarse-aggregation contiguous --coarse-basis smoothed) ;;
            graph_piecewise) opts=(--mode direct --coarse-aggregation graph --coarse-basis piecewise) ;;
            graph_smoothed) opts=(--mode direct --coarse-aggregation graph --coarse-basis smoothed) ;;
            strong_graph_piecewise) opts=(--mode direct --coarse-aggregation strong-graph --coarse-basis piecewise) ;;
            strong_graph_smoothed) opts=(--mode direct --coarse-aggregation strong-graph --coarse-basis smoothed) ;;
        esac
        for policy in a5 p16 p32; do
            stem="${case_name}_${arm}_${policy}_1"
            periodic_opts=()
            case "$policy" in
                a5) ;;
                p16) periodic_opts=(--cert-periodic-check 16 --cert-gap-ratio 2) ;;
                p32) periodic_opts=(--cert-periodic-check 32 --cert-gap-ratio 2) ;;
            esac
            if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
                -o "$run_dir/${stem}.time" \
                target/release/examples/fem_coarse_crosscheck \
                --matrix "$matrix" --coarse-dofs 1 --coarse-target 512 \
                --coarse-values f64 --tol 1e-10 --max-iters 600 \
                --report-c1h12a4 --certified-restart \
                "${opts[@]}" "${periodic_opts[@]}" \
                > "$run_dir/${stem}.log" 2>&1; then
                echo "RUN $stem: exit 0"
            else
                echo "RUN $stem: NONZERO PROCESS EXIT, raw log retained" >&2
            fi
        done
    done
done

echo "Raw C1-H12-A6 results: $run_dir"
python3 -B benchmarks/scripts/C1H12_A6_periodic_gate.py "$previous" "$run_dir"
