#!/usr/bin/env bash
# C1-H12-A7: original-CSR-certified native two-level SPD composite comparison.
set -euo pipefail

mode="${1:---dry-run}"
previous="${2:-}"
corpus="${3:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$previous" ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A7_symmetric_pilot.sh [--dry-run|--run] PREVIOUS_A6_DIR [corpus_dir]' >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0 branch' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '1f3eaa5' ]] || {
    echo 'Expected frozen checkpoint HEAD 1f3eaa5; review before rebasing' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes are not expected' >&2; exit 1;
}
[[ -s "$previous/HEAD.txt" && -s "$previous/matrix_sha256.txt" ]] || {
    echo 'Missing historical A6 provenance' >&2; exit 1;
}
[[ "$(<"$previous/HEAD.txt")" == "$(git rev-parse HEAD)" ]] || {
    echo 'Historical A6 HEAD mismatch' >&2; exit 1;
}
sha256sum --check --strict --quiet "$previous/matrix_sha256.txt"
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$matrix" ]] || { echo "Missing matrix: $matrix" >&2; exit 1; }
    for arm in contiguous_piecewise contiguous_smoothed graph_piecewise graph_smoothed strong_graph_piecewise strong_graph_smoothed; do
        [[ -s "$previous/${case_name}_${arm}_a5_1.log" ]] || {
            echo "Missing historical A6/A5 record: $case_name/$arm" >&2; exit 1;
        }
    done
done

echo 'C1-H12-A7 native two-level SPD composite preflight PASS'
echo "  repo=$repo_root; HEAD=$(git rev-parse --short HEAD)"
echo "  historical A6=$previous; corpus=$corpus"
echo '  3 matrices x 6 coarse arms x 2 policies = 36 native PCG processes'
echo '  additive A5 control versus symmetric Jacobi + additive composite'
echo '  tol=1e-10; max_iters=600; one thread; original-CSR true-residual gate'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU /usr/bin/time required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -v benchmarks.scripts.test_C1H12_A7_symmetric_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a7_symmetric_composite_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a7_symmetric_composite_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a7_symmetric_composite_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a7_symmetric_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cmp "$previous/matrix_sha256.txt" "$run_dir/matrix_sha256.txt" || {
    echo "Input matrix SHA mismatch; evidence remains at $run_dir" >&2; exit 1;
}
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1' > "$run_dir/thread_config.txt"
echo 'A7: same A5 certified-restart PCG; additive versus symmetric composite' > "$run_dir/policy.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1

arms=(contiguous_piecewise contiguous_smoothed graph_piecewise graph_smoothed strong_graph_piecewise strong_graph_smoothed)
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    for arm in "${arms[@]}"; do
        case "$arm" in
            contiguous_piecewise) aggregation=contiguous; basis=piecewise ;;
            contiguous_smoothed) aggregation=contiguous; basis=smoothed ;;
            graph_piecewise) aggregation=graph; basis=piecewise ;;
            graph_smoothed) aggregation=graph; basis=smoothed ;;
            strong_graph_piecewise) aggregation=strong-graph; basis=piecewise ;;
            strong_graph_smoothed) aggregation=strong-graph; basis=smoothed ;;
        esac
        for policy in additive symmetric; do
            stem="${case_name}_${arm}_${policy}_1"
            if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
                -o "$run_dir/${stem}.time" \
                target/release/examples/c1h12_a7_symmetric_composite_probe \
                --matrix "$matrix" --case "$case_name" \
                --aggregation "$aggregation" --basis "$basis" --policy "$policy" \
                > "$run_dir/${stem}.log" 2>&1; then
                echo "RUN $stem: exit 0"
            else
                echo "RUN $stem: NONZERO PROCESS EXIT, log retained" >&2
            fi
        done
    done
done

echo "Raw C1-H12-A7 results: $run_dir"
python3 -B benchmarks/scripts/C1H12_A7_symmetric_gate.py "$previous" "$run_dir"
