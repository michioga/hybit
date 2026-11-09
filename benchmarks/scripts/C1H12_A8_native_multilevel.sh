#!/usr/bin/env bash
# C1-H12-A8: 36 paired native PCG processes, historical A7 additive as control.
set -euo pipefail

mode="${1:---dry-run}"
previous="${2:-}"
corpus="${3:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$previous" ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A8_native_multilevel.sh [--dry-run|--run] PREVIOUS_A7_DIR [corpus_dir]' >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0 branch' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '1f3eaa5' ]] || {
    echo 'Expected owner checkpoint 1f3eaa5; review baseline changes first' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes not expected for experimental pilot' >&2; exit 1;
}
[[ -s "$previous/HEAD.txt" && -s "$previous/matrix_sha256.txt" ]] || {
    echo 'Missing historical A7 numerical provenance' >&2; exit 1;
}
[[ "$(<"$previous/HEAD.txt")" == "$(git rev-parse HEAD)" ]] || {
    echo 'Historical A7 source HEAD mismatch' >&2; exit 1;
}
sha256sum --check --strict --quiet "$previous/matrix_sha256.txt"
arms=(contiguous_piecewise contiguous_smoothed graph_piecewise graph_smoothed strong_graph_piecewise strong_graph_smoothed)
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$matrix" ]] || { echo "Missing original FEM matrix: $matrix" >&2; exit 1; }
    for arm in "${arms[@]}"; do
        [[ -s "$previous/${case_name}_${arm}_additive_1.log" ]] || {
            echo "Missing historical A7 control: ${case_name}/${arm}" >&2; exit 1;
        }
    done
done

echo 'C1-H12-A8 native multilevel Galerkin/ABTM/V-cycle preflight PASS'
echo "  repo=$repo_root; HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A7=$previous; original FEM=$corpus"
echo '  3 FEM x 6 aggregate/basis arms x 2 preconditioners = 36 processes'
echo '  A5/A7 additive two-level control versus native multilevel symmetric V-cycle'
echo '  tol=1e-10, max_iterations=600, one thread, original-CSR certificate'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU /usr/bin/time required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A7_symmetric_gate \
    benchmarks.scripts.test_C1H12_A8_multilevel_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a8_multilevel_amg_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a8_multilevel_amg_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a8_multilevel_amg_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a8_multilevel_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cmp "$previous/matrix_sha256.txt" "$run_dir/matrix_sha256.txt" || {
    echo "Original FEM SHA mismatch; evidence retained at $run_dir" >&2; exit 1;
}
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1' > "$run_dir/thread_config.txt"
echo 'A8: 4-node aggregation; 0.25 strong threshold; 512 coarsest cap; up to 8 levels; symmetric damped Jacobi V-cycle' > "$run_dir/policy.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1

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
        for policy in two_level multilevel; do
            stem="${case_name}_${arm}_${policy}_1"
            native_policy="${policy//_/-}"
            if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
                -o "$run_dir/${stem}.time" \
                target/release/examples/c1h12_a8_multilevel_amg_probe \
                --matrix "$matrix" --case "$case_name" \
                --aggregation "$aggregation" --basis "$basis" --policy "$native_policy" \
                > "$run_dir/${stem}.log" 2>&1; then
                echo "RUN $stem: exit 0"
            else
                echo "RUN $stem: NONZERO PROCESS EXIT, original raw log retained" >&2
            fi
        done
    done
done

echo "Raw C1-H12-A8 results: $run_dir"
python3 -B benchmarks/scripts/C1H12_A8_multilevel_gate.py "$previous" "$run_dir"
