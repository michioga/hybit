#!/usr/bin/env bash
# C1-H12-A9: independent 72-process A8 AMG RHS-diversity numerical pilot.
set -euo pipefail

mode="${1:---dry-run}"
previous="${2:-}"
corpus="${3:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$previous" ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A9_rhs_diversity.sh [--dry-run|--run] A8_RUN_DIR [corpus_dir]' >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0 branch' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '1f3eaa5' ]] || {
    echo 'Expected frozen owner checkpoint 1f3eaa5; review changes first' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes are not expected for isolated A9 experiment' >&2; exit 1;
}
[[ -s "$previous/HEAD.txt" && -s "$previous/matrix_sha256.txt" ]] || {
    echo 'Missing A8 original numerical provenance' >&2; exit 1;
}
[[ "$(<"$previous/HEAD.txt")" == "$(git rev-parse HEAD)" ]] || {
    echo 'A8 full HEAD source mismatch' >&2; exit 1;
}
sha256sum --check --strict --quiet "$previous/matrix_sha256.txt"

arms=(graph_piecewise strong_graph_piecewise strong_graph_smoothed)
rhs_modes=(ones index_smooth seeded_random point_load)
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$matrix" ]] || { echo "Missing original FEM matrix: $matrix" >&2; exit 1; }
    for arm in "${arms[@]}"; do
        for policy in two_level multilevel; do
            [[ -s "$previous/${case_name}_${arm}_${policy}_1.log" ]] || {
                echo "Missing frozen A8: ${case_name}/${arm}/${policy}" >&2; exit 1;
            }
        done
    done
done

echo 'C1-H12-A9 RHS diversity numerical preflight PASS'
echo "  repo=$repo_root; HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A8=$previous; original FEM=$corpus"
echo '  3 FEM x 3 selected coarse arms x 4 RHS modes x 2 policies = 72 processes'
echo '  RHS: A8 ones replay, indexed smooth truth, seeded random truth, single point load'
echo '  tol=1e-10; max_iterations=600; one thread; original-CSR certificate'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU /usr/bin/time is required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A8_multilevel_gate \
    benchmarks.scripts.test_C1H12_A9_rhs_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a9_rhs_diversity_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a9_rhs_diversity_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a9_rhs_diversity_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a9_rhs_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cmp "$previous/matrix_sha256.txt" "$run_dir/matrix_sha256.txt" || {
    echo "Original FEM matrix hashes differ from frozen A8; evidence at $run_dir" >&2; exit 1;
}
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1' > "$run_dir/thread_config.txt"
echo 'A9: unchanged A8 4-node aggregation; 512 coarsest cap; 8 levels max; symmetric damped Jacobi' > "$run_dir/policy.txt"
echo 'ones/index_smooth/seeded_random manufactured x; point_load b[n/2]=1; seeded xorshift64 fixed seed' > "$run_dir/rhs_policy.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1

for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    for arm in "${arms[@]}"; do
        case "$arm" in
            graph_piecewise) aggregation=graph; basis=piecewise ;;
            strong_graph_piecewise) aggregation=strong-graph; basis=piecewise ;;
            strong_graph_smoothed) aggregation=strong-graph; basis=smoothed ;;
        esac
        for rhs_mode in "${rhs_modes[@]}"; do
            native_rhs="${rhs_mode//_/-}"
            for policy in two_level multilevel; do
                stem="${case_name}_${arm}_${rhs_mode}_${policy}_1"
                native_policy="${policy//_/-}"
                if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
                    -o "$run_dir/${stem}.time" \
                    target/release/examples/c1h12_a9_rhs_diversity_probe \
                    --matrix "$matrix" --case "$case_name" \
                    --aggregation "$aggregation" --basis "$basis" \
                    --rhs-mode "$native_rhs" --policy "$native_policy" \
                    > "$run_dir/${stem}.log" 2>&1; then
                    echo "RUN $stem: exit 0"
                else
                    echo "RUN $stem: NONZERO PROCESS EXIT; raw logs retained" >&2
                fi
            done
        done
    done
done

echo "Raw C1-H12-A9 results: $run_dir"
python3 -B benchmarks/scripts/C1H12_A9_rhs_gate.py "$previous" "$run_dir"
