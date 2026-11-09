#!/usr/bin/env bash
# Isolated A10-A manufactured-solution tolerance/accuracy experiment.
set -euo pipefail

mode="${1:---dry-run}"
previous="${2:-}"
corpus="${3:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$previous" ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A10A_accuracy_pilot.sh [--dry-run|--run] A9_RUN_DIR [corpus_dir]' >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0 branch' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '1f3eaa5' ]] || {
    echo 'Expected frozen HEAD 1f3eaa5; inspect source history first' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Unexpected staged changes' >&2; exit 1;
}
[[ -s "$previous/HEAD.txt" && -s "$previous/matrix_sha256.txt" ]] || {
    echo 'Missing A9 owner numerical provenance' >&2; exit 1;
}
[[ "$(<"$previous/HEAD.txt")" == "$(git rev-parse HEAD)" ]] || {
    echo 'Frozen A9 source HEAD differs' >&2; exit 1;
}
sha256sum --check --strict --quiet "$previous/matrix_sha256.txt"

arms=(graph_piecewise strong_graph_piecewise strong_graph_smoothed)
rhs_modes=(ones index_smooth seeded_random)
tol_tags=(t10 t12 t13)
matrix="$corpus/c1g_p1_island_shell_n144.mtx"
[[ -s "$matrix" ]] || { echo "Missing original FEM matrix: $matrix" >&2; exit 1; }
for arm in "${arms[@]}"; do
    for rhs_mode in "${rhs_modes[@]}"; do
        for policy in two_level multilevel; do
            [[ -s "$previous/island_shell_${arm}_${rhs_mode}_${policy}_1.log" ]] || {
                echo "Missing A9 input control ${arm}/${rhs_mode}/${policy}" >&2; exit 1;
            }
        done
    done
done

echo 'C1-H12-A10A independent PCG tolerance/accuracy preflight PASS'
echo "  repo=$repo_root; HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A9=$previous; FEM=$matrix"
echo '  island_shell x 3 AMG arms x 3 exact-solution RHS x 3 tolerances x 2 policies = 54 processes'
echo '  tol=1e-10, 1e-12, 1e-13; max_iters=1200; one thread; original CSR plus Kahan residual'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU time required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
git diff --check
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A9_rhs_gate \
    benchmarks.scripts.test_C1H12_A10A_accuracy_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a10a_accuracy_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a10a_accuracy_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a10a_accuracy_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10a_accuracy_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cmp "$previous/matrix_sha256.txt" "$run_dir/matrix_sha256.txt" || {
    echo "A9 original matrix SHA mismatch; raw evidence at $run_dir" >&2; exit 1;
}
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1' > "$run_dir/thread_config.txt"
echo 'A10-A exact-solution tolerance ladder; max_iterations=1200; original-CSR and compensated residual; fixed A9 AMG policy' > "$run_dir/policy.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1

for arm in "${arms[@]}"; do
    case "$arm" in
        graph_piecewise) aggregation=graph; basis=piecewise ;;
        strong_graph_piecewise) aggregation=strong-graph; basis=piecewise ;;
        strong_graph_smoothed) aggregation=strong-graph; basis=smoothed ;;
    esac
    for rhs_mode in "${rhs_modes[@]}"; do
        native_rhs="${rhs_mode//_/-}"
        for tol_tag in "${tol_tags[@]}"; do
            case "$tol_tag" in
                t10) native_tol=1e-10 ;;
                t12) native_tol=1e-12 ;;
                t13) native_tol=1e-13 ;;
            esac
            for policy in two_level multilevel; do
                stem="island_shell_${arm}_${rhs_mode}_${tol_tag}_${policy}_1"
                native_policy="${policy//_/-}"
                if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
                    -o "$run_dir/${stem}.time" \
                    target/release/examples/c1h12_a10a_accuracy_probe \
                    --matrix "$matrix" --case island_shell \
                    --aggregation "$aggregation" --basis "$basis" \
                    --rhs-mode "$native_rhs" --policy "$native_policy" \
                    --tol "$native_tol" \
                    > "$run_dir/${stem}.log" 2>&1; then
                    echo "RUN $stem: exit 0"
                else
                    echo "RUN $stem: nonzero exit; evidence preserved" >&2
                fi
            done
        done
    done
done

echo "Raw C1-H12-A10A results: $run_dir"
python3 -B benchmarks/scripts/C1H12_A10A_accuracy_gate.py "$previous" "$run_dir"
