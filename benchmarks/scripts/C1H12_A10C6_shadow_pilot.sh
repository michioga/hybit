#!/usr/bin/env bash
# C1-H12-A10C6: observation-only continuation after first stagnation marker.
set -euo pipefail

mode="${1:-}"
a10c5="${2:-}"
a10c4="${3:-}"
a10b="${4:-}"
corpus="${5:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$a10c5" || -z "$a10c4" || -z "$a10b" ]]; then
    echo 'Usage: C1H12_A10C6_shadow_pilot.sh --dry-run|--run FROZEN_A10C5 FROZEN_A10C4 FROZEN_A10B [ORIGINAL_FEM_CORPUS]' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[[ "$(git rev-parse --show-toplevel)" == "$repo" ]] || { echo 'Repository root mismatch' >&2; exit 1; }
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || { echo 'Unexpected branch' >&2; exit 1; }
[[ "$(git rev-parse --short HEAD)" == '60104c9' ]] || { echo 'Unexpected historical research checkpoint' >&2; exit 1; }
[[ -z "$(git diff --cached --name-only)" ]] || { echo 'Staged changes require review' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
[[ "$(<"$a10c5/HEAD.txt")" == "$(git rev-parse '60104c9^{commit}')" ]] || { echo 'Frozen A10C5 HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10c4/HEAD.txt")" == "$(git rev-parse '60104c9^{commit}')" ]] || { echo 'Frozen A10C4 HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10b/HEAD.txt")" == "$(git rev-parse '1f3eaa5^{commit}')" ]] || { echo 'Frozen A10B historical HEAD mismatch' >&2; exit 1; }
cmp "$a10b/matrix_sha256.txt" "$a10c4/matrix_sha256.txt"
cmp "$a10b/matrix_sha256.txt" "$a10c5/matrix_sha256.txt"
sha256sum --check --strict --quiet "$a10b/matrix_sha256.txt"
[[ -s "$a10c4/reliable_update_comparison.csv" && -s "$a10c4/reliable_residual_trace.csv" ]] || { echo 'Frozen A10C4 108-case evidence missing' >&2; exit 1; }
[[ -s "$a10c5/adaptive_update_comparison.csv" && -s "$a10c5/adaptive_residual_trace.csv" ]] || { echo 'Frozen A10C5 72-case evidence missing' >&2; exit 1; }
[[ -s "$a10b/prepared_reuse_comparison.csv" ]] || { echo 'Frozen A10B evidence missing' >&2; exit 1; }
matrix="$corpus/c1g_p1_multi_patch_n144.mtx"
[[ -s "$matrix" ]] || { echo 'Original multi_patch FEM matrix missing' >&2; exit 1; }

echo 'C1-H12-A10C6 independent PCG shadow-stagnation preflight PASS'
echo "  repository=$repo; committed HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A10C5=$a10c5; A10C4=$a10c4; A10B=$a10b"
echo '  multi_patch x 3 ABTM arms x 2 hierarchy policies = 6 OS processes'
echo '  2 RHS x 3 tolerances x cooldown 25/50 = 72 diagnostic continuations'
echo '  A10C5 prefix equivalence through first stagnation candidate, then continue to 1200 or certified convergence'
echo '  frozen A10C4 convergence comparison mandatory; any regression reported separately'
echo '  production PCG, API/ABI, defaults, A10C5 and prior raw evidence unchanged'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

git diff --check
bash -n benchmarks/scripts/C1H12_A10C6_shadow_pilot.sh
PYTHONPATH="$repo" python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A10C5_adaptive_gate \
    benchmarks.scripts.test_C1H12_A10C6_shadow_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a10c6_shadow_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a10c6_shadow_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a10c6_shadow_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10c6_shadow_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
cp "$a10c5/HEAD.txt" "$run_dir/FROZEN_A10C5_HEAD.txt"
cp "$a10c4/HEAD.txt" "$run_dir/FROZEN_A10C4_HEAD.txt"
cp "$a10b/HEAD.txt" "$run_dir/FROZEN_A10B_HEAD.txt"
cp "$a10b/matrix_sha256.txt" "$run_dir/matrix_sha256.txt"
sha256sum "$a10c5/adaptive_update_comparison.csv" "$a10c5/adaptive_residual_trace.csv" \
    > "$run_dir/FROZEN_A10C5_SHA256.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for arm in graph_piecewise strong_graph_piecewise strong_graph_smoothed; do
    case "$arm" in
        graph_piecewise) aggregation=graph; basis=piecewise ;;
        strong_graph_piecewise) aggregation=strong-graph; basis=piecewise ;;
        strong_graph_smoothed) aggregation=strong-graph; basis=smoothed ;;
    esac
    for policy in two_level multilevel; do
        stem="multi_patch_${arm}_${policy}_1"
        /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
            -o "$run_dir/$stem.time" \
            target/release/examples/c1h12_a10c6_shadow_probe \
            --matrix "$matrix" --case multi_patch --aggregation "$aggregation" \
            --basis "$basis" --policy "${policy//_/-}" \
            --reuse-mode reuse --repeats 2 --tol 1e-12 \
            > "$run_dir/$stem.log" 2>&1
        echo "RUN $stem: exit 0"
    done
done
echo "C1-H12-A10C6 raw results: $run_dir"
PYTHONPATH="$repo" python3 -B benchmarks/scripts/C1H12_A10C6_shadow_gate.py \
    "$a10c5" "$a10c4" "$a10b" "$run_dir"
echo "C1-H12-A10C6 accepted evidence: $run_dir"
