#!/usr/bin/env bash
# Isolated A10C8-A PCG phase accounting, reusing frozen A10C7 matched controls.
set -euo pipefail

mode="${1:-}"
a10c7="${2:-}"
a10c6="${3:-}"
a10c4="${4:-}"
a10b="${5:-}"
corpus="${6:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$a10c7" || -z "$a10c6" || -z "$a10c4" || -z "$a10b" ]]; then
    echo 'Usage: C1H12_A10C8A_profile_pilot.sh --dry-run|--run FROZEN_A10C7 FROZEN_A10C6 FROZEN_A10C4 FROZEN_A10B [ORIGINAL_FEM_CORPUS]' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[[ "$(git rev-parse --show-toplevel)" == "$repo" ]] || { echo 'Repository root mismatch' >&2; exit 1; }
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || { echo 'Unexpected branch' >&2; exit 1; }
[[ "$(git rev-parse --short HEAD)" == '2c5cbc6' ]] || { echo 'Expected committed A10C7 checkpoint 2c5cbc6' >&2; exit 1; }
[[ -z "$(git diff --cached --name-only)" ]] || { echo 'Staged changes require review' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt

historical="$(git rev-parse '60104c9^{commit}')"
[[ "$(<"$a10c7/HEAD.txt")" == "$historical" ]] || { echo 'Frozen A10C7 historical HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10c6/HEAD.txt")" == "$historical" ]] || { echo 'Frozen A10C6 historical HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10c4/HEAD.txt")" == "$historical" ]] || { echo 'Frozen A10C4 historical HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10b/HEAD.txt")" == "$(git rev-parse '1f3eaa5^{commit}')" ]] || { echo 'Frozen A10B historical HEAD mismatch' >&2; exit 1; }
cmp "$a10b/matrix_sha256.txt" "$a10c7/matrix_sha256.txt"
cmp "$a10b/matrix_sha256.txt" "$a10c6/matrix_sha256.txt"
cmp "$a10b/matrix_sha256.txt" "$a10c4/matrix_sha256.txt"
sha256sum --check --strict --quiet "$a10b/matrix_sha256.txt"
for file in \
    "$a10c7/matched_timing_samples.csv" "$a10c7/matched_timing_summary.csv" \
    "$a10c4/reliable_update_comparison.csv" "$a10c6/shadow_comparison.csv"; do
    [[ -s "$file" ]] || { echo "Missing frozen evidence: $file" >&2; exit 1; }
done
matrix="$corpus/c1g_p1_multi_patch_n144.mtx"
[[ -s "$matrix" ]] || { echo 'Original FEM matrix missing' >&2; exit 1; }

echo 'C1-H12-A10C8-A independent PCG phase attribution preflight PASS'
echo "  repo=$repo; committed HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A10C7=$a10c7; A10C6=$a10c6; A10C4=$a10c4; A10B=$a10b"
echo '  3 ABTM arms x 2 policies x 2 RHS x 3 tolerances x 2 cooldowns = 72 conditions'
echo '  6 OS processes, 1 paired warmup + 6 balanced AB/BA repeats = 864 timed solves'
echo '  exclusive phase clocks: CSR, Kahan, preconditioner, SpMV, dots, vectors, replacement SpMV/vector, direction, other'
echo '  frozen A10C7 original-CSR/Kahan trajectories checked; no timing or speedup threshold'
echo '  no production PCG, API/ABI/defaults or frozen evidence modifications'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

git diff --check
bash -n benchmarks/scripts/C1H12_A10C8A_profile_pilot.sh
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A10C7_matched_gate \
    benchmarks.scripts.test_C1H12_A10C8A_profile_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a10c8a_profile_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a10c8a_profile_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a10c8a_profile_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10c8a_profile_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
cp "$a10c7/HEAD.txt" "$run_dir/FROZEN_A10C7_HEAD.txt"
cp "$a10c6/HEAD.txt" "$run_dir/FROZEN_A10C6_HEAD.txt"
cp "$a10c4/HEAD.txt" "$run_dir/FROZEN_A10C4_HEAD.txt"
cp "$a10b/HEAD.txt" "$run_dir/FROZEN_A10B_HEAD.txt"
cp "$a10b/matrix_sha256.txt" "$run_dir/matrix_sha256.txt"
sha256sum "$a10c7/matched_timing_samples.csv" "$a10c7/matched_timing_summary.csv" \
    > "$run_dir/FROZEN_A10C7_SHA256.txt"
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
            target/release/examples/c1h12_a10c8a_profile_probe \
            --matrix "$matrix" --case multi_patch --aggregation "$aggregation" \
            --basis "$basis" --policy "${policy//_/-}" \
            --reuse-mode reuse --repeats 2 --tol 1e-12 \
            > "$run_dir/$stem.log" 2>&1
        echo "RUN $stem: exit 0"
    done
done
echo "C1-H12-A10C8-A raw evidence: $run_dir"
python3 -B benchmarks/scripts/C1H12_A10C8A_profile_gate.py \
    "$a10c7" "$a10c6" "$a10c4" "$a10b" "$run_dir"
echo "C1-H12-A10C8-A accepted evidence: $run_dir"
