#!/usr/bin/env bash
# C1-H12-A10C3: independent recursive/original CSR residual-gap pilot.
set -euo pipefail

mode="${1:-}"
a10c2="${2:-}"
a10c1="${3:-}"
a10b="${4:-}"
corpus="${5:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$a10c2" || -z "$a10c1" || -z "$a10b" ]]; then
    echo 'Usage: C1H12_A10C3_trace_pilot.sh --dry-run|--run FROZEN_A10C2 FROZEN_A10C1 FROZEN_A10B [ORIGINAL_FEM_CORPUS]' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[[ "$(git rev-parse --show-toplevel)" == "$repo" ]] || { echo 'Repository root mismatch' >&2; exit 1; }
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || { echo 'Unexpected branch' >&2; exit 1; }
[[ "$(git rev-parse --short HEAD)" == '60104c9' ]] || { echo 'Unexpected historical checkpoint' >&2; exit 1; }
[[ -z "$(git diff --cached --name-only)" ]] || { echo 'Staged changes require review' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt

historical_a10b="$(git rev-parse '1f3eaa5^{commit}')"
historical_current="$(git rev-parse '60104c9^{commit}')"
[[ "$(<"$a10b/HEAD.txt")" == "$historical_a10b" ]] || { echo 'Frozen A10B historical HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10c1/HEAD.txt")" == "$historical_current" ]] || { echo 'Frozen A10C1 HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10c2/HEAD.txt")" == "$historical_current" ]] || { echo 'Frozen A10C2 HEAD mismatch' >&2; exit 1; }
cmp "$a10b/matrix_sha256.txt" "$a10c1/matrix_sha256.txt"
cmp "$a10b/matrix_sha256.txt" "$a10c2/matrix_sha256.txt"
sha256sum --check --strict --quiet "$a10b/matrix_sha256.txt"
[[ -s "$a10b/prepared_reuse_comparison.csv" && -s "$a10c1/precision_floor.csv" ]] || { echo 'Missing frozen A10B/A10C1 evidence' >&2; exit 1; }
[[ -s "$a10c2/spectral_ritz.csv" && -s "$a10c2/coarse_projection.csv" ]] || { echo 'Missing frozen A10C2 spectral evidence' >&2; exit 1; }
matrix="$corpus/c1g_p1_multi_patch_n144.mtx"
[[ -s "$matrix" ]] || { echo 'Missing original multi_patch FEM matrix' >&2; exit 1; }

echo 'C1-H12-A10C3 independent PCG residual trace preflight PASS'
echo "  repo=$repo; committed HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A10B=$a10b; A10C1=$a10c1; A10C2=$a10c2"
echo '  multi_patch x 3 ABTM hierarchy arms x 2 preconditioners x 4 RHS = 24 traces'
echo '  original CSR residual, Kahan residual, recursive residual every 25 iterations'
echo '  independent diagnostic PCG versus separately rerun original certified PCG'
echo '  original production PCG, API, prior pilots and raw evidence unchanged'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

git diff --check
bash -n benchmarks/scripts/C1H12_A10C3_trace_pilot.sh
python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A10C2_spectral \
    benchmarks.scripts.test_C1H12_A10C3_trace_gate
python3 -B benchmarks/scripts/C1H12_A10C1_precision_gate.py \
    "$a10c1/precision_floor.csv" "$a10b"
python3 -B benchmarks/scripts/C1H12_A10C2_spectral_gate.py \
    "$a10c2" "$a10b" "$a10c1"
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a10c3_residual_trace_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a10c3_residual_trace_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a10c3_residual_trace_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10c3_trace_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
cp "$a10b/HEAD.txt" "$run_dir/FROZEN_A10B_HEAD.txt"
cp "$a10c2/HEAD.txt" "$run_dir/FROZEN_A10C2_HEAD.txt"
cp "$a10b/matrix_sha256.txt" "$run_dir/matrix_sha256.txt"
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
            target/release/examples/c1h12_a10c3_residual_trace_probe \
            --matrix "$matrix" --case multi_patch --aggregation "$aggregation" \
            --basis "$basis" --policy "${policy//_/-}" \
            --reuse-mode reuse --repeats 2 --tol 1e-12 \
            > "$run_dir/$stem.log" 2>&1
        echo "RUN $stem: exit 0"
    done
done
echo "C1-H12-A10C3 raw evidence: $run_dir"
PYTHONPATH="$repo" python3 -B benchmarks/scripts/C1H12_A10C3_trace_gate.py \
    "$a10b" "$a10c2" "$run_dir"
echo "C1-H12-A10C3 accepted evidence: $run_dir"
