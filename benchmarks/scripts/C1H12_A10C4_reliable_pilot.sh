#!/usr/bin/env bash
# C1-H12-A10C4: independent original-CSR residual replacement / restart pilot.
set -euo pipefail

mode="${1:-}"
a10c3="${2:-}"
a10b="${3:-}"
corpus="${4:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$a10c3" || -z "$a10b" ]]; then
    echo 'Usage: C1H12_A10C4_reliable_pilot.sh --dry-run|--run FROZEN_A10C3 FROZEN_A10B [ORIGINAL_FEM_CORPUS]' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[[ "$(git rev-parse --show-toplevel)" == "$repo" ]] || { echo 'Repository root mismatch' >&2; exit 1; }
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || { echo 'Unexpected branch' >&2; exit 1; }
[[ "$(git rev-parse --short HEAD)" == '60104c9' ]] || { echo 'Unexpected historical checkpoint' >&2; exit 1; }
[[ -z "$(git diff --cached --name-only)" ]] || { echo 'Staged changes require review' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
[[ "$(<"$a10c3/HEAD.txt")" == "$(git rev-parse '60104c9^{commit}')" ]] || { echo 'Frozen A10C3 HEAD mismatch' >&2; exit 1; }
[[ "$(<"$a10b/HEAD.txt")" == "$(git rev-parse '1f3eaa5^{commit}')" ]] || { echo 'Frozen A10B historical HEAD mismatch' >&2; exit 1; }
cmp "$a10b/matrix_sha256.txt" "$a10c3/matrix_sha256.txt"
sha256sum --check --strict --quiet "$a10b/matrix_sha256.txt"
[[ -s "$a10c3/trace_comparison.csv" && -s "$a10c3/residual_trace.csv" ]] || { echo 'Missing frozen A10C3 trace evidence' >&2; exit 1; }
[[ -s "$a10b/prepared_reuse_comparison.csv" ]] || { echo 'Missing frozen A10B evidence' >&2; exit 1; }
matrix="$corpus/c1g_p1_multi_patch_n144.mtx"
[[ -s "$matrix" ]] || { echo 'Original multi_patch FEM matrix missing' >&2; exit 1; }
[[ -s benchmarks/scripts/C1H12_A10C3_trace_gate.py ]] || { echo 'A10C3 installed gate missing' >&2; exit 1; }

echo 'C1-H12-A10C4 independent original-CSR reliable-update preflight PASS'
echo "  repo=$repo; committed HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A10C3=$a10c3; A10B=$a10b"
echo '  multi_patch x 3 ABTM hierarchy arms x 2 policies = 6 processes'
echo '  2 difficult RHS x 3 tolerances (1e-10/1e-11/1e-12) x 3 replacement periods (0/25/50) = 108 independent outcomes'
echo '  periodic original CSR and Kahan residual; restart p=z after recomputation; 25-step evidence checkpoints'
echo '  no production PCG, public API/ABI, defaults or frozen raw evidence modifications'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

git diff --check
bash -n benchmarks/scripts/C1H12_A10C4_reliable_pilot.sh
PYTHONPATH="$repo" python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A10C3_trace_gate \
    benchmarks.scripts.test_C1H12_A10C4_reliable_gate
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a10c4_reliable_update_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a10c4_reliable_update_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a10c4_reliable_update_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10c4_reliable_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
cp "$a10c3/HEAD.txt" "$run_dir/FROZEN_A10C3_HEAD.txt"
cp "$a10b/HEAD.txt" "$run_dir/FROZEN_A10B_HEAD.txt"
cp "$a10b/matrix_sha256.txt" "$run_dir/matrix_sha256.txt"
sha256sum "$a10c3/trace_comparison.csv" "$a10c3/residual_trace.csv" \
    > "$run_dir/FROZEN_A10C3_SHA256.txt"
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
            target/release/examples/c1h12_a10c4_reliable_update_probe \
            --matrix "$matrix" --case multi_patch --aggregation "$aggregation" \
            --basis "$basis" --policy "${policy//_/-}" \
            --reuse-mode reuse --repeats 2 --tol 1e-12 \
            > "$run_dir/$stem.log" 2>&1
        echo "RUN $stem: exit 0"
    done
done
echo "C1-H12-A10C4 raw results: $run_dir"
PYTHONPATH="$repo" python3 -B benchmarks/scripts/C1H12_A10C4_reliable_gate.py \
    "$a10c3" "$a10b" "$run_dir"
echo "C1-H12-A10C4 accepted evidence: $run_dir"
