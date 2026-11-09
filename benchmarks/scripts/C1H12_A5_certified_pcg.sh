#!/usr/bin/env bash
# Opt-in true-residual-certified native PCG; compare against original A4 logs.
set -euo pipefail

mode="${1:---dry-run}"
baseline="${2:-}"
corpus="${3:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ "$mode" != '--dry-run' && "$mode" != '--run' || -z "$baseline" ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A5_certified_pcg.sh [--dry-run|--run] A4_RUN_DIR [corpus_dir]' >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0 branch' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '4593c20' ]] || {
    echo 'Expected HEAD 4593c20: uncommitted A2/A3/A4/A5 experiments' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes not expected in this pilot' >&2; exit 1;
}
[[ -f "$baseline/HEAD.txt" && -f "$baseline/matrix_sha256.txt" ]] || {
    echo "Missing C1-H12-A4 baseline provenance in $baseline" >&2; exit 1;
}
[[ "$(<"$baseline/HEAD.txt")" == "$(git rev-parse HEAD)" ]] || {
    echo 'Baseline full Git commit ID mismatch' >&2; exit 1;
}
sha256sum --check --strict --quiet "$baseline/matrix_sha256.txt"
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$matrix" ]] || { echo "Missing matrix: $matrix" >&2; exit 1; }
done
echo 'C1-H12-A5 true-residual-certified PCG preflight PASS'
echo "  A4 baseline=$baseline; corpus=$corpus"
echo '  3 matrices x 7 same preconditioner arms; 600 iterations global, tol=1e-10'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU /usr/bin/time required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
python3 -B -m unittest -q \
    benchmarks.scripts.test_C1H12_A4_numerical_gate \
    benchmarks.scripts.test_C1H12_A5_certification_gate
cargo +1.99.1 test --release --locked -p hybit --example fem_coarse_crosscheck
cargo +1.99.1 clippy --release --locked -p hybit --example fem_coarse_crosscheck -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example fem_coarse_crosscheck

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a5_certified_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"
cmp "$baseline/matrix_sha256.txt" "$run_dir/matrix_sha256.txt" || {
    echo "Baseline corpus/input hashes changed; raw results in $run_dir" >&2; exit 1;
}
echo 'RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1' > "$run_dir/thread_config.txt"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1

arms=(jacobi contiguous_piecewise contiguous_smoothed graph_piecewise graph_smoothed strong_graph_piecewise strong_graph_smoothed)
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    for arm in "${arms[@]}"; do
        stem="${case_name}_${arm}_1"
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
        if /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
            -o "$run_dir/${stem}.time" \
            target/release/examples/fem_coarse_crosscheck \
            --matrix "$matrix" --coarse-dofs 1 --coarse-target 512 \
            --coarse-values f64 --tol 1e-10 --max-iters 600 \
            --report-c1h12a4 --certified-restart "${opts[@]}" \
            > "$run_dir/${stem}.log" 2>&1; then
            echo "RUN $stem: process exit 0"
        else
            echo "RUN $stem: NONZERO PROCESS EXIT; preserved for diagnosis" >&2
        fi
    done
done

echo "Raw C1-H12-A5 results: $run_dir"
gate_fail=0
python3 -B benchmarks/scripts/C1H12_A4_numerical_gate.py "$run_dir" --repeats 1 || gate_fail=1
python3 -B benchmarks/scripts/C1H12_A5_certification_gate.py "$baseline" "$run_dir" || gate_fail=1
echo 'Scope: same native two-level PCG; source default untouched; no PETSc/hypre dependency or speedup claim.'
exit "$gate_fail"
