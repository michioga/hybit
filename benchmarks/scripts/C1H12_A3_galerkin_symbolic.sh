#!/usr/bin/env bash
# Opt-in C1-H12-A3 symbolic P^T A P (piecewise constant P) comparator.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:---dry-run}"
if [[ "$mode" != '--dry-run' && "$mode" != '--run' ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A3_galerkin_symbolic.sh [--dry-run|--run] [corpus_dir]' >&2
    exit 2
fi
corpus="${2:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
cd "$repo_root"

[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Expected develop/0.10.0' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '4593c20' ]] || {
    echo 'Expected baseline commit 4593c20' >&2; exit 1;
}
[[ -f crates/hybit/examples/c1h12_a2_abtm_symbolic_probe.rs ]] || {
    echo 'Apply verified C1-H12-A2 candidate first' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Staged changes are not expected for this experiment' >&2; exit 1;
}
for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$matrix" ]] || { echo "Missing matrix: $matrix" >&2; exit 1; }
done

echo "C1-H12-A3 preflight PASS; HEAD=$(git rev-parse --short HEAD)"
echo "  corpus=$corpus"
echo '  3 FEM matrices x 3 aggregate sizes; CSR versus ABTM symbolic Galerkin candidate'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

[[ -x /usr/bin/time ]] || { echo 'GNU /usr/bin/time required' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a3_galerkin_symbolic_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a3_galerkin_symbolic_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a3_galerkin_symbolic_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a3_galerkin_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"

for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    for aggregate_nodes in 4 16 40; do
        stem="${case_name}_a${aggregate_nodes}"
        /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
            -o "$run_dir/${stem}.time" \
            target/release/examples/c1h12_a3_galerkin_symbolic_probe \
            --matrix "$matrix" --aggregate-nodes "$aggregate_nodes" --repeats 5 \
            > "$run_dir/${stem}.log" 2>&1
        grep -Eq '^C1H12A3\|.*\|exact_equal=true$' "$run_dir/${stem}.log"
        echo "=== $case_name / aggregate_nodes=$aggregate_nodes ==="
        grep '^C1H12A3|' "$run_dir/${stem}.log"
        cat "$run_dir/${stem}.time"
    done
done

echo "C1-H12-A3 9/9 structural and numerical-reference gates PASS; results=$run_dir"
echo 'Scope: one-hot piecewise-constant P only, Boolean candidate P^T A P; no AMG solve, PCG or PETSc runtime claim.'
