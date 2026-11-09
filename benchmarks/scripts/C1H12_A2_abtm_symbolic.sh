#!/usr/bin/env bash
# Opt-in C1-H12-A2 native HyBIT CSR-vs-ABTM symbolic comparison.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:---dry-run}"
if [[ "$mode" != '--dry-run' && "$mode" != '--run' ]]; then
    echo 'Usage: bash benchmarks/scripts/C1H12_A2_abtm_symbolic.sh [--dry-run|--run] [corpus_directory]' >&2
    exit 2
fi

corpus="${2:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
cd "$repo_root"

[[ "$(git branch --show-current)" == 'develop/0.10.0' ]]
[[ "$(git rev-parse --short HEAD)" == '4593c20' ]]
[[ -z "$(git diff --cached --name-only)" ]]

for case_name in island_shell multi_patch rotated_anisotropy; do
    file="$corpus/c1g_p1_${case_name}_n144.mtx"
    [[ -s "$file" ]] || { echo "Missing matrix: $file" >&2; exit 1; }
done

echo "C1-H12-A2 preflight PASS; HEAD=$(git rev-parse --short HEAD)"
echo "  corpus=$corpus"
echo '  Workload: strength-of-connection shared; CSR vs ABTM exact two-hop support'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

if [[ ! -x /usr/bin/time ]]; then
    echo 'GNU /usr/bin/time is required for elapsed/RSS evidence' >&2
    exit 1
fi

sha256sum --check --strict --quiet SOURCE_SHA256.txt
cargo +1.99.1 fmt --all -- --check
cargo +1.99.1 test --release --locked -p hybit --example c1h12_a2_abtm_symbolic_probe
cargo +1.99.1 clippy --release --locked -p hybit --example c1h12_a2_abtm_symbolic_probe -- -D warnings
cargo +1.99.1 build --release --locked -p hybit --example c1h12_a2_abtm_symbolic_probe

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a2_symbolic_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
sha256sum "$corpus"/c1g_p1_{island_shell,multi_patch,rotated_anisotropy}_n144.mtx > "$run_dir/matrix_sha256.txt"

for case_name in island_shell multi_patch rotated_anisotropy; do
    matrix="$corpus/c1g_p1_${case_name}_n144.mtx"
    /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
        -o "$run_dir/${case_name}.time" \
        target/release/examples/c1h12_a2_abtm_symbolic_probe \
        --matrix "$matrix" --threshold 0.25 --repeats 5 \
        > "$run_dir/${case_name}.log" 2>&1
    grep -Eq '^C1H12A2\|.*\|exact_equal=true$' "$run_dir/${case_name}.log"
    echo "=== $case_name ==="
    grep '^C1H12A2|' "$run_dir/${case_name}.log"
    cat "$run_dir/${case_name}.time"
done

echo "C1-H12-A2 3/3 structural equivalence PASS; results=$run_dir"
echo 'Scope: symbolic two-hop union only, not full Galerkin, AMG or PCG runtime.'
