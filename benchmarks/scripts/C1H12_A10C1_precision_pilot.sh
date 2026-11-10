#!/usr/bin/env bash
# C1-H12-A10C1: independent CSR roundoff/cancellation on frozen A10B corpus.
set -euo pipefail

mode="${1:-}"
a10b="${2:-}"
a10a="${3:-}"
a9="${4:-}"
corpus="${5:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$a10b" || -z "$a10a" || -z "$a9" ]]; then
    echo 'Usage: C1H12_A10C1_precision_pilot.sh --dry-run|--run FROZEN_A10B FROZEN_A10A FROZEN_A9 [CORPUS]' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || {
    echo 'Unexpected branch' >&2; exit 1;
}
[[ "$(git rev-parse --short HEAD)" == '60104c9' ]] || {
    echo 'Expected committed A10B checkpoint 60104c9' >&2; exit 1;
}
[[ -z "$(git diff --cached --name-only)" ]] || {
    echo 'Unexpected staged changes' >&2; exit 1;
}
frozen_head="$(git rev-parse '1f3eaa5^{commit}')"
for frozen in "$a9" "$a10a" "$a10b"; do
    [[ -s "$frozen/HEAD.txt" && -s "$frozen/matrix_sha256.txt" ]] || {
        echo "Missing frozen provenance: $frozen" >&2; exit 1;
    }
    [[ "$(<"$frozen/HEAD.txt")" == "$frozen_head" ]] || {
        echo "Frozen A9/A10B full HEAD mismatch (expected $frozen_head): $frozen" >&2; exit 1;
    }
done
cmp "$a9/matrix_sha256.txt" "$a10a/matrix_sha256.txt"
cmp "$a9/matrix_sha256.txt" "$a10b/matrix_sha256.txt"
sha256sum --check --strict --quiet "$a10b/matrix_sha256.txt"
[[ -s "$a10b/prepared_reuse_comparison.csv" ]] || {
    echo 'Missing accepted A10B 720-sample comparison' >&2; exit 1;
}
for case in island_shell multi_patch rotated_anisotropy; do
    [[ -s "$corpus/c1g_p1_${case}_n144.mtx" ]] || {
        echo "Missing FEM original $case" >&2; exit 1;
    }
done
sha256sum --check --strict --quiet SOURCE_SHA256.txt
echo 'C1-H12-A10C1 precision-floor preflight PASS'
echo "  repo=$repo; committed HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A10B=$a10b; historical HEAD=$(<"$a10b/HEAD.txt")"
echo '  3 matrices x 4 RHS = 12 independent CSR diagnostics'
echo '  Fixed A10B 720 samples; no repeated PCG runs or modified raw evidence'
echo '  Diagnostics: cancellation ratio, dot-product roundoff envelope, exact-vector Rayleigh quotient'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

git diff --check
bash -n benchmarks/scripts/C1H12_A10C1_precision_pilot.sh
python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A10B_prepared_gate \
    benchmarks.scripts.test_C1H12_A10C1_precision
evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10c1_precision_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
cp "$a10b/HEAD.txt" "$run_dir/FROZEN_HEAD.txt"
cp "$a10b/matrix_sha256.txt" "$run_dir/matrix_sha256.txt"
printf 'Independent Python CSR diagnostic; no solver execution, no Ritz/eigenvalue claims\n' > "$run_dir/policy.txt"

PYTHONPATH="$repo" python3 -B benchmarks/scripts/C1H12_A10C1_precision.py \
    "$corpus" "$a9" "$a10a" "$a10b" "$run_dir/precision_floor.csv"
PYTHONPATH="$repo" python3 -B benchmarks/scripts/C1H12_A10C1_precision_gate.py \
    "$run_dir/precision_floor.csv" "$a10b"
echo "C1-H12-A10C1 raw diagnostics: $run_dir"
echo 'Scope: estimates original CSR dot-product cancellation only; not proven PCG floor or ABTM speedup.'
