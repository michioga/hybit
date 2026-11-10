#!/usr/bin/env bash
# C1-H12-A10C2 independent spectral/ABTM coarse-space research, no PCG runs.
set -euo pipefail

mode="${1:-}"
a10c1="${2:-}"
a10b="${3:-}"
corpus="${4:-$HOME/mtx/c1g_g2_hard_z32Jkd}"
if [[ ("$mode" != '--dry-run' && "$mode" != '--run') || -z "$a10c1" || -z "$a10b" ]]; then
    echo 'Usage: C1H12_A10C2_spectral_pilot.sh --dry-run|--run FROZEN_A10C1 FROZEN_A10B [ORIGINAL_FEM_CORPUS]' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[[ "$(git rev-parse --show-toplevel)" == "$repo" ]] || { echo 'Repository root mismatch' >&2; exit 1; }
[[ "$(git branch --show-current)" == 'develop/0.10.0' ]] || { echo 'Unexpected branch' >&2; exit 1; }
[[ "$(git rev-parse --short HEAD)" == '60104c9' ]] || { echo 'Unexpected committed A10B checkpoint' >&2; exit 1; }
[[ -z "$(git diff --cached --name-only)" ]] || { echo 'Staged changes require review' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt

frozen_a10b_head="$(git rev-parse '1f3eaa5^{commit}')"
frozen_a10c1_head="$(git rev-parse '60104c9^{commit}')"
[[ -s "$a10b/HEAD.txt" && "$(<"$a10b/HEAD.txt")" == "$frozen_a10b_head" ]] || {
    echo 'A10B frozen historical HEAD mismatch' >&2; exit 1;
}
[[ -s "$a10c1/HEAD.txt" && "$(<"$a10c1/HEAD.txt")" == "$frozen_a10c1_head" ]] || {
    echo 'A10C1 frozen historical HEAD mismatch' >&2; exit 1;
}
[[ -s "$a10b/matrix_sha256.txt" && -s "$a10c1/matrix_sha256.txt" ]] || {
    echo 'Missing frozen original-CSR SHA manifest' >&2; exit 1;
}
cmp "$a10b/matrix_sha256.txt" "$a10c1/matrix_sha256.txt"
[[ -s "$a10c1/precision_floor.csv" ]] || { echo 'Missing accepted A10C1 evidence' >&2; exit 1; }
for case in island_shell multi_patch rotated_anisotropy; do
    [[ -s "$corpus/c1g_p1_${case}_n144.mtx" ]] || { echo "Missing original FEM: $case" >&2; exit 1; }
done

echo 'C1-H12-A10C2 independent spectral / ABTM coarse-space preflight PASS'
echo "  repository=$repo; committed HEAD=$(git rev-parse --short HEAD)"
echo "  frozen A10C1=$a10c1; A10B=$a10b"
echo '  3 original FEM x 2 fixed Lanczos starts = 6 Ritz records'
echo '  3 FEM x 3 ABTM aggregate/basis arms x 3 low-energy probes = 27 D-projection records'
echo '  48-step, two-pass reorthogonalized symmetric Lanczos; u64-word bit adjacency'
echo '  no PCG replays, raw evidence updates, production/API/default edits or speedup claim'
if [[ "$mode" == '--dry-run' ]]; then
    echo 'DRY RUN: no repository or evidence files changed'
    exit 0
fi

git diff --check
bash -n benchmarks/scripts/C1H12_A10C2_spectral_pilot.sh
python3 -B -m unittest -v \
    benchmarks.scripts.test_C1H12_A10C1_precision \
    benchmarks.scripts.test_C1H12_A10C2_spectral

evidence_root="${HYBIT_C1H12_EVIDENCE_ROOT:-$HOME/c1h11_evidence}"
mkdir -p "$evidence_root"
run_dir="$(mktemp -d "$evidence_root/c1h12_a10c2_spectral_XXXXXX")"
git rev-parse HEAD > "$run_dir/HEAD.txt"
cp "$a10b/HEAD.txt" "$run_dir/FROZEN_A10B_HEAD.txt"
cp "$a10c1/HEAD.txt" "$run_dir/FROZEN_A10C1_HEAD.txt"
sha256sum "$a10c1/precision_floor.csv" | awk '{print $1}' > "$run_dir/FROZEN_A10C1_SHA256.txt"
cp "$a10b/matrix_sha256.txt" "$run_dir/matrix_sha256.txt"
printf 'Ritz estimates of D^-1/2 A D^-1/2, original CSR; not eigenvalue extremal certificates\n' > "$run_dir/policy.txt"

PYTHONPATH="$repo" python3 -B benchmarks/scripts/C1H12_A10C2_spectral.py \
    "$corpus" "$a10b" "$a10c1" "$run_dir"
PYTHONPATH="$repo" python3 -B benchmarks/scripts/C1H12_A10C2_spectral_gate.py \
    "$run_dir" "$a10b" "$a10c1"
echo "C1-H12-A10C2 preserved evidence: $run_dir"
