#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: bash benchmarks/scripts/P2B_pilot.sh --dry-run|--run [case_manifest.tsv]' >&2
  exit 2
}
[[ $# == 1 || $# == 2 ]] || usage
MODE=$1
[[ $MODE == '--dry-run' || $MODE == '--run' ]] || usage
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$REPO"
MANIFEST=${2:-benchmarks/scripts/P2B_cases.tsv}
[[ -r $MANIFEST ]] || { echo "missing case manifest: $MANIFEST" >&2; exit 1; }
[[ $(git branch --show-current) == 'develop/0.10.0' ]] || { echo 'wrong branch' >&2; exit 1; }
git merge-base --is-ancestor fdb047e8af63d38abe8af34e33b7f5a5b7e0b799 HEAD || { echo 'P2-B requires descendant of fdb047e' >&2; exit 1; }
[[ -z $(git diff --cached --name-only) ]] || { echo 'staged changes exist' >&2; exit 1; }
sha256sum --check --strict --quiet SOURCE_SHA256.txt
python3 -B -m unittest -q benchmarks.scripts.test_P2B_gate

# Read source manifest into canonical absolute paths, without evaluating any shell.
CASES=$(mktemp)
trap 'rm -f -- "$CASES"' EXIT
count=0
while IFS=$'\t' read -r label matrix rhs xref || [[ -n $label ]]; do
  [[ -z $label || $label == \#* ]] && continue
  [[ $label =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] || { echo "invalid case name: $label" >&2; exit 1; }
  for variable in matrix rhs xref; do
    path=${!variable}
    [[ -n $path ]] || { echo "missing path: $label.$variable" >&2; exit 1; }
    if [[ $path == '$HOME/'* ]]; then
      path="$HOME/${path#\$HOME/}"
    fi
    [[ -f $path ]] || { echo "missing input: $label.$variable: $path" >&2; exit 1; }
    printf -v "$variable" '%s' "$(realpath "$path")"
  done
  [[ $matrix != *$'\t'* && $rhs != *$'\t'* && $xref != *$'\t'* ]] || {
    echo 'tab in path not supported' >&2; exit 1;
  }
  if awk -F '\t' -v l="$label" '$1 == l {found=1} END {exit !found}' "$CASES"; then
    echo "duplicate case: $label" >&2; exit 1
  fi
  printf '%s\t%s\t%s\t%s\n' "$label" "$matrix" "$rhs" "$xref" >> "$CASES"
  count=$((count + 1))
done < "$MANIFEST"
(( count > 0 )) || { echo 'no valid cases in manifest' >&2; exit 1; }

echo "P2-B numerical preflight PASS: HEAD=$(git rev-parse --short=7 HEAD); $count cases x feature OFF/ON"
echo '  fixed: original CSR, exact same matrix/RHS/xref, tolerance 1e-10, 600 iterations'
echo '  zero RHS / nonzero x: separate Jacobi-only observation in both feature modes'
cat "$CASES"
if [[ $MODE == '--dry-run' ]]; then
  echo 'DRY RUN: no source, evidence or build artifacts modified'
  exit 0
fi

for executable in cargo timeout /usr/bin/time sha256sum; do
  command -v "$executable" > /dev/null || { echo "missing dependency: $executable" >&2; exit 1; }
done
mkdir -p "$HOME/c1h11_evidence"
RUN_DIR=$(mktemp -d "$HOME/c1h11_evidence/p2b_paired_XXXXXX")
echo "P2-B output: $RUN_DIR"
cp -- "$CASES" "$RUN_DIR/cases_effective.tsv"
cp -- "$MANIFEST" "$RUN_DIR/cases_source.tsv"
git rev-parse HEAD > "$RUN_DIR/git_head.txt"
git status --short --branch > "$RUN_DIR/git_status.txt"
sha256sum benchmarks/scripts/P2B_gate.py benchmarks/scripts/P2B_pilot.sh \
  crates/hybit/examples/p2b_zero_rhs_probe.rs > "$RUN_DIR/harness_sha256.txt"
while IFS=$'\t' read -r label matrix rhs xref; do
  sha256sum "$matrix" "$rhs" "$xref"
done < "$CASES" > "$RUN_DIR/input_sha256.txt"

for mode in off on; do
  features=()
  [[ $mode == on ]] && features=(--features p2-certified-restart)
  echo "=== build feature $mode ==="
  cargo +1.99.1 build --release --locked -p hybit \
    --example fem_bench --example p2b_zero_rhs_probe \
    "${features[@]}" > "$RUN_DIR/build_${mode}.log" 2>&1 || {
      cat "$RUN_DIR/build_${mode}.log" >&2; exit 1;
    }
  mkdir -p "$RUN_DIR/bin_$mode"
  cp -- target/release/examples/fem_bench "$RUN_DIR/bin_$mode/fem_bench"
  cp -- target/release/examples/p2b_zero_rhs_probe "$RUN_DIR/bin_$mode/p2b_zero_rhs_probe"
  sha256sum "$RUN_DIR/bin_$mode/"* >> "$RUN_DIR/binary_sha256.txt"

done

for mode in off on; do
  echo "=== zero-RHS contract: $mode ==="
  /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
    -o "$RUN_DIR/zero_${mode}.time" \
    timeout --signal=TERM --kill-after=10s 90s \
    "$RUN_DIR/bin_$mode/p2b_zero_rhs_probe" \
    > "$RUN_DIR/zero_${mode}.log" 2>&1 || {
      cat "$RUN_DIR/zero_${mode}.log" >&2; exit 1;
    }
  rg '^P2B_CASE' "$RUN_DIR/zero_${mode}.log" || true
  while IFS=$'\t' read -r label matrix rhs xref; do
    echo "=== P2-B case=$label mode=$mode ==="
    /usr/bin/time -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
      -o "$RUN_DIR/${label}_${mode}.time" \
      timeout --signal=TERM --kill-after=10s 240s \
      "$RUN_DIR/bin_$mode/fem_bench" \
        --matrix "$matrix" --rhs "$rhs" --xref "$xref" \
        --tol 1e-10 --max-iters 600 --skip-plain --backend csr \
        > "$RUN_DIR/${label}_${mode}.log" 2>&1 || {
          cat "$RUN_DIR/${label}_${mode}.log" >&2; exit 1;
        }
    grep -E '^(status|iterations|verified residual|relative x error)\s*:' \
      "$RUN_DIR/${label}_${mode}.log" || true
  done < "$CASES"
done
python3 -B benchmarks/scripts/P2B_gate.py "$RUN_DIR" --tol 1e-10 --max-iters 600
printf '\nP2-B evidence directory: %s\n' "$RUN_DIR"
echo 'No commit/push performed. Preserve raw .log/.time and hashes for review.'
