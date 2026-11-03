#!/usr/bin/env bash
# C1-H12: gated research comparison of canonical ILU(0) and ABTM word operations.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash benchmarks/scripts/C1H12_run_abtm_ilu0_word_ops.sh MATRIX.mm [OUTPUT_DIR]" >&2
  exit 2
fi

MATRIX_INPUT="$1"
if [[ ! -f "$MATRIX_INPUT" ]]; then
  echo "Matrix not found: $MATRIX_INPUT" >&2
  exit 2
fi
MATRIX_INPUT="$(realpath "$MATRIX_INPUT")"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -eq 2 ]]; then
  OUTPUT_DIR="$2"
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
else
  EVIDENCE_BASE="${HOME}/c1h11_evidence"
  mkdir -p "$EVIDENCE_BASE"
  OUTPUT_DIR="$(mktemp -d "$EVIDENCE_BASE/C1H12_word_ops_XXXXXX")"
fi
echo "Evidence directory: $OUTPUT_DIR"

git rev-parse HEAD > "$OUTPUT_DIR/git_head.txt"
git status --short --branch > "$OUTPUT_DIR/git_status.txt"
sha256sum "$MATRIX_INPUT" > "$OUTPUT_DIR/matrix.sha256"

cargo +1.99.1 build --release --locked -p hybit \
  --features ilu0-abtm-experiments \
  --example abtm_ilu0_word_ops_experiment

BENCH="$REPO_ROOT/target/release/examples/abtm_ilu0_word_ops_experiment"
ARMS=(csr legacy-lut direct-lut direct-popcnt direct-rotate-popcnt)

# Correctness gates run separately so the reference factor does not contaminate RSS.
for arm in "${ARMS[@]}"; do
  "$BENCH" --matrix "$MATRIX_INPUT" --arm "$arm" \
    --verify --warmups 0 --repeats 1 \
    > "$OUTPUT_DIR/verify_${arm}.log" 2>&1
  grep '^C1H12_VERIFY' "$OUTPUT_DIR/verify_${arm}.log"
done

# Rotate launch order across 3 rounds. Each run is a separate timed process.
# Memory here is process peak RSS, including matrix parsing and factor preparation.
for round in 0 1 2; do
  for index in 0 1 2 3 4; do
    arm="${ARMS[$(((index + round) % ${#ARMS[@]}))]}"
    prefix="$OUTPUT_DIR/r${round}_${arm}"
    /usr/bin/time \
      -f 'elapsed_s=%e peak_rss_kib=%M exit_status=%x' \
      -o "${prefix}.time" \
      "$BENCH" --matrix "$MATRIX_INPUT" --arm "$arm" \
        --warmups 2 --repeats 7 \
      > "${prefix}.log" 2>&1
    echo "round=$round arm=$arm"
    grep '^C1H12_RESULT' "${prefix}.log"
    cat "${prefix}.time"
  done
done

echo "C1-H12 evidence completed: $OUTPUT_DIR"
