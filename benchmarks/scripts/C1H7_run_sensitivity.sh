#!/usr/bin/env bash
set -euo pipefail
cd "${HYBIT_REPO:-$HOME/hybit}"
OUT="${C1H7_OUT:-$HOME/c1h7_abtm_sensitivity}"
MODE="${1:-smoke}"
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo 'Usage: bash benchmarks/scripts/C1H7_run_sensitivity.sh [smoke|full]' >&2
  exit 2
fi
python3 -m unittest -v benchmarks/scripts/test_C1H7_sensitivity.py
cargo +1.99.1 build --release --locked -p hybit --example fem_bench
python3 benchmarks/scripts/C1H7_sensitivity.py --repo "$PWD" --output "$OUT/$MODE" --mode "$MODE"
sha256sum --check --strict --quiet SOURCE_SHA256.txt
printf 'C1-H7 %s sensitivity analysis PASS (no production policy changes)\n' "$MODE"
