#!/usr/bin/env bash
set -euo pipefail
cd "${HYBIT_REPO:-$HOME/hybit}"
MODE="${1:-smoke}"
case "$MODE" in smoke|full) ;; *) echo 'usage: C1H8_run_natural_region.sh smoke|full' >&2; exit 2 ;; esac
python3 -B -m unittest -v benchmarks.scripts.test_C1H8_natural_region
cargo +1.99.1 build --release --locked -p hybit --example fem_bench
python3 -B -m benchmarks.scripts.C1H8_natural_region \
  --mode "$MODE" \
  --repo "$PWD" \
  --output "${C1H8_OUT:-$HOME/c1h8_natural_region}/$MODE"
