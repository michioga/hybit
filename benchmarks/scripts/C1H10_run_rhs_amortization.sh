#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "${HYBIT_REPO:-$HOME/hybit}"
MODE="${1:-smoke}"
case "$MODE" in
  smoke) COUNTS=1,2,4; REPEATS=3; WARMUPS=1 ;;
  full)  COUNTS=1,2,4,8,16,32; REPEATS=11; WARMUPS=2 ;;
  *) echo 'Usage: C1H10_run_rhs_amortization.sh smoke|full' >&2; exit 2 ;;
esac
OUT="${C1H10_OUT:-$HOME/c1h10_rhs_amortization}/$MODE"
mkdir -p "$OUT"
python3 -B -m unittest -v benchmarks.scripts.test_C1H10_analyze
cargo +1.99.1 test --release --locked -p hybit --example c1h10_rhs_amortization
cargo +1.99.1 build --release --locked -p hybit --example c1h10_rhs_amortization
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 target/release/examples/c1h10_rhs_amortization \
  --counts "$COUNTS" --repeats "$REPEATS" --warmups "$WARMUPS" > "$OUT/raw.csv"
python3 -B benchmarks/scripts/C1H10_analyze.py --raw "$OUT/raw.csv" --output "$OUT/summary.csv"
printf '%s\n' "C1-H10 $MODE gate PASS; raw=$OUT/raw.csv summary=$OUT/summary.csv"
