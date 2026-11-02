#!/usr/bin/env bash
# Evidence-only. Never imply a machine-generated SPD certificate or physical RHS.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  build)
    cargo +1.99.1 build --release --locked -p hybit \
      --example mtx_fem_audit --example fem_bench
    python3 -B -m unittest -v benchmarks.scripts.test_C1H11_real_fem_evidence
    ;;
  audit|run)
    mode="$1"; shift
    python3 -B benchmarks/scripts/C1H11_real_fem_evidence.py \
      --repo "$PWD" --mode "$mode" "$@"
    ;;
  *)
    echo 'Usage: bash benchmarks/scripts/C1H11_run_real_fem.sh build' >&2
    echo 'or: bash benchmarks/scripts/C1H11_run_real_fem.sh audit|run --manifest FILE --out NEW_DIR [--budgets 120 360 1200 --caps 128 256 512]' >&2
    exit 2
    ;;
esac
