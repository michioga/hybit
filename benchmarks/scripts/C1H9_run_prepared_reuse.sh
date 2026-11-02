#!/usr/bin/env bash
set -euo pipefail
cd "${HYBIT_REPO:-$HOME/hybit}"
export PYTHONDONTWRITEBYTECODE=1
cargo +1.99.1 test --release --locked -p hybit --test c1h9_prepared_weighted_reuse -- --nocapture
cargo +1.99.1 test --release --locked -p hybit-auto abtm_weighted_routing
bash benchmarks/scripts/C1H8_run_natural_region.sh smoke
bash benchmarks/scripts/C1H6_run_auto_route.sh
sha256sum --check --strict --quiet SOURCE_SHA256.txt
printf '%s\n' 'C1-H9 prepared weighted routing smoke gate PASS'
