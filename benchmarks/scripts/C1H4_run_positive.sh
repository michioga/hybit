#!/usr/bin/env bash
set -euo pipefail
cd "${HYBIT_REPO:-$HOME/hybit}"
cargo test --release --locked -p hybit --example c1h4_weighted_local_rescue
cargo build --release --locked -p hybit --example c1h4_weighted_local_rescue
mkdir -p "${HYBIT_C1H4_OUT:-$HOME/c1h4_weighted_rescue}"
LOG="${HYBIT_C1H4_OUT:-$HOME/c1h4_weighted_rescue}/result.log"
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  target/release/examples/c1h4_weighted_local_rescue | tee "$LOG"
grep -qx 'local rescue gate  : PASS' "$LOG"
grep -qx 'qualification      : independent example, not HyBIT Auto routing' "$LOG"
sha256sum --check --strict --quiet SOURCE_SHA256.txt
printf '%s\n' 'C1-H4 independent local-direct rescue gate PASS (not an auto-routing claim)'
