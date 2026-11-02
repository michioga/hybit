#!/usr/bin/env bash
# C1-H5 opt-in experimental routing trial. NO production Auto changes.
set -euo pipefail
cd "$HOME/hybit"

cargo test --release --locked -p hybit --example c1h5_topology_routing_trial
cargo build --release --locked -p hybit --example c1h5_topology_routing_trial

OUT="$HOME/c1h5_routing_trial"
mkdir -p "$OUT"
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  target/release/examples/c1h5_topology_routing_trial | tee "$OUT/routing.log"

grep -Fxq 'C1-H5 routing control gate PASS (no production Auto claim)' "$OUT/routing.log"
grep -Fxq 'local rescue gate  : PASS' "$OUT/routing.log"
grep -Fxq 'keep PCG invariant : PASS' "$OUT/routing.log"

# The C1-H3 observations provide a *separate* healthy-case negative control.
# They do not contain the weighted structural-cut information. Do not treat
# this as an end-to-end test of a production ABTM promotion rule.
python3 -B benchmarks/scripts/C1H3_shadow_policy.py \
  --logs "$HOME/c1h2_abtm_observer" --out "$OUT/h3_negatives" \
  --rate-limit 0.995 --require-corpus

python3 - "$OUT/h3_negatives/summary.json" <<'PY'
from pathlib import Path
import json
import sys
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding='utf-8'))
assert d['run_count'] == 15, d
assert d['candidate_windows'] == 0, d
assert all(s['status'] == 'Converged' and s['invariant'] == 'PASS'
           and s['shadow_candidates'] == 0 for s in d['summaries']), d
print('C1-H5 C1-H3 negative-corpus guard PASS (15 observation-only logs)')
PY

sha256sum --check --strict --quiet SOURCE_SHA256.txt
printf '%s\n' 'C1-H5 routing experiment gate PASS (separate evidence arms; no Auto claim)'
