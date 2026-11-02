#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hybit"
python3 -B -m unittest discover -s benchmarks/scripts -p 'test_C1H3_shadow_policy.py' -v
python3 -B benchmarks/scripts/C1H3_shadow_policy.py \
  --logs "$HOME/c1h2_abtm_observer" \
  --out "$HOME/c1h3_shadow" \
  --rate-limit 0.995 --require-corpus
