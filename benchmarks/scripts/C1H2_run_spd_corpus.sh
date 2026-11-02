#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hybit"
BIN=target/release/examples/fem_abtm_residual_observer
if [[ ! -x "$BIN" ]]; then
  echo "Build the example first: cargo build --release --locked -p hybit --example fem_abtm_residual_observer" >&2
  exit 2
fi
OUT="$HOME/c1h2_abtm_observer"
mkdir -p "$OUT"
for CASE in uniform_n48 local_soft_n72 island_shell_n72 multi_patch_n72 rotated_anisotropy_n72; do
  M="$HOME/mtx/c1g/c1g_p1_${CASE}"
  for FRAC in 0.05 0.2 0.5; do
    LOG="$OUT/${CASE}_fraction${FRAC}.log"
    echo "=== $CASE fraction=$FRAC ==="
    OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 "$BIN" \
      --matrix "$M.mtx" --rhs "$M.rhs.txt" --xref "$M.xref.txt" \
      --tol 1e-10 --max-iters 1200 --first-check 12 \
      --monitor-iters 64 --seed-fraction "$FRAC" | tee "$LOG"
    if ! grep -q '^PCG invariant      : PASS$' "$LOG"; then
      echo "FAIL: PCG invariant missing for $CASE fraction=$FRAC" >&2
      exit 1
    fi
  done
done
echo "C1-H2 observation-only corpus complete; logs: $OUT"
