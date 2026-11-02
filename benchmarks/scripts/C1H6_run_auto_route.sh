#!/usr/bin/env bash
set -euo pipefail
cd "${HYBIT_REPO:-$HOME/hybit}"
OUT="${C1H6_OUT:-$HOME/c1h6_abtm_auto}"
mkdir -p "$OUT"
python3 benchmarks/scripts/C1H6_generate_cases.py --output "$OUT"

cargo test --release --locked -p hybit-auto abtm_weighted_routing
cargo build --release --locked -p hybit --example fem_bench
BIN=target/release/examples/fem_bench
for NAME in localized_chain global_chain; do
  BASE="$OUT/c1h6_$NAME"
  echo "=== C1-H6 Auto opt-in $NAME ==="
  OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 "$BIN" \
    --matrix "$BASE.mtx" --rhs "$BASE.rhs.txt" --xref "$BASE.xref.txt" \
    --tol 1e-10 --max-iters 30 --probe-iters 12 \
    --max-region 96 --max-regions 1 --max-escalations 1 \
    --overlap 0 --factor-budget-mib 1 --backend csr \
    --abtm-weighted-routing | tee "$OUT/$NAME.log"
done
python3 - "$OUT" <<'PY'
from pathlib import Path
import re
import sys
out = Path(sys.argv[1])
for name, want_status, want_esc in [
    ('localized_chain', 'Converged', 1),
    ('global_chain', 'MaxIterations', 0),
]:
    text = (out / f'{name}.log').read_text()
    assert '[1/2] Plain Jacobi-PCG' in text and '[2/2] HyBIT Auto' in text, name
    plain, hybrid = text.split('[2/2] HyBIT Auto', 1)
    assert re.search(r'^status\s+: MaxIterations$', plain, re.M), name
    assert re.search(rf'^status\s+: {want_status}$', hybrid, re.M), name
    assert re.search(rf'^escalations\s+: {want_esc}$', hybrid, re.M), name
    rows = re.findall(r'^verified residual\s+:\s+([0-9.eE+-]+)$', hybrid, re.M)
    assert len(rows) == 1, (name, rows)
    if name == 'localized_chain':
        assert float(rows[0]) <= 1e-10, (name, rows[0])
    print(f'C1-H6 {name}: status={want_status} escalations={want_esc} verified={rows[0]} PASS')
PY

# Directly apply the exact C1-H6 opt-in policy to the five C1-G matrices.
# Unlike C1-H3 offline negatives, these runs exercise the same Auto code.
for CASE in uniform_n48 local_soft_n72 island_shell_n72 multi_patch_n72 rotated_anisotropy_n72; do
  BASE="$HOME/mtx/c1g/c1g_p1_$CASE"
  if [[ ! -f "$BASE.mtx" || ! -f "$BASE.rhs.txt" ]]; then
    echo "Missing C1-G corpus: $BASE" >&2
    exit 2
  fi
  OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 "$BIN" \
    --matrix "$BASE.mtx" --rhs "$BASE.rhs.txt" \
    --tol 1e-10 --max-iters 1200 --probe-iters 12 \
    --max-region 96 --max-regions 1 --max-escalations 1 \
    --overlap 0 --factor-budget-mib 1 --backend csr \
    --abtm-weighted-routing | tee "$OUT/c1g_$CASE.log"
done
python3 - "$OUT" <<'PY'
from pathlib import Path
import re
import sys
out = Path(sys.argv[1])
for name in ('uniform_n48','local_soft_n72','island_shell_n72','multi_patch_n72','rotated_anisotropy_n72'):
    text = (out / f'c1g_{name}.log').read_text()
    assert '[2/2] HyBIT Auto' in text, name
    auto = text.split('[2/2] HyBIT Auto', 1)[1]
    m = re.search(r'^escalations\s+:\s+(\d+)$', auto, re.M)
    assert m, name
    assert int(m.group(1)) == 0, f'{name}: unexpected opt-in promotion; inspect {out}/c1g_{name}.log'
    print(f'C1-H6 C1-G control {name}: no promotion PASS')
PY
printf '%s\n' 'C1-H6 opt-in Auto gate PASS (withheld until actual WSL2 run)'
