#!/usr/bin/env python3
"""C1-H7: evidence-only robustness sweep of C1-H6 *real* HyBIT Auto.

Pure Python standard library; does not modify solver, matrix library, or rust
options. Every variant is an explicitly constructed SPD operator, and the same
RHS, iteration budget, tolerance and initial guess are used in both arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

N = 160
LOCAL = 96
BUDGET = 30
TOL = 1e-10


def system(kind: str, bridge: float, rhs: str) -> tuple[list[dict[int, float]], list[float], list[float]]:
    if kind not in ('local', 'global') or rhs not in ('smooth', 'flat', 'oscillatory'):
        raise ValueError('unsupported matrix or reference family')
    if not math.isfinite(bridge) or bridge <= 0:
        raise ValueError('bridge stiffness must be finite and positive')
    rows: list[dict[int, float]] = [dict() for _ in range(N)]
    diag = [2.0] * N
    for i in range(N - 1):
        if kind == 'global' or i < LOCAL - 1:
            coupling = 1.0
        elif i == LOCAL - 1:
            coupling = bridge
            diag[i] += bridge
            diag[i + 1] += bridge
        else:
            continue
        rows[i][i+1] = -coupling
        rows[i+1][i] = -coupling
    for i in range(N):
        rows[i][i] = diag[i]
    if rhs == 'flat':
        xref = [1.0] * N
    elif rhs == 'oscillatory':
        xref = [0.5 + 0.4 * math.sin(0.31 * i) + 0.13 * math.cos(0.09 * i) for i in range(N)]
    elif kind == 'global':
        xref = [math.sin(math.pi*(i+1)/(N+1)) + 0.11*math.sin(0.21*i) for i in range(N)]
    else:
        xref = [math.sin(math.pi*(i+1)/(LOCAL+1)) if i < LOCAL else
                0.7 + 0.2*math.sin(0.13*i) for i in range(N)]
    b = [sum(v * xref[j] for j, v in row.items()) for row in rows]
    return rows, b, xref


def emit_case(out: Path, kind: str, bridge: float, rhs: str) -> Path:
    rows, b, xref = system(kind, bridge, rhs)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f'c1h7_{kind}_{bridge:.0e}_{rhs}'
    entries = [(i+1, j+1, value) for i, row in enumerate(rows)
               for j, value in sorted(row.items())]
    with base.with_suffix('.mtx').open('w', encoding='utf-8') as f:
        f.write('%%MatrixMarket matrix coordinate real general\n')
        f.write('% C1-H7 controlled SPD Dirichlet chain plus PSD weak-edge Laplacian\n')
        f.write(f'{N} {N} {len(entries)}\n')
        for i, j, v in entries:
            f.write(f'{i} {j} {v:.17g}\n')
    for suffix, values in [('.rhs.txt', b), ('.xref.txt', xref)]:
        (out / (base.name + suffix)).write_text(
            ''.join(f'{v:.17g}\n' for v in values), encoding='utf-8')
    return base


def field(section: str, name: str) -> str:
    found = re.findall(r'^' + re.escape(name) + r'\s*:\s*(.+)$', section, re.M)
    if len(found) != 1:
        raise ValueError(f'{name}: expected exactly 1 line, got {len(found)}')
    return found[0].strip()


def parse_output(raw: str) -> dict[str, object]:
    if raw.count('[1/2] Plain Jacobi-PCG') != 1 or raw.count('[2/2] HyBIT Auto') != 1:
        raise ValueError('missing/duplicated benchmark arm markers')
    plain_text, auto_text = raw.split('[2/2] HyBIT Auto', 1)
    plain_text = plain_text.split('[1/2] Plain Jacobi-PCG', 1)[1]
    result: dict[str, object] = {}
    for prefix, part in [('plain', plain_text), ('auto', auto_text)]:
        status = field(part, 'status')
        if status not in ('Converged', 'MaxIterations', 'Breakdown'):
            raise ValueError('invalid solve status')
        result[prefix + '_status'] = status
        result[prefix + '_iterations'] = int(field(part, 'iterations'))
        result[prefix + '_residual'] = float(field(part, 'verified residual'))
        result[prefix + '_xerror'] = float(field(part, 'relative x error'))
        if prefix == 'auto':
            result['escalations'] = int(field(part, 'escalations'))
            factor_str = field(part, 'factor memory')
            result['factor_mib'] = float(factor_str.split()[0])
            result['wall_ms'] = float(field(part, 'total wall').split()[0])
    if (not all(math.isfinite(v) and v >= 0 for k, v in result.items()
                if k.endswith(('_residual', '_xerror')) or k in ('factor_mib', 'wall_ms'))
            or any(result[k] > BUDGET or result[k] < 0 for k in ('plain_iterations', 'auto_iterations'))
            or result['escalations'] not in (0, 1)):
        raise ValueError('invalid numeric telemetry')
    if result['escalations'] == 0:
        if (result['plain_status'] != result['auto_status'] or
                result['plain_iterations'] != result['auto_iterations']):
            raise ValueError('KEEP_PCG status/iteration mismatch; investigate recurrence')
    result['rescue'] = (
        result['plain_status'] == 'MaxIterations' and
        result['plain_residual'] > TOL and
        result['auto_status'] == 'Converged' and
        result['auto_residual'] <= TOL and
        result['auto_xerror'] <= 1e-6 and
        result['escalations'] == 1
    )
    result['unnecessary_promotion'] = (result['plain_status'] == 'Converged' and
                                     result['escalations'] > 0)
    return result


def run_one(binary: Path, base: Path, kind: str, bridge: float, rhs: str,
            capacity: int, probe: int, logdir: Path) -> dict[str, object]:
    name = f'{kind}_{bridge:.0e}_{rhs}_cap{capacity}_probe{probe}'
    argv = [str(binary), '--matrix', str(base) + '.mtx',
            '--rhs', str(base.parent / (base.name + '.rhs.txt')),
            '--xref', str(base.parent / (base.name + '.xref.txt')),
            '--tol', str(TOL), '--max-iters', str(BUDGET),
            '--probe-iters', str(probe), '--max-region', str(capacity),
            '--max-regions', '1', '--max-escalations', '1', '--overlap', '0',
            '--factor-budget-mib', '1', '--backend', 'csr', '--abtm-weighted-routing']
    environment = os.environ.copy()
    environment.update({'OMP_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1'})
    proc = subprocess.run(argv, capture_output=True, text=True, env=environment, timeout=45)
    logfile = logdir / (name + '.log')
    logfile.write_text(proc.stdout + '\n=== stderr ===\n' + proc.stderr, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'{name} returned {proc.returncode}: {logfile}')
    result = parse_output(proc.stdout)
    return {'case': name, 'kind': kind, 'bridge': bridge,
            'rhs': rhs, 'capacity': capacity, 'probe': probe,
            'strong_cut_expected': bridge/(2.0+bridge) >= 0.1 if kind == 'local' else True,
            'log': str(logfile), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    args = parser.parse_args()
    binary = (args.repo.resolve() / 'target/release/examples/fem_bench')
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f'missing built fem_bench: {binary}')
    out = args.output.expanduser().resolve()
    matrices = out / 'matrices'
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    if args.mode == 'smoke':
        grid = [('local', 1e-4, 'smooth', 96, 12),
                ('global', 1.0, 'smooth', 96, 12),
                ('local', 1e-4, 'smooth', 64, 12),
                ('local', 1e-4, 'flat', 96, 12),
                ('local', 0.5, 'smooth', 96, 12)]
    else:
        grid = [(kind, bridge, rhs, capacity, probe)
                for kind, bridges in [('local', (1e-6, 1e-4, 1e-2, 1e-1, 5e-1)),
                                      ('global', (1.0,))]
                for bridge in bridges for rhs in ('smooth', 'flat', 'oscillatory')
                for capacity in (64, 96, 128) for probe in (8, 12, 20)]
    cache: dict[tuple[str, float, str], Path] = {}
    results: list[dict[str, object]] = []
    for idx, (kind, bridge, rhs, capacity, probe) in enumerate(grid):
        key = kind, bridge, rhs
        if key not in cache:
            cache[key] = emit_case(matrices, *key)
        results.append(run_one(binary, cache[key], kind, bridge, rhs, capacity, probe, logs))
        if (idx + 1) % 25 == 0:
            print(f'C1-H7 sweep {idx+1}/{len(grid)}')
    golden = next(r for r in results if r['case'] == 'local_1e-04_smooth_cap96_probe12')
    assert golden['rescue'], 'C1-H6 canonical rescue regressed; see logged evidence'
    control = next(r for r in results if r['case'] == 'global_1e+00_smooth_cap96_probe12')
    assert control['escalations'] == 0, 'canonical global KEEP_PCG regressed'
    target = out / 'summary.csv'
    with target.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    recap = {'mode': args.mode, 'runs': len(results), 'rescue': sum(r['rescue'] for r in results),
             'unnecessary_promotion': sum(r['unnecessary_promotion'] for r in results),
             'promotions': sum(r['escalations'] for r in results),
             'no_promotion': sum(r['escalations'] == 0 for r in results),
             'golden_local_rescue': bool(golden['rescue']),
             'golden_global_keep': control['escalations'] == 0}
    (out / 'summary.json').write_text(json.dumps(recap, indent=2) + '\n', encoding='utf-8')
    print(f'C1-H7 {args.mode} PASS: {recap}; CSV: {target}')
    print('Note: outside two canonical controls, classifications are empirical, NOT predeclared passes.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.TimeoutExpired, AssertionError) as exc:
        print(f'C1-H7 FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
