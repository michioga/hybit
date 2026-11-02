#!/usr/bin/env python3
"""C1-H8: evaluate closed strong-ABTM regions of varying natural size.

Research corpus only: synthetic symmetric positive definite chain problems, not
real FEM / production solver policy. Uses actual HyBIT Auto opt-in executable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .C1H7_sensitivity import parse_output

N = 160
TOL = 1.0e-10
BUDGET = 30
BRIDGE = 1.0e-4
LOCAL_SIZES = (64, 96, 128)
CAPACITIES = (64, 96, 128)
RHS = ('smooth', 'flat', 'oscillatory')


def make_system(local: int | None, rhs_kind: str) -> tuple[list[dict[int, float]], list[float], list[float]]:
    if (local is not None and (local not in LOCAL_SIZES)) or rhs_kind not in RHS:
        raise ValueError('unsupported natural-region control')
    rows: list[dict[int, float]] = [dict() for _ in range(N)]
    diagonal = [2.0] * N
    for i in range(N - 1):
        if local is None or i < local - 1:
            edge = 1.0
        elif i == local - 1:
            edge = BRIDGE
            diagonal[i] += edge
            diagonal[i + 1] += edge
        else:
            continue
        rows[i][i + 1] = -edge
        rows[i + 1][i] = -edge
    for i in range(N):
        rows[i][i] = diagonal[i]

    if rhs_kind == 'flat':
        xref = [1.0] * N
    elif rhs_kind == 'oscillatory':
        xref = [0.5 + 0.4 * math.sin(0.31 * i) + 0.13 * math.cos(0.09 * i)
                for i in range(N)]
    elif local is None:
        xref = [math.sin(math.pi * (i + 1) / (N + 1)) + 0.11 * math.sin(0.21 * i)
                for i in range(N)]
    else:
        xref = [math.sin(math.pi * (i + 1) / (local + 1)) if i < local else
                0.7 + 0.2 * math.sin(0.13 * i) for i in range(N)]
    b = [sum(val * xref[j] for j, val in row.items()) for row in rows]
    return rows, b, xref


def write_case(out: Path, local: int | None, rhs_kind: str) -> Path:
    rows, b, reference = make_system(local, rhs_kind)
    name = f'c1h8_{"global" if local is None else "local" + str(local)}_{rhs_kind}'
    base = out / name
    out.mkdir(parents=True, exist_ok=True)
    entries = [(i + 1, j + 1, val) for i, row in enumerate(rows)
               for j, val in sorted(row.items())]
    (out / (name + '.mtx')).write_text(
        '%%MatrixMarket matrix coordinate real general\n'
        '% C1-H8 constructed SPD: Dirichlet local chain plus PSD weak bridge\n'
        f'{N} {N} {len(entries)}\n' +
        ''.join(f'{i} {j} {v:.17g}\n' for i, j, v in entries), encoding='utf-8')
    for suffix, values in (('.rhs.txt', b), ('.xref.txt', reference)):
        (out / (name + suffix)).write_text(''.join(f'{v:.17g}\n' for v in values),
                                               encoding='utf-8')
    return base


def grid(mode: str) -> list[tuple[int | None, str, int]]:
    if mode == 'smoke':
        return [(96, 'smooth', 96), (96, 'smooth', 128),
                (96, 'smooth', 64), (64, 'smooth', 128), (None, 'smooth', 128)]
    if mode == 'full':
        return [(local, rhs, cap) for local in (*LOCAL_SIZES, None)
                for rhs in RHS for cap in CAPACITIES]
    raise ValueError('unknown mode')


def classify(result: dict[str, object], local: int | None, cap: int) -> str:
    if local is None:
        return 'global_keep' if result['escalations'] == 0 else 'global_promotion'
    if cap < local:
        return 'capped_keep' if result['escalations'] == 0 else 'truncated_promotion'
    if result['rescue']:
        return 'rescue'
    if result['unnecessary_promotion']:
        return 'unnecessary_promotion'
    if result['escalations'] == 0:
        return 'unpromoted'
    return 'promoted_not_rescued'


def run_one(bin_path: Path, base: Path, local: int | None, rhs: str, cap: int,
            log_dir: Path) -> dict[str, object]:
    name = f'{"global" if local is None else "local" + str(local)}_{rhs}_cap{cap}'
    argv = [str(bin_path), '--matrix', str(base) + '.mtx',
            '--rhs', str(base.parent / (base.name + '.rhs.txt')),
            '--xref', str(base.parent / (base.name + '.xref.txt')),
            '--tol', str(TOL), '--max-iters', str(BUDGET),
            '--probe-iters', '12', '--max-region', str(cap),
            '--max-regions', '1', '--max-escalations', '1', '--overlap', '0',
            '--factor-budget-mib', '1', '--backend', 'csr', '--abtm-weighted-routing']
    env = dict(os.environ, OMP_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=45)
    log = log_dir / (name + '.log')
    log.write_text(proc.stdout + '\n=== stderr ===\n' + proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f'{name} failed ({proc.returncode}); see {log}')
    measures = parse_output(proc.stdout)
    return {'case': name, 'local_dofs': local if local is not None else 'global',
            'rhs': rhs, 'capacity': cap, 'log': str(log), **measures,
            'classification': classify(measures, local, cap)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    args = parser.parse_args()
    binary = args.repo.resolve() / 'target/release/examples/fem_bench'
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f'build fem_bench first: {binary}')
    out = args.output.expanduser().resolve()
    log_dir = out / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    inputs = out / 'inputs'
    cache: dict[tuple[int | None, str], Path] = {}
    results: list[dict[str, object]] = []
    for local, rhs, cap in grid(args.mode):
        key = local, rhs
        if key not in cache:
            cache[key] = write_case(inputs, *key)
        results.append(run_one(binary, cache[key], local, rhs, cap, log_dir))

    by_name = {row['case']: row for row in results}
    assert by_name['local96_smooth_cap96']['rescue'], 'C1-H6 canonical positive regressed'
    assert by_name['local96_smooth_cap128']['rescue'], 'natural-size positive rescue failed'
    assert by_name['local96_smooth_cap64']['escalations'] == 0, 'capped region was admitted'
    assert by_name['global_smooth_cap128']['escalations'] == 0, 'global strong chain admitted'
    assert all(r['escalations'] == 0 for r in results if r['classification'] in
               ('global_keep', 'capped_keep')), 'unexpected promotion'
    assert all(r['classification'] not in ('global_promotion', 'truncated_promotion')
               for r in results), 'incomplete/strong component was promoted'

    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    classes = {name: sum(r['classification'] == name for r in results)
               for name in sorted({r['classification'] for r in results})}
    report = {'mode': args.mode, 'runs': len(results),
              'classes': classes, 'promotions': sum(int(r['escalations']) for r in results),
              'rescues': sum(bool(r['rescue']) for r in results),
              'unnecessary_promotions': sum(bool(r['unnecessary_promotion']) for r in results),
              'canonical_rescue': True, 'natural_capacity_rescue': True,
              'truncation_control': True, 'global_control': True}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'C1-H8 {args.mode} analysis PASS: {report}; CSV: {out / "summary.csv"}')
    print('Research observations outside four controls; no general FEM or speedup claim.')


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f'C1-H8 FAIL: {error}', file=sys.stderr)
        raise SystemExit(1) from error
