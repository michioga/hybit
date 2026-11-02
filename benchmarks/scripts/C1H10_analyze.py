#!/usr/bin/env python3
"""Validate C1-H10 repeated-RHS timing CSV; report descriptive median comparisons."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys

MODES = ('plain', 'fresh', 'prepared')
FIELDS = ('wall_ms', 'analysis_ms', 'prepare_ms', 'diagnostics_ms',
          'factor_ms', 'probe_ms', 'solve_ms', 'max_true_residual', 'max_forward_error')
COUNTS = ('iterations', 'converged', 'promotions', 'reuse_events', 'peak_factor_bytes')
COLUMNS = ('mode', 'rhs_count', 'repeat', 'warmup', *FIELDS[:7], *COUNTS, *FIELDS[7:])


def read_rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f'invalid C1-H10 CSV header: {reader.fieldnames}')
        rows = []
        for row in reader:
            if set(row) != set(COLUMNS) or any(v is None for v in row.values()):
                raise ValueError('incomplete C1-H10 CSV row')
            if row['mode'] not in MODES:
                raise ValueError('unrecognized mode')
            converted = {'mode': row['mode']}
            for key in ('rhs_count', 'repeat', 'warmup', *COUNTS):
                converted[key] = int(row[key])
            for key in FIELDS:
                converted[key] = float(row[key])
                if not math.isfinite(converted[key]) or converted[key] < 0:
                    raise ValueError(f'nonfinite/negative {key}')
            if converted['rhs_count'] < 1 or converted['repeat'] < 0 or converted['warmup'] not in (0, 1):
                raise ValueError('invalid experiment identifiers')
            if any(converted[k] < 0 for k in COUNTS):
                raise ValueError('negative count')
            if converted['converged'] > converted['rhs_count'] or converted['promotions'] > converted['rhs_count']:
                raise ValueError('out-of-range event count')
            if converted['reuse_events'] > converted['promotions']:
                raise ValueError('reuse exceeds promotions')
            if converted['iterations'] > converted['rhs_count'] * 30:
                raise ValueError('iteration budget overflow')
            rows.append(converted)
    if not rows:
        raise ValueError('no timing rows')
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = {(r['rhs_count'], r['repeat'], r['warmup']) for r in rows}
    for key in keys:
        arms = [r for r in rows if (r['rhs_count'], r['repeat'], r['warmup']) == key]
        if len(arms) != 3 or {r['mode'] for r in arms} != set(MODES):
            raise ValueError(f'missing/duplicate arm in {key}')
        for arm in arms:
            if arm['mode'] == 'plain' and (arm['promotions'] or arm['reuse_events']):
                raise ValueError('plain PCG must not promote or reuse')
            if arm['mode'] == 'fresh' and arm['reuse_events']:
                raise ValueError('fresh Auto must not reuse across RHS')
            if arm['mode'] == 'prepared' and key[0] > 1 and arm['reuse_events'] < 1:
                raise ValueError('matching prepared factor was not reused')
        vals = {r['mode']: r for r in arms}
        if (vals['fresh']['converged'] != vals['prepared']['converged']
                or vals['fresh']['promotions'] != vals['prepared']['promotions']
                or vals['fresh']['iterations'] != vals['prepared']['iterations']):
            raise ValueError('fresh vs prepared solve mismatch')
    output = []
    for count in sorted({r['rhs_count'] for r in rows}):
        sample = [r for r in rows if r['rhs_count'] == count and r['warmup'] == 0]
        ids = sorted({r['repeat'] for r in sample})
        if len(ids) < 3:
            raise ValueError('at least 3 measured repetitions required')
        if any(len([r for r in sample if r['repeat'] == i]) != 3 for i in ids):
            raise ValueError('missing measurements')
        by_mode = {mode: [r for r in sample if r['mode'] == mode] for mode in MODES}
        median = {mode: statistics.median(r['wall_ms'] for r in records)
                  for mode, records in by_mode.items()}
        if min(median.values()) <= 0:
            raise ValueError('median wall time not positive')
        # Medians are descriptive: no significance claim or automatic optimization.
        output.append({
            'rhs_count': count,
            'repetitions': len(ids),
            'plain_median_ms': median['plain'],
            'fresh_median_ms': median['fresh'],
            'prepared_median_ms': median['prepared'],
            'prepared_over_fresh': median['prepared'] / median['fresh'],
            'prepared_over_plain': median['prepared'] / median['plain'],
            'fresh_factor_median_ms': statistics.median(r['factor_ms'] for r in by_mode['fresh']),
            'prepared_factor_median_ms': statistics.median(r['factor_ms'] for r in by_mode['prepared']),
            'prepared_reuse_events': min(r['reuse_events'] for r in by_mode['prepared']),
            'fresh_promotions': min(r['promotions'] for r in by_mode['fresh']),
            'prepared_promotions': min(r['promotions'] for r in by_mode['prepared']),
            'fresh_converged': min(r['converged'] for r in by_mode['fresh']),
            'prepared_converged': min(r['converged'] for r in by_mode['prepared']),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = summarize(read_rows(args.raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        print(f"C1-H10 RHS={r['rhs_count']:>2} plain={r['plain_median_ms']:.4f}ms "
              f"fresh={r['fresh_median_ms']:.4f}ms prepared={r['prepared_median_ms']:.4f}ms "
              f"prepared/fresh={r['prepared_over_fresh']:.3f} "
              f"prepared/plain={r['prepared_over_plain']:.3f} "
              f"minimum cache reuses={r['prepared_reuse_events']}")
    print(f'C1-H10 TIMING DATA VALIDATION PASS: {args.output}')
    print('CAUTION: medians from small synthetic SPD, not a speedup or real-FEM proof.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, csv.Error) as exc:
        print(f'C1-H10 ANALYSIS FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
