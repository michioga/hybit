#!/usr/bin/env python3
"""C1-H3: evidence-only, offline ABTM-hotspot shadow gating.

Never changes a solver or promotes a preconditioner. No speed/rescue claims.
Consumes the C1-H2 observation-only text format.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import re
import sys

CASES = (
    'uniform_n48', 'local_soft_n72', 'island_shell_n72',
    'multi_patch_n72', 'rotated_anisotropy_n72',
)
FRACTIONS = ('0.05', '0.2', '0.5')

class AuditError(ValueError):
    pass

@dataclass(frozen=True)
class Record:
    iteration: int
    residual: float
    active: int
    energy: float
    jaccard: float | None
    persist2: int
    persist3: int
    components: int
    max_component: int
    frontier: int

@dataclass(frozen=True)
class Audit:
    name: str
    fraction: float
    status: str
    dimension: int
    rows: list[Record]


def parse_log(name: str, source: str) -> Audit:
    fraction: float | None = None
    dimension: int | None = None
    status: str | None = None
    invariant = False
    header = False
    rows: list[Record] = []
    for line in source.splitlines():
        s = line.strip()
        if s.startswith('seed fraction') and ':' in s:
            try:
                fraction = float(s.split(':', 1)[1])
            except ValueError as e:
                raise AuditError(f'{name}: invalid seed fraction') from e
        elif s.startswith('dimensions') and ':' in s:
            try:
                dimension = int(s.split(':', 1)[1])
            except ValueError as e:
                raise AuditError(f'{name}: invalid dimension') from e
        elif s.startswith('observer status') and ':' in s:
            status = s.split(':', 1)[1].strip()
        elif s.startswith('PCG invariant') and ':' in s:
            invariant = s.split(':', 1)[1].strip() == 'PASS'
        elif s.startswith('iteration true_residual active '):
            header = True
        elif header and re.match(r'^\d+\s', s):
            fields = s.split()
            if len(fields) != 10:
                raise AuditError(f'{name}: malformed checkpoint: {s}')
            try:
                j = None if fields[4] == '-' else float(fields[4])
                row = Record(
                    int(fields[0]), float(fields[1]), int(fields[2]),
                    float(fields[3]), j, int(fields[5]), int(fields[6]),
                    int(fields[7]), int(fields[8]), int(fields[9]),
                )
            except ValueError as e:
                raise AuditError(f'{name}: invalid checkpoint: {s}') from e
            rows.append(row)
    if not header or not rows or fraction is None or dimension is None or status not in ('Converged', 'MaxIterations', 'Breakdown') or not invariant:
        raise AuditError(f'{name}: missing/invalid C1-H2 metadata or PCG invariant not PASS')
    if dimension <= 0 or not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise AuditError(f'{name}: invalid dimension/fraction')
    previous = 0
    for row in rows:
        if row.iteration <= previous or row.residual < 0 or not math.isfinite(row.residual) or not math.isfinite(row.energy) or not 0 <= row.energy <= 1.0 + 1e-10:
            raise AuditError(f'{name}: nonmonotonic iteration/non-finite residual or energy')
        if row.jaccard is not None and (not math.isfinite(row.jaccard) or not 0 <= row.jaccard <= 1.0):
            raise AuditError(f'{name}: invalid Jaccard')
        if any(v < 0 or v > dimension for v in (row.active, row.persist2, row.persist3, row.max_component, row.frontier)) or row.components > dimension or row.components < 0:
            raise AuditError(f'{name}: invalid mask counts')
        if row.persist2 > row.active or row.persist3 > row.persist2 or row.max_component > row.active or row.components > row.active:
            raise AuditError(f'{name}: inconsistent mask statistics')
        previous = row.iteration
    if rows[0].jaccard is not None or any(r.jaccard is None for r in rows[1:]):
        raise AuditError(f'{name}: missing/misplaced mask overlap')
    return Audit(name, fraction, status, dimension, rows)


def assess(audit: Audit, *, rate_limit: float = 0.995, min_energy: float = 0.5,
           min_jaccard: float = 0.2, min_persist3_fraction: float = 0.15,
           consecutive: int = 2) -> tuple[list[dict], dict]:
    if not 0 < rate_limit < 1 or not 0 <= min_energy <= 1 or not 0 <= min_jaccard <= 1 or not 0 <= min_persist3_fraction <= 1 or consecutive < 2:
        raise AuditError('invalid shadow policy thresholds')
    records: list[dict] = []
    bad_streak = 0
    tentative = 0
    candidate = 0
    recovered_spike = 0
    prev = None
    for r in audit.rows:
        step = r.iteration - prev.iteration if prev else 0
        if prev is None:
            weak_progress = False
            geometric_rate = None
        elif prev.residual == 0:
            weak_progress = False
            geometric_rate = None
        elif r.residual == 0:
            weak_progress = False
            geometric_rate = 0.0
        else:
            log_ratio_per_iteration = (math.log(r.residual) - math.log(prev.residual)) / step
            geometric_rate = math.exp(min(log_ratio_per_iteration, 700.0))
            weak_progress = log_ratio_per_iteration >= math.log(rate_limit)
        if not weak_progress and bad_streak:
            recovered_spike += 1
        bad_streak = bad_streak + 1 if weak_progress else 0
        persists = (r.active > 0 and r.persist3 / r.active >= min_persist3_fraction)
        spatial = (r.energy >= min_energy and r.jaccard is not None and r.jaccard >= min_jaccard and persists)
        # A shadow *candidate*, not a recommendation or a proof of rescue.
        action = ('shadow_candidate' if bad_streak >= consecutive and spatial
                  else 'isolated_bad_window' if weak_progress and bad_streak == 1
                  else 'unconfirmed_stall' if weak_progress
                  else 'progress_ok')
        if action == 'shadow_candidate':
            candidate += 1
        elif action == 'isolated_bad_window':
            tentative += 1
        records.append({
            'case': audit.name, 'seed_fraction': audit.fraction, 'iteration': r.iteration,
            'relative_residual': r.residual, 'delta_iter': step,
            'geometric_rate': geometric_rate,
            'weak_progress': weak_progress, 'bad_streak': bad_streak,
            'jaccard': r.jaccard, 'energy_fraction': r.energy,
            'persistent3_fraction': (r.persist3 / r.active if r.active else 0.0),
            'spatial_evidence': spatial, 'action': action,
        })
        prev = r
    summary = {
        'case': audit.name, 'seed_fraction': audit.fraction, 'status': audit.status,
        'checkpoints': len(records), 'isolated_bad_windows': tentative,
        'recovered_spike_windows': recovered_spike,
        'shadow_candidates': candidate,
        'final_action': records[-1]['action'],
        'invariant': 'PASS',
    }
    return records, summary


def run(paths: list[Path], output: Path, *, rate_limit: float) -> tuple[list[dict], list[dict]]:
    audits = [parse_log(p.name, p.read_text(encoding='utf-8')) for p in paths]
    events, summaries = [], []
    for audit in audits:
        e, s = assess(audit, rate_limit=rate_limit)
        events.extend(e)
        summaries.append(s)
    output.mkdir(parents=True, exist_ok=True)
    for fname, data in [('events.csv', events), ('summary.csv', summaries)]:
        with (output / fname).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output / 'summary.json').write_text(json.dumps({
        'policy': 'observation-only shadow; not real factorization or convergence rescue',
        'rate_limit': rate_limit, 'run_count': len(summaries),
        'candidate_windows': sum(s['shadow_candidates'] for s in summaries),
        'summaries': summaries,
    }, indent=2) + '\n', encoding='utf-8')
    return events, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rate-limit', type=float, default=0.995)
    parser.add_argument('--require-corpus', action='store_true', help='require exactly 5 cases x 3 thresholds')
    args = parser.parse_args()
    try:
        logs = sorted(args.logs.glob('*_fraction*.log'))
        if not logs:
            raise AuditError(f'no C1-H2 logs in {args.logs}')
        if args.require_corpus:
            expected = {f'{case}_fraction{fraction}.log' for case in CASES for fraction in FRACTIONS}
            actual = {p.name for p in logs}
            if actual != expected:
                raise AuditError(f'corpus filenames mismatch: missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}')
        events, sums = run(logs, args.out, rate_limit=args.rate_limit)
        print(f'C1-H3 shadow: {len(sums)} logs; {len(events)} checkpoints; '
              f'{sum(s["isolated_bad_windows"] for s in sums)} isolated bad windows; '
              f'{sum(s["shadow_candidates"] for s in sums)} hypothetical candidate windows')
        for s in sums:
            print(f'{s["case"]}: status={s["status"]} fraction={s["seed_fraction"]:.2f} '
                  f'isolated={s["isolated_bad_windows"]} recovered={s["recovered_spike_windows"]} '
                  f'candidates={s["shadow_candidates"]}')
        print(f'CSV/JSON: {args.out}')
        return 0
    except (AuditError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    sys.exit(main())
