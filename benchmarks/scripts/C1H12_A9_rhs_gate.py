#!/usr/bin/env python3
"""Fail-closed A9 RHS-diversity audit, reusing the A8 CSR numerical gate."""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

try:
    from benchmarks.scripts.C1H12_A4_numerical_gate import CASES, parse_time
    from benchmarks.scripts.C1H12_A8_multilevel_gate import FIELDS as A8_FIELDS, parse_a8
except ModuleNotFoundError:
    from C1H12_A4_numerical_gate import CASES, parse_time
    from C1H12_A8_multilevel_gate import FIELDS as A8_FIELDS, parse_a8


ARMS = ('graph_piecewise', 'strong_graph_piecewise', 'strong_graph_smoothed')
MODES = {
    'ones': 'Ones',
    'index_smooth': 'IndexSmooth',
    'seeded_random': 'SeededRandom',
    'point_load': 'PointLoad',
}
POLICIES = ('two_level', 'multilevel')
FIELDS = A8_FIELDS | {'rhs_mode', 'rhs_hash', 'rhs_norm'}
CSV_FIELDS = (
    'case', 'arm', 'rhs_mode', 'policy', 'rhs_hash', 'rhs_norm', 'status',
    'iterations', 'reported_relative', 'verified_true_relative',
    'relative_x_error', 'levels', 'level_dims', 'level_nnz', 'coarse_dimension',
    'coarse_factor_bytes', 'symbolic_nnz', 'abtm_words', 'precond_calls',
    'extra_spmv_calls', 'true_checks', 'restarts', 'setup_ms', 'solve_ms',
    'total_wall_ms', 'elapsed_s', 'peak_rss_kib', 'two_level_status',
    'two_level_iterations', 'two_level_true_relative', 'two_level_solve_ms',
)


def parse_a9(log: str, case: str, arm: str, mode: str, policy: str) -> dict:
    records = [line for line in log.splitlines() if line.startswith('C1H12A9|')]
    if len(records) != 1:
        raise ValueError(f'expected exactly one A9 record, found {len(records)}')
    raw = {}
    for token in records[0].split('|')[1:]:
        if '=' not in token:
            raise ValueError('malformed A9 field')
        key, value = token.split('=', 1)
        if not key or not value or key in raw:
            raise ValueError(f'duplicate or empty A9 field: {key}')
        raw[key] = value
    if raw.keys() != FIELDS:
        raise ValueError(f'A9 schema mismatch: {raw.keys() ^ FIELDS}')
    if raw['rhs_mode'] != MODES[mode]:
        raise ValueError('RHS mode mismatch')
    if re.fullmatch(r'[0-9a-f]{16}', raw['rhs_hash']) is None:
        raise ValueError('bad RHS fingerprint')
    rhs_norm = float(raw['rhs_norm'])
    if not math.isfinite(rhs_norm) or rhs_norm <= 0:
        raise ValueError('nonfinite or zero RHS norm')
    xerror = raw['relative_x_error']
    if mode == 'point_load':
        if xerror != 'n/a':
            raise ValueError('point-load claims a manufactured exact solution')
    elif xerror == 'n/a':
        raise ValueError('manufactured RHS lacks exact-solution error')

    # The A8 gate checks the hierarchy, SPD-work counters, status, tolerances,
    # and independent original-CSR residual. PointLoad is the single case
    # without a manufactured truth, so only that comparison is unavailable.
    a8_row = {key: raw[key] for key in A8_FIELDS}
    if mode == 'point_load':
        a8_row['relative_x_error'] = '0'
    a8_line = 'C1H12A8|' + '|'.join(f'{key}={value}' for key, value in a8_row.items())
    result = parse_a8(a8_line, case, arm, policy)
    result['relative_x_error'] = 'n/a' if mode == 'point_load' else result['relative_x_error']
    result.update(rhs_mode=mode, rhs_hash=raw['rhs_hash'], rhs_norm=rhs_norm)
    return result


def audit(previous_a8: Path, trial: Path) -> int:
    errors = []
    observed = {}
    for filename in ('HEAD.txt', 'matrix_sha256.txt'):
        old = previous_a8 / filename
        new = trial / filename
        if not old.is_file() or not new.is_file() or old.read_bytes() != new.read_bytes():
            errors.append(f'missing or mismatched historical {filename}')

    expected = {
        f'{case}_{arm}_{mode}_{policy}_1'
        for case in CASES for arm in ARMS for mode in MODES for policy in POLICIES
    }
    for extension in ('log', 'time'):
        present = {path.stem for path in trial.glob(f'*.{extension}')}
        errors.extend(f'missing {stem}.{extension}' for stem in sorted(expected - present))
        errors.extend(f'unexpected {stem}.{extension}' for stem in sorted(present - expected))

    fingerprints = {}
    rhs_norms = {}
    for case in CASES:
        for arm in ARMS:
            for mode in MODES:
                for policy in POLICIES:
                    stem = f'{case}_{arm}_{mode}_{policy}_1'
                    log_path = trial / f'{stem}.log'
                    time_path = trial / f'{stem}.time'
                    if not log_path.is_file() or not time_path.is_file():
                        continue
                    try:
                        timing = parse_time(time_path.read_text(encoding='utf-8'))
                        if timing['exit_status'] != 0:
                            raise ValueError('nonzero process exit')
                        row = parse_a9(
                            log_path.read_text(encoding='utf-8'), case, arm, mode, policy
                        )
                        identity = (case, mode)
                        if identity in fingerprints and fingerprints[identity] != row['rhs_hash']:
                            raise ValueError('RHS fingerprint changed between arms/policies')
                        if identity in rhs_norms and not math.isclose(
                            rhs_norms[identity], row['rhs_norm'], rel_tol=1e-12
                        ):
                            raise ValueError('RHS norm changed between arms/policies')
                        fingerprints[identity] = row['rhs_hash']
                        rhs_norms[identity] = row['rhs_norm']

                        if mode == 'ones':
                            historical = parse_a8(
                                (previous_a8 / f'{case}_{arm}_{policy}_1.log').read_text(
                                    encoding='utf-8'
                                ), case, arm, policy,
                            )
                            for field in (
                                'status', 'iterations', 'coarse_dimension',
                                'coarse_factor_bytes', 'level_dims', 'level_nnz',
                                'symbolic_nnz', 'abtm_words', 'restarts',
                            ):
                                if row[field] != historical[field]:
                                    raise ValueError(f'A8 ones replay changed {field}')
                            if not math.isclose(
                                row['verified_true_relative'],
                                historical['verified_true_relative'],
                                abs_tol=1e-12, rel_tol=0.05,
                            ):
                                raise ValueError('A8 ones replay changed true residual')
                        observed[(case, arm, mode, policy)] = (row, timing)
                    except (OSError, ValueError, UnicodeError) as exc:
                        errors.append(f'{stem}: {exc}')

    for case in CASES:
        fingerprints_for_case = [fingerprints.get((case, mode)) for mode in MODES]
        if None not in fingerprints_for_case and len(set(fingerprints_for_case)) != len(MODES):
            errors.append(f'{case}: distinct RHS modes produced identical fingerprints')

    csv_rows = []
    for case in CASES:
        for arm in ARMS:
            for mode in MODES:
                control = observed.get((case, arm, mode, 'two_level'))
                for policy in POLICIES:
                    pair = observed.get((case, arm, mode, policy))
                    if pair is None:
                        continue
                    result, timing = pair
                    reference = control[0] if control else None
                    csv_rows.append({
                        'case': case, 'arm': arm, 'rhs_mode': mode, 'policy': policy,
                        **{key: result[key] for key in (
                            'rhs_hash', 'rhs_norm', 'status', 'iterations',
                            'reported_relative', 'verified_true_relative', 'relative_x_error',
                            'levels', 'level_dims', 'level_nnz', 'coarse_dimension',
                            'coarse_factor_bytes', 'symbolic_nnz', 'abtm_words',
                            'precond_calls', 'extra_spmv_calls', 'true_checks', 'restarts',
                            'setup_ms', 'solve_ms', 'total_wall_ms',
                        )},
                        'elapsed_s': timing['elapsed_s'],
                        'peak_rss_kib': timing['peak_rss_kib'],
                        'two_level_status': reference['status'] if reference else 'MISSING',
                        'two_level_iterations': reference['iterations'] if reference else -1,
                        'two_level_true_relative': (
                            reference['verified_true_relative'] if reference else -1
                        ),
                        'two_level_solve_ms': reference['solve_ms'] if reference else -1,
                    })

    with (trial / 'rhs_diversity_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'C1-H12-A9 {len(observed)}/72 numerical/log/provenance records complete')
    for mode in MODES:
        for policy in POLICIES:
            subset = [
                entry for (case, arm, m, p), (entry, _) in observed.items()
                if m == mode and p == policy
            ]
            print(f'  {mode}/{policy}: {sum(r["status"] == "Converged" for r in subset)}/'
                  f'{len(subset)} converged; iterations={sum(r["iterations"] for r in subset)}; '
                  f'solve_ms={sum(r["solve_ms"] for r in subset):.3f}')
    print(f'CSV: {trial / "rhs_diversity_comparison.csv"}')
    for error in errors:
        print(f'FAIL: {error}', file=sys.stderr)
    if errors or len(observed) != 72:
        return 1
    print('C1-H12-A9 72/72 CSR-certificate, RHS identity and frozen-A8 gates PASS')
    print('Scope: single-process numerical prototype, no production AMG or speedup claim.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_a8', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    if not args.previous_a8.is_dir() or not args.trial.is_dir():
        parser.error('historical A8 and new A9 trial directories required')
    sys.exit(audit(args.previous_a8, args.trial))
