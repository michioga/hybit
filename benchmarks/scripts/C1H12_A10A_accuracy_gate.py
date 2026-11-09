#!/usr/bin/env python3
"""Fail-closed 54-case A10-A PCG tolerance and exact-error diagnosis."""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

try:
    from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS as ARM_INFO, CASES, parse_time
    from benchmarks.scripts.C1H12_A8_multilevel_gate import FIELDS as A8_FIELDS
    from benchmarks.scripts.C1H12_A9_rhs_gate import MODES as A9_MODES, parse_a9
except ModuleNotFoundError:
    from C1H12_A4_numerical_gate import ARMS as ARM_INFO, CASES, parse_time
    from C1H12_A8_multilevel_gate import FIELDS as A8_FIELDS
    from C1H12_A9_rhs_gate import MODES as A9_MODES, parse_a9


CASE = 'island_shell'
ARMS = ('graph_piecewise', 'strong_graph_piecewise', 'strong_graph_smoothed')
MODES = ('ones', 'index_smooth', 'seeded_random')
POLICIES = {'two_level': 'TwoLevel', 'multilevel': 'Multilevel'}
TOLERANCES = {'t10': 1e-10, 't12': 1e-12, 't13': 1e-13}
FIELDS = A8_FIELDS | {
    'rhs_mode', 'rhs_hash', 'rhs_norm', 'max_iterations',
    'compensated_true_relative', 'error_response_relative',
    'error_rayleigh', 'residual_amplification',
}
INTEGERS = (
    'rows', 'nnz', 'levels', 'coarse_dimension', 'coarse_factor_bytes',
    'symbolic_nnz', 'exact_zero_candidates', 'abtm_words', 'iterations',
    'max_iterations', 'true_checks', 'restarts',
    'rejected_recursive_convergences', 'precond_calls', 'extra_spmv_calls',
    'scratch_bytes',
)
FLOATS = (
    'rhs_norm', 'tol', 'reported_relative', 'verified_true_relative',
    'compensated_true_relative', 'relative_x_error', 'error_response_relative',
    'error_rayleigh', 'residual_amplification', 'load_ms', 'setup_ms',
    'solve_ms', 'total_wall_ms',
)
CSV_FIELDS = (
    'case', 'arm', 'rhs_mode', 'tol_tag', 'policy', 'rhs_hash',
    'status', 'iterations', 'tol', 'reported_relative',
    'verified_true_relative', 'compensated_true_relative',
    'relative_x_error', 'error_response_relative', 'error_rayleigh',
    'residual_amplification', 'levels', 'level_dims', 'level_nnz',
    'coarse_dimension', 'coarse_factor_bytes', 'true_checks', 'restarts',
    'precond_calls', 'extra_spmv_calls', 'setup_ms', 'solve_ms',
    'total_wall_ms', 'elapsed_s', 'peak_rss_kib',
    'two_level_status', 'two_level_iterations', 'two_level_relative_x_error',
)


def parse_a10(log: str, arm: str, mode: str, tol_tag: str, policy: str) -> dict:
    lines = [line for line in log.splitlines() if line.startswith('C1H12A10A|')]
    if len(lines) != 1:
        raise ValueError(f'expected one A10-A record, found {len(lines)}')
    raw = {}
    for token in lines[0].split('|')[1:]:
        if '=' not in token:
            raise ValueError('malformed A10-A field')
        key, value = token.split('=', 1)
        if not key or not value or key in raw:
            raise ValueError(f'duplicate or empty A10-A field: {key}')
        raw[key] = value
    if raw.keys() != FIELDS:
        raise ValueError(f'A10-A field schema mismatch: {raw.keys() ^ FIELDS}')

    row = dict(raw)
    for key in INTEGERS:
        row[key] = int(raw[key])
        if row[key] < 0:
            raise ValueError(f'negative {key}')
    for key in FLOATS:
        row[key] = float(raw[key])
        if not math.isfinite(row[key]):
            raise ValueError(f'nonfinite {key}')
        if key != 'error_rayleigh' and row[key] < 0:
            raise ValueError(f'negative {key}')

    if row['case'] != CASE or row['rhs_mode'] != A9_MODES[mode]:
        raise ValueError('FEM case or RHS mode mismatch')
    if row['policy'] != POLICIES[policy]:
        raise ValueError('PCG policy mismatch')
    _, aggregation, basis = ARM_INFO[arm]
    if row['aggregation'] != aggregation or row['basis'] != basis:
        raise ValueError('coarse arm mismatch')
    if row['rows'] != 20736 or row['nnz'] != CASES[CASE]:
        raise ValueError('wrong original FEM matrix')
    if row['tol'] != TOLERANCES[tol_tag] or row['max_iterations'] != 1200:
        raise ValueError('unexpected tolerance or iteration budget')
    if row['iterations'] > 1200 or row['true_checks'] == 0:
        raise ValueError('invalid iteration count or missing true residual check')
    if row['rhs_norm'] <= 0 or not re.fullmatch(r'[a-f0-9]{16}', raw['rhs_hash']):
        raise ValueError('RHS norm/hash invalid')
    if not row['restarts'] <= row['rejected_recursive_convergences'] <= row['true_checks']:
        raise ValueError('certification restart accounting invalid')
    if row['total_wall_ms'] + 1e-6 < row['load_ms'] + row['setup_ms'] + row['solve_ms']:
        raise ValueError('invalid total phase time')
    try:
        dims = [int(v) for v in raw['level_dims'].split(',')]
        nnzs = [int(v) for v in raw['level_nnz'].split(',')]
    except ValueError as exc:
        raise ValueError('invalid hierarchy vectors') from exc
    if len(dims) != row['levels'] or len(nnzs) != row['levels']:
        raise ValueError('hierarchy length mismatch')
    if dims[0] != 20736 or dims[-1] != row['coarse_dimension']:
        raise ValueError('hierarchy dimension endpoints invalid')
    if any(v <= 0 for v in dims) or nnzs[0] != CASES[CASE] or any(v < 0 for v in nnzs):
        raise ValueError('hierarchy contains invalid dimension or nnz')
    if row['coarse_factor_bytes'] <= 0:
        raise ValueError('missing coarse factor')
    if policy == 'two_level':
        if row['levels'] != 2 or nnzs[-1] != 0:
            raise ValueError('invalid two-level control hierarchy')
        if any(row[key] != 0 for key in (
            'symbolic_nnz', 'exact_zero_candidates', 'abtm_words',
            'precond_calls', 'extra_spmv_calls', 'scratch_bytes',
        )):
            raise ValueError('two-level control executed multilevel work')
    else:
        if not 3 <= row['levels'] <= 8 or any(child >= parent for parent, child in zip(dims, dims[1:])):
            raise ValueError('invalid multilevel hierarchy')
        if row['coarse_dimension'] > 512 or any(v <= 0 for v in nnzs):
            raise ValueError('bad AMG hierarchy budget or nnz')
        if row['coarse_factor_bytes'] != 8 * row['coarse_dimension'] ** 2:
            raise ValueError('wrong coarsest factor memory accounting')
        if row['symbolic_nnz'] <= 0 or row['abtm_words'] <= 0 or row['scratch_bytes'] <= 0:
            raise ValueError('no ABTM multilevel numerical work')
        if row['exact_zero_candidates'] > row['symbolic_nnz']:
            raise ValueError('invalid symbolic cancellation accounting')
        if row['precond_calls'] <= 0 or row['extra_spmv_calls'] != 2 * (row['levels'] - 1) * row['precond_calls']:
            raise ValueError('multilevel V-cycle work accounting mismatch')

    tol = row['tol']
    status = row['status']
    if status == 'Converged':
        if row['verified_true_relative'] > tol * 1.001 or row['reported_relative'] > tol * 1.001:
            raise ValueError('false original-CSR convergence')
    elif status == 'MaxIterations':
        if row['iterations'] != 1200 or row['verified_true_relative'] <= tol:
            raise ValueError('false MaxIterations result')
    elif status == 'Breakdown':
        if row['verified_true_relative'] <= tol:
            raise ValueError('false Breakdown result')
    else:
        raise ValueError('unknown PCG status')
    return row


def audit(previous_a9: Path, trial: Path) -> int:
    errors = []
    observed = {}
    fingerprints = {}
    for name in ('HEAD.txt', 'matrix_sha256.txt'):
        old, new = previous_a9 / name, trial / name
        if not old.is_file() or not new.is_file() or old.read_bytes() != new.read_bytes():
            errors.append(f'wrong or missing A9 provenance {name}')
    expected = {
        f'{CASE}_{arm}_{mode}_{tol_tag}_{policy}_1'
        for arm in ARMS for mode in MODES for tol_tag in TOLERANCES for policy in POLICIES
    }
    for ext in ('log', 'time'):
        present = {p.stem for p in trial.glob(f'*.{ext}')}
        errors.extend(f'missing {stem}.{ext}' for stem in sorted(expected - present))
        errors.extend(f'unexpected {stem}.{ext}' for stem in sorted(present - expected))

    for arm in ARMS:
        for mode in MODES:
            for tol_tag in TOLERANCES:
                for policy in POLICIES:
                    stem = f'{CASE}_{arm}_{mode}_{tol_tag}_{policy}_1'
                    log = trial / f'{stem}.log'
                    time = trial / f'{stem}.time'
                    if not log.is_file() or not time.is_file():
                        continue
                    try:
                        timing = parse_time(time.read_text(encoding='utf-8'))
                        if timing['exit_status'] != 0:
                            raise ValueError('process exit nonzero')
                        row = parse_a10(log.read_text(encoding='utf-8'), arm, mode, tol_tag, policy)
                        if mode in fingerprints and fingerprints[mode] != row['rhs_hash']:
                            raise ValueError('RHS changed across policies or tolerances')
                        fingerprints[mode] = row['rhs_hash']
                        if tol_tag == 't10':
                            historical = parse_a9(
                                (previous_a9 / f'{CASE}_{arm}_{mode}_{policy}_1.log').read_text(encoding='utf-8'),
                                CASE, arm, mode, policy,
                            )
                            for key in ('status', 'iterations', 'coarse_dimension',
                                        'coarse_factor_bytes', 'level_dims', 'level_nnz',
                                        'restarts', 'rhs_hash'):
                                if row[key] != historical[key]:
                                    raise ValueError(f'frozen A9 baseline changed {key}')
                            for key in ('verified_true_relative', 'relative_x_error'):
                                if not math.isclose(row[key], historical[key], rel_tol=0.05, abs_tol=1e-12):
                                    raise ValueError(f'frozen A9 baseline changed {key}')
                        observed[(arm, mode, tol_tag, policy)] = (row, timing)
                    except (ValueError, OSError, UnicodeError) as exc:
                        errors.append(f'{stem}: {exc}')

    if len(fingerprints) == 3 and len(set(fingerprints.values())) != 3:
        errors.append('distinct RHS modes produced the same fingerprint')
    data = []
    for arm in ARMS:
        for mode in MODES:
            for tol_tag in TOLERANCES:
                control = observed.get((arm, mode, tol_tag, 'two_level'))
                for policy in POLICIES:
                    item = observed.get((arm, mode, tol_tag, policy))
                    if item is None:
                        continue
                    row, timing = item
                    baseline = control[0] if control else None
                    data.append({
                        'case': CASE, 'arm': arm, 'rhs_mode': mode, 'tol_tag': tol_tag,
                        'policy': policy, 'rhs_hash': row['rhs_hash'],
                        **{key: row[key] for key in (
                            'status', 'iterations', 'tol', 'reported_relative',
                            'verified_true_relative', 'compensated_true_relative',
                            'relative_x_error', 'error_response_relative', 'error_rayleigh',
                            'residual_amplification', 'levels', 'level_dims', 'level_nnz',
                            'coarse_dimension', 'coarse_factor_bytes', 'true_checks',
                            'restarts', 'precond_calls', 'extra_spmv_calls',
                            'setup_ms', 'solve_ms', 'total_wall_ms',
                        )},
                        'elapsed_s': timing['elapsed_s'],
                        'peak_rss_kib': timing['peak_rss_kib'],
                        'two_level_status': baseline['status'] if baseline else 'MISSING',
                        'two_level_iterations': baseline['iterations'] if baseline else -1,
                        'two_level_relative_x_error': baseline['relative_x_error'] if baseline else -1,
                    })
    with (trial / 'accuracy_tolerance_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(data)

    print(f'C1-H12-A10A {len(observed)}/54 numerical and provenance records')
    for tol_tag in TOLERANCES:
        for policy in POLICIES:
            subset = [row for (a, m, t, p), (row, _) in observed.items() if t == tol_tag and p == policy]
            if subset:
                print(f'  {tol_tag}/{policy}: converged={sum(x["status"] == "Converged" for x in subset)}/{len(subset)}'
                      f'; iterations={sum(x["iterations"] for x in subset)}'
                      f'; max_x_error={max(x["relative_x_error"] for x in subset):.4e}'
                      f'; compensated_disagreements={sum(x["status"] == "Converged" and x["compensated_true_relative"] > x["tol"] * 1.001 for x in subset)}')
    print(f'CSV: {trial / "accuracy_tolerance_comparison.csv"}')
    for error in errors:
        print(f'FAIL: {error}', file=sys.stderr)
    if errors or len(observed) != 54:
        return 1
    print('C1-H12-A10A 54/54 input/CSR/status/hierarchy/frozen-A9 gates PASS')
    print('Independent compensated CSR residual is diagnostic and may reveal disagreements.')
    print('Scope: standalone tolerance/accuracy pilot; no production convergence guarantee.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_a9', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    if not args.previous_a9.is_dir() or not args.trial.is_dir():
        parser.error('historical A9 and fresh A10-A evidence directories are required')
    sys.exit(audit(args.previous_a9, args.trial))
