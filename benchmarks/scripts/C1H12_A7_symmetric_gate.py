#!/usr/bin/env python3
"""Numerical and provenance gate for isolated A7 SPD composite experiment."""

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    from benchmarks.scripts.C1H12_A4_numerical_gate import CASES, ARMS, parse_time
    from benchmarks.scripts.C1H12_A5_certification_gate import parse_a5_record
    from benchmarks.scripts.C1H12_A4_numerical_gate import parse_record
except ModuleNotFoundError:
    from C1H12_A4_numerical_gate import CASES, ARMS, parse_time, parse_record
    from C1H12_A5_certification_gate import parse_a5_record


COARSE_ARMS = tuple(arm for arm in ARMS if arm != 'jacobi')
POLICIES = {'additive': 'Additive', 'symmetric': 'SymmetricComposite'}
FIELDS = {
    'case', 'policy', 'aggregation', 'basis', 'rows', 'nnz',
    'coarse_dimension', 'coarse_factor_bytes', 'status', 'iterations',
    'tol', 'reported_relative', 'verified_true_relative', 'relative_x_error',
    'load_ms', 'setup_ms', 'solve_ms', 'total_wall_ms', 'true_checks',
    'restarts', 'rejected_recursive_convergences', 'precond_calls',
    'extra_spmv_calls', 'scratch_bytes',
}
INTEGERS = (
    'rows', 'nnz', 'coarse_dimension', 'coarse_factor_bytes', 'iterations',
    'true_checks', 'restarts', 'rejected_recursive_convergences',
    'precond_calls', 'extra_spmv_calls', 'scratch_bytes',
)
FLOATS = (
    'tol', 'reported_relative', 'verified_true_relative', 'relative_x_error',
    'load_ms', 'setup_ms', 'solve_ms', 'total_wall_ms',
)
CSV_FIELDS = (
    'case', 'arm', 'policy', 'status', 'iterations', 'verified_true_relative',
    'true_checks', 'restarts', 'rejected_recursive_convergences',
    'precond_calls', 'extra_spmv_calls', 'scratch_bytes', 'coarse_factor_bytes',
    'setup_ms', 'solve_ms', 'total_wall_ms', 'elapsed_s', 'peak_rss_kib',
    'additive_status', 'additive_iterations', 'additive_true_relative',
    'additive_solve_ms',
)


def parse_a7(log: str, case: str, arm: str, policy: str) -> dict:
    lines = [line for line in log.splitlines() if line.startswith('C1H12A7|')]
    if len(lines) != 1:
        raise ValueError(f'expected one A7 row, found {len(lines)}')
    raw = {}
    for token in lines[0].split('|')[1:]:
        if '=' not in token:
            raise ValueError('malformed A7 token')
        name, value = token.split('=', 1)
        if not name or not value or name in raw:
            raise ValueError(f'duplicate or missing field: {name}')
        raw[name] = value
    if raw.keys() != FIELDS:
        raise ValueError(f'unexpected A7 schema: {raw.keys() ^ FIELDS}')
    data = dict(raw)
    for key in INTEGERS:
        data[key] = int(raw[key])
        if data[key] < 0:
            raise ValueError(f'negative {key}')
    for key in FLOATS:
        data[key] = float(raw[key])
        if not math.isfinite(data[key]) or data[key] < 0:
            raise ValueError(f'nonfinite/negative {key}')

    _, expected_agg, expected_basis = ARMS[arm]
    if data['case'] != case or data['policy'] != POLICIES[policy]:
        raise ValueError('wrong case/policy')
    if data['aggregation'] != expected_agg or data['basis'] != expected_basis:
        raise ValueError('coarse configuration mismatch')
    if data['rows'] != 20736 or data['nnz'] != CASES[case]:
        raise ValueError('unrecognized input matrix')
    if data['coarse_dimension'] <= 0 or data['coarse_factor_bytes'] <= 0:
        raise ValueError('empty coarse factor')
    if data['tol'] != 1e-10 or data['iterations'] > 600:
        raise ValueError('tolerance or global iteration budget mismatch')
    if data['true_checks'] == 0:
        raise ValueError('no true-residual certificate')
    if not data['restarts'] <= data['rejected_recursive_convergences'] <= data['true_checks']:
        raise ValueError('false-convergence restart accounting')
    if data['total_wall_ms'] + 1e-6 < data['load_ms'] + data['setup_ms'] + data['solve_ms']:
        raise ValueError('underreported total phase time')
    if policy == 'additive':
        if data['precond_calls'] != 0 or data['extra_spmv_calls'] != 0 or data['scratch_bytes'] != 0:
            raise ValueError('unexpected composite work in additive control')
    else:
        if data['precond_calls'] <= 0 or data['extra_spmv_calls'] != 2 * data['precond_calls']:
            raise ValueError('composite apply accounting mismatch')
        if data['scratch_bytes'] != 20736 * 5 * 8:
            raise ValueError('composite scratch size mismatch')

    if data['status'] == 'Converged':
        if data['verified_true_relative'] > 1.001e-10 or data['reported_relative'] > 1.001e-10:
            raise ValueError('false convergence under original CSR')
    elif data['status'] == 'MaxIterations':
        if data['iterations'] != 600 or data['verified_true_relative'] <= 1e-10:
            raise ValueError('false MaxIterations')
    elif data['status'] == 'Breakdown':
        if data['verified_true_relative'] <= 1e-10:
            raise ValueError('false Breakdown')
    else:
        raise ValueError('invalid solver status')
    return data


def audit(previous_a6: Path, run_dir: Path) -> int:
    errors: list[str] = []
    observed: dict[tuple[str, str, str], tuple[dict, dict]] = {}
    if not (previous_a6 / 'matrix_sha256.txt').is_file() or not (run_dir / 'matrix_sha256.txt').is_file():
        errors.append('missing original matrix hashes')
    elif (previous_a6 / 'matrix_sha256.txt').read_bytes() != (run_dir / 'matrix_sha256.txt').read_bytes():
        errors.append('A6/A7 input matrix SHA-256 mismatch')
    if not (previous_a6 / 'HEAD.txt').is_file() or not (run_dir / 'HEAD.txt').is_file():
        errors.append('missing source HEAD provenance')
    elif (previous_a6 / 'HEAD.txt').read_bytes() != (run_dir / 'HEAD.txt').read_bytes():
        errors.append('source HEAD differs from frozen A6 checkpoint')

    expected = {f'{case}_{arm}_{policy}_1'
                for case in CASES for arm in COARSE_ARMS for policy in POLICIES}
    for suffix in ('log', 'time'):
        actual = {p.stem for p in run_dir.glob(f'*.{suffix}')}
        for missing in sorted(expected - actual):
            errors.append(f'missing {missing}.{suffix}')
        for extra in sorted(actual - expected):
            errors.append(f'extra {extra}.{suffix}')

    for case in CASES:
        for arm in COARSE_ARMS:
            historical = previous_a6 / f'{case}_{arm}_a5_1.log'
            try:
                historical_text = historical.read_text(encoding='utf-8')
                historical_a4 = parse_record(historical_text)
                historical_a5 = parse_a5_record(historical_text)
                if historical_a4['status'] != historical_a5['status']:
                    raise ValueError('historical A5 status mismatch')
                if historical_a4['transfer_values'] != 'F64':
                    raise ValueError('historical transfer precision mismatch')
            except (ValueError, OSError, UnicodeError) as exc:
                errors.append(f'bad historical A5 {case}/{arm}: {exc}')
                continue
            for policy in POLICIES:
                stem = f'{case}_{arm}_{policy}_1'
                log = run_dir / f'{stem}.log'
                timing = run_dir / f'{stem}.time'
                if not log.is_file() or not timing.is_file():
                    continue
                try:
                    time_row = parse_time(timing.read_text(encoding='utf-8'))
                    if time_row['exit_status'] != 0:
                        raise ValueError(f'process exit {time_row["exit_status"]}')
                    values = parse_a7(log.read_text(encoding='utf-8'), case, arm, policy)
                    if policy == 'additive' and (
                        values['status'] != historical_a5['status']
                        or values['iterations'] != int(historical_a4['iterations'])
                        or values['coarse_dimension'] != int(historical_a4['coarse_dimension'])
                        or values['coarse_factor_bytes'] != int(historical_a4['coarse_factor_bytes'])
                        or not math.isclose(
                            values['verified_true_relative'],
                            float(historical_a4['verified_true_relative']),
                            abs_tol=1e-12, rel_tol=0.05,
                        )
                    ):
                        raise ValueError('same-policy numerical control changed versus frozen A6/A5')
                    observed[(case, arm, policy)] = (values, time_row)
                except (ValueError, OSError, UnicodeError) as exc:
                    errors.append(f'{stem}: {exc}')

    rows = []
    for case in CASES:
        for arm in COARSE_ARMS:
            control = observed.get((case, arm, 'additive'))
            for policy in POLICIES:
                pair = observed.get((case, arm, policy))
                if pair is None:
                    continue
                data, time = pair
                ctrl = control[0] if control else None
                rows.append({
                    'case': case, 'arm': arm, 'policy': policy,
                    **{key: data[key] for key in (
                        'status', 'iterations', 'verified_true_relative', 'true_checks',
                        'restarts', 'rejected_recursive_convergences', 'precond_calls',
                        'extra_spmv_calls', 'scratch_bytes', 'coarse_factor_bytes',
                        'setup_ms', 'solve_ms', 'total_wall_ms',
                    )},
                    'elapsed_s': time['elapsed_s'], 'peak_rss_kib': time['peak_rss_kib'],
                    'additive_status': ctrl['status'] if ctrl else 'MISSING',
                    'additive_iterations': ctrl['iterations'] if ctrl else -1,
                    'additive_true_relative': ctrl['verified_true_relative'] if ctrl else -1,
                    'additive_solve_ms': ctrl['solve_ms'] if ctrl else -1,
                })
    with (run_dir / 'symmetric_composite_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'C1-H12-A7 {len(observed)}/36 native numerical/log records complete')
    for policy in POLICIES:
        values = [v[0] for (case, arm, p), v in observed.items() if p == policy]
        nconverged = sum(v['status'] == 'Converged' for v in values)
        print(f'  {policy}: Converged={nconverged}/{len(values)}; '
              f'iterations_total={sum(v["iterations"] for v in values)}; '
              f'solve_ms_total={sum(v["solve_ms"] for v in values):.3f}')
    print(f'CSV: {run_dir / "symmetric_composite_comparison.csv"}')
    for error in errors:
        print(f'FAIL: {error}', file=sys.stderr)
    if errors or len(observed) != 36:
        return 1
    print('C1-H12-A7 36/36 original-CSR and provenance gates PASS')
    print('Scope: additive vs symmetric two-level composite only; no AMG multilevel or speedup claim.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_a6', type=Path)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    if not args.previous_a6.is_dir() or not args.run_dir.is_dir():
        parser.error('both A6 evidence and A7 run directories required')
    sys.exit(audit(args.previous_a6, args.run_dir))
