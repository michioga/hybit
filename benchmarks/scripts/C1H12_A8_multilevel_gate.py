#!/usr/bin/env python3
"""Fail-closed original-CSR numerical/provenance gate for native A8 AMG."""

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, CASES, parse_time
    from benchmarks.scripts.C1H12_A7_symmetric_gate import parse_a7
except ModuleNotFoundError:
    from C1H12_A4_numerical_gate import ARMS, CASES, parse_time
    from C1H12_A7_symmetric_gate import parse_a7


COARSE_ARMS = tuple(arm for arm in ARMS if arm != 'jacobi')
POLICIES = {'two_level': 'TwoLevel', 'multilevel': 'Multilevel'}
FIELDS = {
    'case', 'policy', 'aggregation', 'basis', 'rows', 'nnz', 'levels',
    'level_dims', 'level_nnz', 'coarse_dimension', 'coarse_factor_bytes',
    'symbolic_nnz', 'exact_zero_candidates', 'abtm_words', 'status',
    'iterations', 'tol', 'reported_relative', 'verified_true_relative',
    'relative_x_error', 'load_ms', 'setup_ms', 'solve_ms', 'total_wall_ms',
    'true_checks', 'restarts', 'rejected_recursive_convergences',
    'precond_calls', 'extra_spmv_calls', 'scratch_bytes',
}
INTEGERS = (
    'rows', 'nnz', 'levels', 'coarse_dimension', 'coarse_factor_bytes',
    'symbolic_nnz', 'exact_zero_candidates', 'abtm_words', 'iterations',
    'true_checks', 'restarts', 'rejected_recursive_convergences',
    'precond_calls', 'extra_spmv_calls', 'scratch_bytes',
)
FLOATS = (
    'tol', 'reported_relative', 'verified_true_relative', 'relative_x_error',
    'load_ms', 'setup_ms', 'solve_ms', 'total_wall_ms',
)
CSV_FIELDS = (
    'case', 'arm', 'policy', 'status', 'iterations', 'verified_true_relative',
    'levels', 'level_dims', 'level_nnz', 'coarse_dimension', 'coarse_factor_bytes',
    'symbolic_nnz', 'exact_zero_candidates', 'abtm_words', 'true_checks',
    'restarts', 'rejected_recursive_convergences', 'precond_calls',
    'extra_spmv_calls', 'scratch_bytes', 'setup_ms', 'solve_ms',
    'total_wall_ms', 'elapsed_s', 'peak_rss_kib', 'two_level_status',
    'two_level_iterations', 'two_level_true_relative', 'two_level_solve_ms',
)


def parse_a8(log: str, case: str, arm: str, policy: str) -> dict:
    lines = [line for line in log.splitlines() if line.startswith('C1H12A8|')]
    if len(lines) != 1:
        raise ValueError(f'expected exactly one A8 row, found {len(lines)}')
    raw = {}
    for token in lines[0].split('|')[1:]:
        if '=' not in token:
            raise ValueError('malformed A8 field')
        key, value = token.split('=', 1)
        if not key or not value or key in raw:
            raise ValueError(f'duplicate or empty A8 field: {key}')
        raw[key] = value
    if raw.keys() != FIELDS:
        raise ValueError(f'wrong A8 record schema: {raw.keys() ^ FIELDS}')
    result = dict(raw)
    for key in INTEGERS:
        result[key] = int(raw[key])
        if result[key] < 0:
            raise ValueError(f'negative integer field {key}')
    for key in FLOATS:
        result[key] = float(raw[key])
        if result[key] < 0 or not math.isfinite(result[key]):
            raise ValueError(f'nonfinite or negative field {key}')
    try:
        dims = [int(v) for v in raw['level_dims'].split(',')]
        nnzs = [int(v) for v in raw['level_nnz'].split(',')]
    except ValueError as exc:
        raise ValueError('invalid hierarchy dimension or nnz vector') from exc
    if result['levels'] != len(dims) or len(nnzs) != len(dims):
        raise ValueError('hierarchy vector length mismatch')
    if any(v <= 0 for v in dims) or nnzs[0] <= 0 or any(v < 0 for v in nnzs):
        raise ValueError('hierarchy nonpositive dimension/nnz')
    if dims[0] != result['rows'] or dims[-1] != result['coarse_dimension']:
        raise ValueError('hierarchy endpoint mismatch')
    if result['case'] != case or result['policy'] != POLICIES[policy]:
        raise ValueError('case or policy mismatch')
    _, agg, basis = ARMS[arm]
    if result['aggregation'] != agg or result['basis'] != basis:
        raise ValueError('coarse policy mismatch')
    if result['rows'] != 20736 or result['nnz'] != CASES[case] or nnzs[0] != CASES[case]:
        raise ValueError('input matrix identity mismatch')
    if result['tol'] != 1e-10 or result['iterations'] > 600:
        raise ValueError('global PCG tolerance or budget mismatch')
    if result['true_checks'] == 0:
        raise ValueError('missing true-residual verification')
    if not result['restarts'] <= result['rejected_recursive_convergences'] <= result['true_checks']:
        raise ValueError('certification restart accounting invalid')
    if result['total_wall_ms'] + 1e-6 < result['load_ms'] + result['setup_ms'] + result['solve_ms']:
        raise ValueError('total wall time smaller than phase times')
    if result['coarse_factor_bytes'] <= 0:
        raise ValueError('missing direct coarse factor')

    if policy == 'two_level':
        if result['levels'] != 2 or nnzs[1] != 0:
            raise ValueError('two-level control hierarchy mismatch')
        if any(result[key] != 0 for key in (
            'symbolic_nnz', 'exact_zero_candidates', 'abtm_words',
            'precond_calls', 'extra_spmv_calls', 'scratch_bytes',
        )):
            raise ValueError('control unexpectedly executed multilevel work')
    else:
        if result['levels'] < 3 or result['levels'] > 8:
            raise ValueError('invalid AMG level count')
        if any(child >= parent for parent, child in zip(dims, dims[1:])):
            raise ValueError('nondecreasing AMG hierarchy')
        if result['coarse_dimension'] > 512:
            raise ValueError('AMG coarsest factor exceeds budget')
        if any(v == 0 for v in nnzs):
            raise ValueError('empty AMG coarse matrix')
        if result['coarse_factor_bytes'] != dims[-1] * dims[-1] * 8:
            raise ValueError('coarse Cholesky memory accounting invalid')
        if result['symbolic_nnz'] <= 0 or result['abtm_words'] <= 0:
            raise ValueError('no ABTM numerical hierarchy evidence')
        if result['exact_zero_candidates'] > result['symbolic_nnz']:
            raise ValueError('invalid cancellation accounting')
        if result['precond_calls'] <= 0 or result['scratch_bytes'] <= 0:
            raise ValueError('missing V-cycle work counters')
        if result['extra_spmv_calls'] != 2 * (result['levels'] - 1) * result['precond_calls']:
            raise ValueError('V-cycle SpMV work accounting mismatch')

    if result['status'] == 'Converged':
        if result['verified_true_relative'] > 1.001e-10 or result['reported_relative'] > 1.001e-10:
            raise ValueError('false convergence against original CSR')
    elif result['status'] == 'MaxIterations':
        if result['iterations'] != 600 or result['verified_true_relative'] <= 1e-10:
            raise ValueError('false MaxIterations status')
    elif result['status'] == 'Breakdown':
        if result['verified_true_relative'] <= 1e-10:
            raise ValueError('false Breakdown status')
    else:
        raise ValueError('invalid PCG status')
    result['level_dims'] = raw['level_dims']
    result['level_nnz'] = raw['level_nnz']
    return result


def audit(previous_a7: Path, trial: Path) -> int:
    errors = []
    observed = {}
    for filename in ('HEAD.txt', 'matrix_sha256.txt'):
        prior = previous_a7 / filename
        new = trial / filename
        if not prior.is_file() or not new.is_file() or prior.read_bytes() != new.read_bytes():
            errors.append(f'missing/mismatched historical {filename}')
    expected = {f'{case}_{arm}_{policy}_1'
                for case in CASES for arm in COARSE_ARMS for policy in POLICIES}
    for extension in ('log', 'time'):
        present = {path.stem for path in trial.glob(f'*.{extension}')}
        for missing in sorted(expected - present):
            errors.append(f'missing {missing}.{extension}')
        for extra in sorted(present - expected):
            errors.append(f'extra {extra}.{extension}')

    for case in CASES:
        for arm in COARSE_ARMS:
            try:
                historical = parse_a7(
                    (previous_a7 / f'{case}_{arm}_additive_1.log').read_text(encoding='utf-8'),
                    case, arm, 'additive',
                )
            except (ValueError, OSError, UnicodeError) as exc:
                errors.append(f'bad historical A7 additive control {case}/{arm}: {exc}')
                continue
            for policy in POLICIES:
                stem = f'{case}_{arm}_{policy}_1'
                log_path = trial / f'{stem}.log'
                time_path = trial / f'{stem}.time'
                if not log_path.is_file() or not time_path.is_file():
                    continue
                try:
                    time_row = parse_time(time_path.read_text(encoding='utf-8'))
                    if time_row['exit_status'] != 0:
                        raise ValueError('nonzero process exit')
                    row = parse_a8(log_path.read_text(encoding='utf-8'), case, arm, policy)
                    if policy == 'two_level' and (
                        row['status'] != historical['status']
                        or row['iterations'] != historical['iterations']
                        or row['coarse_dimension'] != historical['coarse_dimension']
                        or row['coarse_factor_bytes'] != historical['coarse_factor_bytes']
                        or not math.isclose(
                            row['verified_true_relative'], historical['verified_true_relative'],
                            abs_tol=1e-12, rel_tol=0.05,
                        )
                    ):
                        raise ValueError('A8 control changed versus frozen A7 additive')
                    observed[(case, arm, policy)] = (row, time_row)
                except (ValueError, OSError, UnicodeError) as exc:
                    errors.append(f'{stem}: {exc}')

    rows = []
    for case in CASES:
        for arm in COARSE_ARMS:
            control = observed.get((case, arm, 'two_level'))
            for policy in POLICIES:
                pair = observed.get((case, arm, policy))
                if pair is None:
                    continue
                record, timing = pair
                reference = control[0] if control else None
                rows.append({
                    'case': case, 'arm': arm, 'policy': policy,
                    **{key: record[key] for key in (
                        'status', 'iterations', 'verified_true_relative', 'levels',
                        'level_dims', 'level_nnz', 'coarse_dimension', 'coarse_factor_bytes',
                        'symbolic_nnz', 'exact_zero_candidates', 'abtm_words', 'true_checks',
                        'restarts', 'rejected_recursive_convergences', 'precond_calls',
                        'extra_spmv_calls', 'scratch_bytes', 'setup_ms', 'solve_ms',
                        'total_wall_ms',
                    )},
                    'elapsed_s': timing['elapsed_s'],
                    'peak_rss_kib': timing['peak_rss_kib'],
                    'two_level_status': reference['status'] if reference else 'MISSING',
                    'two_level_iterations': reference['iterations'] if reference else -1,
                    'two_level_true_relative': reference['verified_true_relative'] if reference else -1,
                    'two_level_solve_ms': reference['solve_ms'] if reference else -1,
                })
    with (trial / 'native_multilevel_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'C1-H12-A8 {len(observed)}/36 numerical/log/provenance records complete')
    for policy in POLICIES:
        values = [data for (case, arm, p), (data, _) in observed.items() if p == policy]
        print(f'  {policy}: Converged={sum(x["status"] == "Converged" for x in values)}/{len(values)}, '
              f'MaxIterations={sum(x["status"] == "MaxIterations" for x in values)}, '
              f'Breakdown={sum(x["status"] == "Breakdown" for x in values)}, '
              f'iterations_total={sum(x["iterations"] for x in values)}, '
              f'solve_ms_total={sum(x["solve_ms"] for x in values):.3f}')
    print(f'CSV: {trial / "native_multilevel_comparison.csv"}')
    for error in errors:
        print(f'FAIL: {error}', file=sys.stderr)
    if errors or len(observed) != 36:
        return 1
    print('C1-H12-A8 36/36 original-CSR numerical and provenance gates PASS')
    print('Scope: native prototype V-cycle, not validated as a production multilevel AMG or a speedup.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_a7', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    if not args.previous_a7.is_dir() or not args.trial.is_dir():
        parser.error('both prior A7 evidence and A8 trial directories required')
    sys.exit(audit(args.previous_a7, args.trial))
