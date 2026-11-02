#!/usr/bin/env python3
"""C1-H11: fail-closed, evidence-only external FEM SPD comparison.

This runner never generates matrices, RHS, references, SPD claims or success.
Use independently documented physical FEM input, never an MPI rank-local surrogate.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

STATUSES = {'Converged', 'MaxIterations', 'Breakdown'}
CERTIFIED = {'EXACT_SYMMETRIC_STRICT_DD_SPD',
             'EXACT_SYMMETRIC_IRREDUCIBLE_WEAK_DD_SPD'}
OTHER_SYMMETRIC = {'EXACT_SYMMETRIC_SPD_UNVERIFIED'}
FIELDS = ['case', 'budget', 'cap', 'repeat', 'plain_status', 'auto_status',
          'plain_iterations', 'auto_iterations', 'plain_true_residual',
          'auto_true_residual', 'plain_forward_error', 'auto_forward_error',
          'escalations', 'factor_mib', 'total_wall_ms', 'plain_setup_solve_ms',
          'peak_process_rss_kib', 'class', 'log', 'returncode']


def load_cases(manifest: Path) -> list[dict]:
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if set(data) != {'cases'} or not isinstance(data['cases'], list) or not data['cases']:
        raise ValueError('manifest must contain a nonempty cases array only')
    seen = set()
    out = []
    for item in data['cases']:
        required = {'id', 'matrix', 'rhs', 'origin', 'provenance', 'spd_evidence'}
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required - {'xref'}:
            raise ValueError('case requires id/matrix/rhs/origin/provenance/spd_evidence and optional xref')
        cid = item['id']
        if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', cid) or cid in seen:
            raise ValueError('case IDs must be unique, alphanumeric, dash or underscore')
        seen.add(cid)
        if item['origin'] not in ('frontistr_serial', 'external_fem_serial'):
            raise ValueError('only documented serial FEM snapshots are supported')
        case = {'id': cid, 'origin': item['origin']}
        for key in ('matrix', 'rhs', 'provenance', 'spd_evidence', 'xref'):
            if key == 'xref' and item.get(key) in (None, ''):
                case[key] = None
                continue
            raw = item.get(key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f'{cid}: invalid {key} path')
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = manifest.parent / path
            case[key] = path.resolve(strict=True)
            if not case[key].is_file():
                raise ValueError(f'{cid}: {key} is not a regular file')
        if len(case['spd_evidence'].read_text(encoding='utf-8').strip()) < 40:
            raise ValueError(f'{cid}: independent SPD evidence text is insufficient')
        if len(case['provenance'].read_text(encoding='utf-8').strip()) < 40:
            raise ValueError(f'{cid}: physical provenance text is insufficient')
        if case['origin'] == 'frontistr_serial':
            m = re.fullmatch(r'dump_matrix_(\d+)_0\.mm', case['matrix'].name)
            if not m or case['rhs'] != case['matrix'].with_suffix('.rhs'):
                raise ValueError(f'{cid}: FrontISTR needs a matched call/rank-0 .mm and .rhs pair')
            prefix = 'dump_matrix_' + m.group(1) + '_'
            if any(p.name.startswith(prefix) and re.fullmatch(r'dump_matrix_\d+_[1-9]\d*\.(mm|rhs)', p.name)
                   for p in case['matrix'].parent.iterdir()):
                raise ValueError(f'{cid}: MPI multi-rank snapshot is not a serial global system')
        out.append(case)
    return out


def field(part: str, key: str) -> str:
    hits = re.findall(r'^' + re.escape(key) + r'\s*:\s*(.+)$', part, flags=re.M)
    if len(hits) != 1:
        raise ValueError(f'{key}: expected one field, got {len(hits)}')
    return hits[0].strip()


def parsed_output(raw: str, tolerance: float, budget: int, with_xref: bool) -> dict:
    if raw.count('[1/2] Plain Jacobi-PCG') != 1 or raw.count('[2/2] HyBIT Auto') != 1:
        raise ValueError('missing or duplicated baseline/auto report markers')
    baseline, auto = raw.split('[2/2] HyBIT Auto', 1)
    baseline = baseline.split('[1/2] Plain Jacobi-PCG', 1)[1]
    out = {}
    for prefix, part in [('plain', baseline), ('auto', auto)]:
        status = field(part, 'status')
        if status not in STATUSES:
            raise ValueError(f'invalid {prefix} status')
        iterations = int(field(part, 'iterations'))
        resid = float(field(part, 'verified residual'))
        if not 0 <= iterations <= budget or not math.isfinite(resid) or resid < 0:
            raise ValueError('invalid iteration count or verified residual')
        out[prefix + '_status'] = status
        out[prefix + '_iterations'] = iterations
        out[prefix + '_true_residual'] = resid
        error = float(field(part, 'relative x error')) if with_xref else None
        if error is not None and (not math.isfinite(error) or error < 0):
            raise ValueError('invalid forward error')
        out[prefix + '_forward_error'] = error
    out['escalations'] = int(field(auto, 'escalations'))
    out['factor_mib'] = float(field(auto, 'factor memory').split()[0])
    out['total_wall_ms'] = float(field(auto, 'total wall').split()[0])
    plain_setup_solve = field(baseline, 'setup / solve')
    a, b = plain_setup_solve.split(' / ')
    out['plain_setup_solve_ms'] = float(a) + float(b.split()[0])
    for name in ('factor_mib', 'total_wall_ms', 'plain_setup_solve_ms'):
        if not math.isfinite(out[name]) or out[name] < 0:
            raise ValueError(f'invalid {name}')
    if out['escalations'] not in (0, 1):
        raise ValueError('opt-in routing should have at most one promotion')
    if out['escalations'] == 0 and (out['plain_status'] != out['auto_status']
                                   or out['plain_iterations'] != out['auto_iterations']):
        raise ValueError('KEEP_PCG baseline status/iteration mismatch')
    for prefix in ('plain', 'auto'):
        if out[prefix + '_status'] == 'Converged' and out[prefix + '_true_residual'] > tolerance * 1.00001:
            out['class'] = 'FALSE_REPORTED_CONVERGENCE'
            return out
    baseline_success = out['plain_status'] == 'Converged' and out['plain_true_residual'] <= tolerance
    hybrid_success = out['auto_status'] == 'Converged' and out['auto_true_residual'] <= tolerance
    if not baseline_success and hybrid_success and out['escalations']:
        out['class'] = 'RESCUE'
    elif baseline_success and not hybrid_success:
        out['class'] = 'RELIABILITY_REGRESSION'
    elif baseline_success and hybrid_success:
        out['class'] = 'BOTH_CONVERGED'
    elif not baseline_success and not hybrid_success:
        out['class'] = 'BOTH_NOT_CONVERGED'
    else:
        out['class'] = 'AUTO_CONVERGED_WITHOUT_LOCAL_FACTOR'
    return out


def audit(args, cases, out: Path) -> list[dict]:
    if __package__:
        from .fem_linear_system_probe import audit_matrix, audit_rhs, sha256
    else:
        from fem_linear_system_probe import audit_matrix, audit_rhs, sha256
    binary = args.repo / 'target/release/examples/mtx_fem_audit'
    if not binary.is_file():
        raise ValueError(f'missing matrix audit executable {binary}; run C1H11_run_real_fem.sh build')
    paths = [str(c['matrix']) for c in cases]
    audited = out / 'mtx_audit.csv'
    subprocess.run([str(binary), str(audited), str(args.max_expanded_nnz), *paths],
                   check=True, timeout=args.timeout)
    with audited.open(newline='', encoding='utf-8') as f:
        results = list(csv.DictReader(f))
    if len(results) != len(cases):
        raise ValueError('audit result row count mismatch')
    receipt = []
    for case, line in zip(cases, results):
        if line['file'] != str(case['matrix']):
            raise ValueError('matrix audit order mismatch')
        mat = audit_matrix(case['matrix'])
        rhs = audit_rhs(case['rhs'], mat['rows'])
        if rhs['rhs_l2_norm'] == 0:
            raise ValueError(f'{case["id"]}: zero RHS is excluded from rescue statistics')
        if case['xref']:
            audit_rhs(case['xref'], mat['rows'])
        status = line['status']
        if status not in CERTIFIED | OTHER_SYMMETRIC:
            raise ValueError(f'{case["id"]}: PCG refused; matrix audit={status}')
        receipt.append({
            'id': case['id'], 'origin': case['origin'], 'matrix_audit_class': status,
            'spd_evidence_kind': 'sufficient_numerical_dd_certificate' if status in CERTIFIED
                                 else 'external_evidence_unverified_by_machine',
            'rows': mat['rows'], 'nnz': int(line['csr_nnz']),
            'matrix_sha256': sha256(case['matrix']), 'rhs_sha256': sha256(case['rhs']),
            'xref_sha256': sha256(case['xref']) if case['xref'] else None,
            'provenance_sha256': sha256(case['provenance']),
            'spd_evidence_sha256': sha256(case['spd_evidence']),
        })
    (out / 'intake.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'run'), default='audit')
    parser.add_argument('--budgets', type=int, nargs='+', default=[120, 360, 1200])
    parser.add_argument('--caps', type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--probe', type=int, default=12)
    parser.add_argument('--factor-mib', type=int, default=64)
    parser.add_argument('--max-expanded-nnz', type=int, default=10000000)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if (not math.isfinite(args.tol) or args.tol <= 0 or args.probe <= 0
        or not args.budgets or min(args.budgets) <= args.probe
        or not args.caps or min(args.caps) < 2
        or args.factor_mib <= 0 or not 1 <= args.max_expanded_nnz <= 100000000
        or args.timeout <= 0):
        parser.error('invalid tolerance, probe, budget, factor cap or timeout')
    args.repo = args.repo.resolve()
    cases = load_cases(args.manifest.expanduser().resolve())
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error('output directory must not exist (avoid overwriting/stale results)')
    out.mkdir(parents=True)
    (out / 'invocation.json').write_text(json.dumps({
        'mode': args.mode, 'tol': args.tol, 'budgets': args.budgets, 'caps': args.caps,
        'probe': args.probe, 'factor_mib': args.factor_mib, 'cases': [c['id'] for c in cases],
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=args.repo, text=True),
    }, indent=2) + '\n', encoding='utf-8')
    receipt = audit(args, cases, out)
    print('C1-H11 intake audit PASS:', len(receipt), 'case(s); no SPD assertion inferred from positive diagonals')
    if args.mode == 'audit':
        print('C1-H11 AUDIT ONLY PASS; NO FEM SOLVES PERFORMED')
        return 0
    fem = args.repo / 'target/release/examples/fem_bench'
    if not fem.is_file():
        raise ValueError('missing built fem_bench; run C1H11_run_real_fem.sh build')
    if not Path('/usr/bin/time').is_file():
        raise ValueError('GNU /usr/bin/time required for process peak RSS collection')
    logs = out / 'logs'; logs.mkdir()
    results = []
    for case in cases:
        for budget in args.budgets:
            for cap in args.caps:
                name = f'{case["id"]}_budget{budget}_cap{cap}'
                log = logs / (name + '.log')
                rssfile = logs / (name + '.rss')
                cmd = ['/usr/bin/time', '-f', '%M', '-o', str(rssfile), str(fem),
                       '--matrix', str(case['matrix']), '--rhs', str(case['rhs']),
                       '--tol', repr(args.tol), '--max-iters', str(budget),
                       '--probe-iters', str(args.probe), '--max-region', str(cap),
                       '--max-regions', '1', '--max-escalations', '1', '--overlap', '0',
                       '--factor-budget-mib', str(args.factor_mib), '--backend', 'csr',
                       '--abtm-weighted-routing']
                if case['xref']:
                    cmd += ['--xref', str(case['xref'])]
                row = {'case': case['id'], 'budget': budget, 'cap': cap, 'repeat': 1,
                       'log': str(log), 'returncode': None}
                try:
                    cp = subprocess.run(cmd, capture_output=True, text=True,
                                        check=False, timeout=args.timeout)
                    log.write_text('COMMAND: ' + ' '.join(cmd) + '\n' + cp.stdout
                                   + '\n=== STDERR ===\n' + cp.stderr, encoding='utf-8')
                    row['returncode'] = cp.returncode
                    if cp.returncode:
                        row['class'] = 'PROCESS_ERROR'
                    else:
                        row.update(parsed_output(cp.stdout, args.tol, budget, case['xref'] is not None))
                    row['peak_process_rss_kib'] = int(rssfile.read_text().strip()) if rssfile.is_file() else None
                except (subprocess.TimeoutExpired, ValueError, OSError) as e:
                    row['class'] = 'TIMEOUT_OR_PARSE_ERROR'
                    log.write_text('ERROR: ' + repr(e) + '\n', encoding='utf-8')
                results.append(row)
                print(f'{name}: {row["class"]}', flush=True)
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(results)
    classes = {label: sum(r['class'] == label for r in results)
               for label in sorted({r['class'] for r in results})}
    (out / 'summary.json').write_text(json.dumps({
        'total_attempted': len(results), 'classes': classes,
        'notes': 'no claim of SPD proof or physical provenance without external documentation; no PETSc/hypre result'
    }, indent=2) + '\n', encoding='utf-8')
    if any(r['class'] in ('TIMEOUT_OR_PARSE_ERROR', 'PROCESS_ERROR', 'FALSE_REPORTED_CONVERGENCE',
                          'RELIABILITY_REGRESSION') for r in results):
        print('C1-H11 outcomes include error or regression: inspect all rows; no cherry-picking')
        return 1
    print('C1-H11 EVIDENCE RUN COMPLETE: classification is empirical, NOT a production policy gate')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f'C1-H11 FAIL: {exc}', file=sys.stderr)
        sys.exit(1)
