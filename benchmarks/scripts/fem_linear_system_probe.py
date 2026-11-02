#!/usr/bin/env python3
"""C1-H1 FEM-agnostic Matrix Market / RHS audit and optional HyBIT runner.

FrontISTR serial dumps are one supported adapter, not a dependency of HyBIT.
NEVER merge distributed rank dumps or infer SPD from file format/diagonal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

DUMP = re.compile(r"dump_matrix_(\d+)_(\d+)\.(mm|rhs)$")
FIELDS = ("status", "iterations", "verified residual", "reported residual", "relative x error", "preconditioner", "total wall")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_real(token: str) -> bool:
    try:
        return math.isfinite(float(token.replace('D', 'E').replace('d', 'e')))
    except ValueError:
        return False


def audit_matrix(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        header = f.readline().strip().lower()
        valid_headers = {
            '%%matrixmarket matrix coordinate real general',
            '%%matrixmarket matrix coordinate real symmetric',
        }
        if header not in valid_headers:
            raise ValueError(f'{path}: unsupported header {header!r} (require real coordinate general/symmetric)')
        dimensions = None
        count = 0
        first_bad = None
        for row in f:
            line = row.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            if dimensions is None:
                if len(parts) != 3:
                    raise ValueError(f'{path}: expected Matrix Market size line')
                nrow, ncol, ndata = map(int, parts)
                if nrow <= 0 or ncol <= 0 or ndata < 0:
                    raise ValueError(f'{path}: invalid dimensions')
                dimensions = (nrow, ncol, ndata)
                continue
            count += 1
            if len(parts) != 3 or not valid_real(parts[2]):
                first_bad = (count, 'bad value or entry')
                break
            try:
                rowidx, colidx = int(parts[0]), int(parts[1])
            except ValueError:
                first_bad = (count, 'bad indices')
                break
            if not (1 <= rowidx <= dimensions[0] and 1 <= colidx <= dimensions[1]):
                first_bad = (count, 'out-of-bounds index')
                break
        if dimensions is None:
            raise ValueError(f'{path}: size line missing')
        if first_bad:
            raise ValueError(f'{path}: entry {first_bad[0]}: {first_bad[1]}')
        if count != dimensions[2]:
            raise ValueError(f'{path}: entries {count}, header specified {dimensions[2]}')
        if dimensions[0] != dimensions[1]:
            raise ValueError(f'{path}: non-square dump ({dimensions[0]} x {dimensions[1]}), cannot use serial HyBIT direct reader')
        return {'rows': dimensions[0], 'cols': dimensions[1], 'stored_entries': count,
                'symmetry_declared': header.rsplit(' ', 1)[-1], 'sha256': sha256(path)}


def audit_rhs(path: Path, n: int) -> dict:
    count, squared = 0, 0.0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            for token in line.split():
                if not valid_real(token):
                    raise ValueError(f'{path}: non-finite/invalid RHS token at position {count+1}')
                count += 1
                if count > n:
                    raise ValueError(f'{path}: RHS length exceeds {n}')
                squared = math.hypot(squared, float(token.replace('D','E').replace('d','e')))
    if count != n:
        raise ValueError(f'{path}: RHS length {count}, expected {n}')
    return {'length': count, 'rhs_l2_norm': squared, 'sha256': sha256(path)}


def discover(dump_dir: Path, call: int | None) -> tuple[Path, Path, int]:
    files = {}
    for p in dump_dir.iterdir():
        match = DUMP.fullmatch(p.name)
        if match and p.is_file():
            number, rank, ext = int(match[1]), int(match[2]), match[3]
            files[(number, rank, ext)] = p
    if not files:
        raise ValueError(f'{dump_dir}: no dump_matrix_CALL_RANK.(mm|rhs) files')
    ranks = {key[1] for key in files}
    if ranks != {0}:
        raise ValueError(f'distributed rank dump(s) detected {sorted(ranks)}; no automatic MPI rank merge')
    choices = sorted({key[0] for key in files})
    if call is None:
        if len(choices) != 1:
            raise ValueError(f'multiple solver calls {choices}; select one with --call N')
        call = choices[0]
    if (call, 0, 'mm') not in files or (call, 0, 'rhs') not in files:
        raise ValueError(f'call {call} requires both dump_matrix_{call}_0.mm and .rhs')
    return files[(call, 0, 'mm')], files[(call, 0, 'rhs')], call


def collect(raw: str, tol: float, code: int) -> dict:
    found = {}
    for line in raw.splitlines():
        if ':' not in line:
            continue
        key, val = line.split(':', 1)
        if key.strip() in FIELDS:
            found[key.strip()] = val.strip()
    status = found.get('status', 'MISSING')
    try:
        rr = float(found.get('verified residual', 'nan'))
    except ValueError:
        rr = float('nan')
    passed = code == 0 and status == 'Converged' and math.isfinite(rr) and rr <= tol * (1+1e-10)
    return {'outcome': 'CONVERGED' if passed else 'NOT_CONVERGED_OR_ERROR', 'exit_code': code,
            'fields': found, 'numerical_gate': passed}


def run_case(argv: list[str], log: Path, tol: float) -> dict:
    cp = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.write_text(cp.stdout, encoding='utf-8')
    result = collect(cp.stdout, tol, cp.returncode)
    result['command'] = argv
    result['raw_log'] = str(log)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--matrix', type=Path, help='Matrix Market input from any FEM host')
    src.add_argument('--frontistr-dump-dir', type=Path, help='FrontISTR single-rank dump directory')
    parser.add_argument('--rhs', type=Path, help='paired RHS; required with --matrix')
    parser.add_argument('--xref', type=Path, help='optional reference solution, only for forward-error diagnostics')
    parser.add_argument('--frontistr-serial-confirmed', action='store_true',
                        help='operator confirms the FrontISTR dump came from a ONE-rank run')
    parser.add_argument('--call', type=int, help='FrontISTR dump call number')
    parser.add_argument('--run', action='store_true', help='run FGMRES/Jacobi and ILU(0) after audit')
    parser.add_argument('--declared-spd', action='store_true', help='add SPD PCG/Hybrid only when independently justified')
    parser.add_argument('--hybit-dir', type=Path, default=Path.cwd())
    parser.add_argument('--out-dir', type=Path, default=Path('fem_probe_results'))
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument('--restart', type=int, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.tol) or args.tol <= 0 or args.max_iters <= 0 or args.restart <= 0:
        parser.error('tol, max-iters and restart must be positive and finite')
    try:
        if args.frontistr_dump_dir:
            if args.rhs is not None:
                raise ValueError('--rhs is only allowed with --matrix (FrontISTR RHS is auto-paired)')
            matrix, rhs, call = discover(args.frontistr_dump_dir, args.call)
            if args.run and not args.frontistr_serial_confirmed:
                raise ValueError('--frontistr-serial-confirmed is required to solve a FrontISTR dump')
            mode = 'frontistr-serial-dump'
            source_contract = 'same solver call/rank; serial provenance must be confirmed separately'
        else:
            if args.call is not None or args.frontistr_serial_confirmed:
                raise ValueError('--call and --frontistr-serial-confirmed require --frontistr-dump-dir')
            if args.rhs is None:
                raise ValueError('--matrix requires --rhs')
            matrix, rhs, call = args.matrix, args.rhs, None
            mode = 'generic-fem-matrix-market'
            source_contract = 'matrix and RHS pairing/ownership must be established by host'
        mm = audit_matrix(matrix)
        b = audit_rhs(rhs, mm['rows'])
        result = {'host_adapter': mode, 'frontistr_dump_call': call,
                  'matrix': str(matrix.resolve()), 'rhs': str(rhs.resolve()),
                  'matrix_audit': mm, 'rhs_audit': b, 'spd_status': 'NOT_CERTIFIED',
                  'source_contract': source_contract, 'results': {}}
        if args.run:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            fgmres = args.hybit_dir / 'target/release/examples/fem_fgmres_bench'
            if not fgmres.is_file():
                raise ValueError(f'build {fgmres} first (cargo build --release --locked -p hybit --example fem_fgmres_bench)')
            common = ['--matrix', str(matrix.resolve()), '--rhs', str(rhs.resolve()),
                      '--tol', str(args.tol), '--max-iters', str(args.max_iters)]
            if args.xref:
                common += ['--xref', str(args.xref.resolve())]
            for name, prec in [('fgmres_jacobi','jacobi'),('fgmres_ilu0','ilu0')]:
                argv = [str(fgmres), *common, '--precond', prec,
                        '--restart', str(args.restart), '--restart-policy', 'fixed']
                result['results'][name] = run_case(argv, args.out_dir / f'{name}.log', args.tol)
            if args.declared_spd:
                fem = args.hybit_dir / 'target/release/examples/fem_bench'
                if not fem.is_file():
                    raise ValueError(f'build {fem} first')
                for name, extra in [('pcg_jacobi', ['--monitor-only']),
                                    ('pcg_hybrid', [])]:
                    argv = [str(fem), *common, '--probe-iters','12','--monitor-iters','64',
                            '--monitor-rate-limit','0.99','--max-region','128',
                            '--max-escalations','1','--skip-plain',*extra]
                    result['results'][name] = run_case(argv, args.out_dir / f'{name}.log', args.tol)
            (args.out_dir / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'FEM PROBE FAIL: {error}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
