#!/usr/bin/env python3
"""Read-only, fail-closed C1-G5E numerical / nominal-cost outcome inspection.

The caller must first pass C1G5E_evidence.py --mode gate against the original
raw logs, binary SHA and corpus. This audit never trains or routes a solver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

HEAD = '04851d7317ae984e8ec15c81e6d7537d0b7144ec'
CASES = ('elastic_uniform', 'elastic_stiff_lens', 'elastic_shear_layers')
RHS = ('ones', 'index_smooth', 'seeded_random', 'point_load')
ARMS = ('jacobi_pcg', 'hybrid_local_cholesky', 'two_level_graph_piecewise', 'multilevel_graph_piecewise')
ARM_SHORT = {'jacobi_pcg': 'J', 'hybrid_local_cholesky': 'H',
             'two_level_graph_piecewise': 'T', 'multilevel_graph_piecewise': 'M'}
MATRIX_SHA256 = {
    'elastic_uniform': '7a594bd286172a8787db65bde0b9e74bd6bf887473371ce12f7523245f8a23ab',
    'elastic_stiff_lens': 'f8f2da4fb488dded8d0882c2f0e4ede6f3db3e8ad321231ea368437c7cba9905',
    'elastic_shear_layers': 'b6ba04300933286611802515e9dbfe52ddfd532d6927f85202fc12ed19a4319d',
}
REQUIRED = ('case', 'rhs_mode', 'arm', 'matrix_sha256', 'rhs_hash', 'status',
            'iterations', 'tol', 'max_iterations', 'verified_true_relative',
            'compensated_true_relative', 'relative_x_error', 'factor_bytes',
            'local_regions', 'escalations', 'effective_preconditioner',
            'arm_work_ms', 'inprocess_total_wall_ms', 'elapsed_s',
            'peak_rss_kib', 'process_exit_status', 'record_status')
COST = 'inprocess_total_wall_ms' # includes load, RHS, setup, solve, verification
OUTCOME_COLUMNS = ('case', 'rhs_mode', 'converged_arms', 'no_feasible_arm',
                   'nominal_fastest_converged_arm', 'nominal_fastest_wall_ms',
                   'jacobi_converged', 'hybrid_converged', 'hybrid_local_regions',
                   'hybrid_outcome_vs_jacobi')


def _read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or len(r.fieldnames) != len(set(r.fieldnames)):
            raise ValueError('missing or duplicate CSV headers')
        if not set(REQUIRED).issubset(r.fieldnames):
            raise ValueError('missing C1-G5E columns: ' + repr(sorted(set(REQUIRED)-set(r.fieldnames))))
        return list(r)


def _finite(row, name, *, positive=False):
    try:
        val = float(row[name])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError('bad numeric field ' + name) from exc
    if not math.isfinite(val) or val < 0 or (positive and val <= 0):
        raise ValueError('invalid nonfinite, negative, or zero field '+name)
    return val


def _integer(row, name):
    x = row[name]
    if not re.fullmatch(r'0|[1-9][0-9]*', x):
        raise ValueError('bad nonnegative integer '+name)
    return int(x)


def audit_rows(rows):
    if len(rows) != 48:
        raise ValueError(f'expected 48 rows, got {len(rows)}')
    by_key = {}
    rhs_fingerprints = {}
    for row in rows:
        key = (row['case'], row['rhs_mode'], row['arm'])
        if (row['case'] not in CASES or row['rhs_mode'] not in RHS
                or row['arm'] not in ARMS or key in by_key):
            raise ValueError('unexpected or duplicate comparison key: '+str(key))
        if row['matrix_sha256'] != MATRIX_SHA256[row['case']]:
            raise ValueError('matrix identity mismatch: '+str(key))
        if not re.fullmatch(r'[0-9a-f]{16}', row['rhs_hash']):
            raise ValueError('invalid RHS fingerprint')
        rhs_key = key[:2]
        if rhs_key in rhs_fingerprints and rhs_fingerprints[rhs_key] != row['rhs_hash']:
            raise ValueError('RHS differs between arms: '+str(rhs_key))
        rhs_fingerprints[rhs_key] = row['rhs_hash']
        if row['record_status'] != 'MEASURED' or row['process_exit_status'] != '0':
            raise ValueError('C1-G5E has unexpected process failure; do not omit it: '+str(key))
        if row['status'] not in ('Converged','MaxIterations','Breakdown'):
            raise ValueError('unknown status: '+str(key))
        iters = _integer(row,'iterations')
        if _integer(row, 'max_iterations') != 600 or iters > 600:
            raise ValueError('iteration contract mismatch')
        if row['status'] == 'MaxIterations' and iters != 600:
            raise ValueError('not a fully expended MaxIterations')
        tol = _finite(row,'tol',positive=True)
        if abs(tol-1e-10) > 1e-22:
            raise ValueError('tolerance differs')
        for name in ('verified_true_relative','compensated_true_relative','arm_work_ms',
                     'inprocess_total_wall_ms','elapsed_s'):
            _finite(row,name)
        for name in ('factor_bytes','local_regions','escalations','peak_rss_kib'):
            _integer(row,name)
        if row['status'] == 'Converged' and any(_finite(row,name) > tol*(1+1e-12)
                for name in ('verified_true_relative','compensated_true_relative')):
            raise ValueError('false convergence: '+str(key))
        if row['rhs_mode'] == 'point_load':
            if row['relative_x_error'] != 'n/a':
                raise ValueError('point_load must not fabricate an x-reference')
        else:
            _finite(row,'relative_x_error')
        if row['arm'] == 'hybrid_local_cholesky':
            has_local = _integer(row,'local_regions') > 0 and _integer(row,'factor_bytes') > 0
            if (row['effective_preconditioner'] == 'Hybrid') != has_local:
                raise ValueError('Hybrid factor-use contradiction')
            if _integer(row,'factor_bytes') > 64*1024*1024:
                raise ValueError('local factor budget exceeded')
        elif _integer(row,'local_regions') or _integer(row,'escalations'):
            raise ValueError('local factors attributed to non-Hybrid arm')
        by_key[key] = row
    for case in CASES:
        for rhs in RHS:
            for arm in ARMS:
                if (case,rhs,arm) not in by_key:
                    raise ValueError('missing candidate: '+str((case,rhs,arm)))
    return by_key


def _hybrid_label(j,h):
    jok = j['status']=='Converged'
    hok = h['status']=='Converged'
    if jok and not hok:return 'HYBRID_REGRESSION'
    if hok and not jok:return 'HYBRID_RESCUE'
    if jok and hok:return 'BOTH_CONVERGED'
    return 'BOTH_NOT_CONVERGED'


def analyze(rows, summary, csv_digest):
    by_key = audit_rows(rows)
    if summary.get('source_head') != HEAD or summary.get('processes') != 48 \
            or summary.get('process_errors') != 0 or summary.get('measured') != 48:
        raise ValueError('source / measured records inconsistent with frozen C1-G5E summary')
    counts = Counter(r['arm'] for r in rows if r['status']=='Converged')
    if sum(counts.values()) != summary.get('converged'):
        raise ValueError('source summary contradicts measured convergence')
    if summary.get('complete_groups') != 12 or summary.get('promotion_allowed') is not False:
        raise ValueError('source summary is missing blocked-promotion contract')
    groups_by_key={(g['case'],g['rhs_mode']):g for g in summary['groups']}
    if len(groups_by_key)!=12:
        raise ValueError('source summary does not contain exactly 12 distinct groups')
    matrix_summary=[]
    by_group=[]
    for case in CASES:
        for rhs in RHS:
            members=[by_key[(case,rhs,arm)] for arm in ARMS]
            valid=[row for row in members if row['status']=='Converged']
            observed=min(valid,key=lambda row:(_finite(row,COST),ARMS.index(row['arm']))) if valid else None
            j=by_key[(case,rhs,'jacobi_pcg')]
            h=by_key[(case,rhs,'hybrid_local_cholesky')]
            g=groups_by_key.get((case,rhs))
            if not g or set(g['valid_numerical_arms']) != {r['arm'] for r in valid}:
                raise ValueError('source summary group disagrees: '+str((case,rhs)))
            by_group.append({
                'case':case,'rhs_mode':rhs,'converged_arms':','.join(ARM_SHORT[r['arm']] for r in valid),
                'no_feasible_arm':len(valid)==0,
                'nominal_fastest_converged_arm':observed['arm'] if observed else 'NO_FEASIBLE_ARM',
                'nominal_fastest_wall_ms':float(observed[COST]) if observed else None,
                'jacobi_converged':j['status']=='Converged',
                'hybrid_converged':h['status']=='Converged',
                'hybrid_local_regions':int(h['local_regions']),
                'hybrid_outcome_vs_jacobi':_hybrid_label(j,h),
                'records':[{
                    'arm':r['arm'],'status':r['status'],'iterations':int(r['iterations']),
                    'true_relative':float(r['verified_true_relative']),
                    'compensated_relative':float(r['compensated_true_relative']),
                    'relative_x_error':None if r['relative_x_error']=='n/a' else float(r['relative_x_error']),
                    'inprocess_total_wall_ms':float(r[COST]),
                    'arm_work_ms':float(r['arm_work_ms']),
                    'peak_rss_kib':int(r['peak_rss_kib']),
                    'local_regions':int(r['local_regions']),
                    'factor_bytes':int(r['factor_bytes'])
                } for r in members]
            })
    for case in CASES:
        sets=[r for r in by_group if r['case']==case]
        matrix_summary.append({'case':case,'converged':sum(sum(record['status']=='Converged' for record in g['records']) for g in sets),
                               'no_feasible_groups':sum(r['no_feasible_arm'] for r in sets)})
    return {'checkpoint':'C1-G5E independent four-arm outcome audit (descriptive, single runs)',
            'source_head':HEAD,'results_sha256':csv_digest,
            'measured':48,'converged':sum(counts.values()),'unconverged':48-sum(counts.values()),
            'per_arm':[{'arm':arm,'converged':counts[arm], 'not_converged':12-counts[arm],
                        'hybrid_local_activated':sum(1 for r in rows if r['arm']==arm and int(r['local_regions'])>0),
                        'median_wall_ms_converged':_median([float(r[COST]) for r in rows
                            if r['arm']==arm and r['status']=='Converged'])} for arm in ARMS],
            'hybrid_labels':dict(Counter(r['hybrid_outcome_vs_jacobi'] for r in by_group)),
            'no_feasible_groups':sum(g['no_feasible_arm'] for g in by_group),
            'per_matrix':matrix_summary, 'groups':by_group,
            'limitations':('Synthetic FEM, three cases in one elasticity generator family. '
                'Every candidate measured once. Nominal best is a within-harness descriptive '
                'minimum among converged candidates; not a robust speedup, trained predictor, '
                'external validation or production recommendation. Hybrid internal setup_ms '
                'overlaps solve_ms, so both are NOT summed; common inprocess_total_wall_ms is used. '
                'No production model promotion allowed.'),
            'promotion_allowed':False}


def _median(arr):
    if not arr:return None
    arr.sort(); n=len(arr)
    return arr[n//2] if n%2 else (arr[n//2-1]+arr[n//2])/2


def render(report):
    out=[]
    out.append('# C1-G5E four-arm outcome audit (one run per arm)')
    out.append('')
    out.append('**Numerical evidence:** 48 measured / '+str(report['converged'])+' converged / '+str(report['unconverged'])+' not converged. Production routing remains BLOCKED.')
    out.append('')
    out.append('## Arm summaries')
    out.append('')
    out.append('| Arm | Converged/12 | Unconverged/12 | Local factors used | Median end-to-end ms (converged only) |')
    out.append('|---|---:|---:|---:|---:|')
    for a in report['per_arm']:
        med='n/a' if a['median_wall_ms_converged'] is None else f"{a['median_wall_ms_converged']:.3f}"
        out.append(f"| `{a['arm']}` | {a['converged']} | {a['not_converged']} | {a['hybrid_local_activated']} | {med} |")
    out.append('')
    out.append('## 12 matched problem groups')
    out.append('')
    out.append('J=Jacobi, H=Hybrid, T=Two-Level, M=Multilevel; ✓ means verified convergence, — means not converged.')
    out.append('')
    out.append('| Matrix | RHS | J | H | T | M | Nominal fastest converged* | Hybrid vs Jacobi |')
    out.append('|---|---|:---:|:---:|:---:|:---:|---|---|')
    for g in report['groups']:
        ss={r['arm']:r['status']=='Converged' for r in g['records']}
        st=['✓' if ss[a] else '—' for a in ARMS]
        fastest=g['nominal_fastest_converged_arm']
        if fastest!='NO_FEASIBLE_ARM':fastest+=f" ({g['nominal_fastest_wall_ms']:.3f} ms)"
        out.append('| '+ ' | '.join([g['case'],g['rhs_mode'],*st,fastest,g['hybrid_outcome_vs_jacobi']])+' |')
    out.append('')
    out.append('*Nominal fastest: minimum measured in-process end-to-end wall time among arms that converged in this single run. It is not a production selection rule. Different model families are not pooled for training.')
    out.append('')
    out.append('## Safety and interpretation')
    out.append('')
    out.append('- `Converged` requires original-CSR and compensated residual checks.')
    out.append('- Failure or MaxIterations never wins a cost comparison.')
    out.append('- Hybrid factor activation is recorded separately; an unactivated Hybrid is effectively JacobiOnly.')
    out.append('- Rust Hybrid internal setup and solve intervals overlap; do not add those phases.')
    out.append('- Use the result CSV and original logs for per-arm true residual, iteration, factor memory and peak RSS.')
    out.append('- This audit does not train or change the embedded/external `hybit-auto` selection model.')
    out.append('')
    return '\n'.join(out)


def output_csv(report):
    buf=io.StringIO(newline='')
    w=csv.DictWriter(buf,fieldnames=OUTCOME_COLUMNS,lineterminator='\n',extrasaction='ignore')
    w.writeheader()
    for x in report['groups']:
        row=x.copy()
        row['no_feasible_arm']=str(x['no_feasible_arm']).lower()
        row['jacobi_converged']=str(x['jacobi_converged']).lower()
        row['hybrid_converged']=str(x['hybrid_converged']).lower()
        row['nominal_fastest_wall_ms']='' if x['nominal_fastest_wall_ms'] is None else f"{x['nominal_fastest_wall_ms']:.6f}"
        w.writerow(row)
    return buf.getvalue()


def build(evidence: Path):
    csv_path=evidence/'results.csv'
    json_path=evidence/'summary.json'
    raw=csv_path.read_bytes()
    rows=_read_csv(csv_path)
    summary=json.loads(json_path.read_text(encoding='utf-8'))
    report=analyze(rows,summary,hashlib.sha256(raw).hexdigest())
    return {'outcome_audit.json':json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n',
            'outcome_audit.csv':output_csv(report),
            'outcome_audit.md':render(report)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,help='new output directory, must not preexist')
    a=p.parse_args()
    try:
        evidence=a.evidence.resolve(strict=True)
        files=build(evidence)
        print(files['outcome_audit.md'])
        if a.out:
            if a.out.exists():
                raise ValueError('audit destination exists; refuses overwrite')
            a.out.mkdir(parents=True)
            for name,data in files.items():
                (a.out/name).write_text(data,encoding='utf-8')
            print('C1-G5E AUDIT ARTIFACTS:',a.out)
        print('C1-G5E OUTCOME AUDIT PASS: descriptive only, zero production/model changes')
    except Exception as exc:
        print('C1-G5E OUTCOME AUDIT FAIL:',str(exc),file=sys.stderr)
        sys.exit(1)

if __name__=='__main__':main()
