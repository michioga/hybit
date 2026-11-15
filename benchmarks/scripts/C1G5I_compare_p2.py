#!/usr/bin/env python3
"""Compare source-pinned 12 P2-off and 12 opt-in P2-on trials; descriptive only."""
from __future__ import annotations
import argparse,csv,hashlib,math
from pathlib import Path

PRIOR_SHA='0195f6dba0d1df2caeb91fa9af44e2943cd449c16a7fab23dcf9cd5b9b08d5db'
FIELDS=('case','family','arm','status','iterations','true_relative','kahan_relative','reported_relative',
        'baseline_T_status','baseline_T_iterations','rhs_hash','local_regions','local_bytes',
        'coarse_bytes','escalations','wall_ms','elapsed_s','peak_rss_kib','rc')
CASES=('scalar_multi_patch','elastic_stiff_lens','elastic_shear_layers')
ARMS=('two_level_graph_piecewise','tplus_h_adaptive','c1g5i_coarse_no_segment','c1g5i_coarse_observe')

def read(path:Path,strict_sha:str|None=None)->dict:
    data=path.read_bytes()
    if strict_sha is not None and hashlib.sha256(data).hexdigest()!=strict_sha:
        raise ValueError('pinned prior results SHA-256 mismatch')
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=FIELDS:raise ValueError('unexpected results schema')
        rows=list(reader)
    if len(rows)!=12:raise ValueError('not exactly 12 records')
    m={}
    for r in rows:
        key=(r['case'],r['arm'])
        if key[0] not in CASES or key[1] not in ARMS or key in m or r['rc']!='0':
            raise ValueError('duplicate, invalid, or failed process')
        if r['status'] not in ('Converged','Breakdown','MaxIterations') or not (0<=int(r['iterations'])<=600):
            raise ValueError('invalid outcome')
        if r['status']=='MaxIterations' and int(r['iterations'])!=600:raise ValueError('invalid iteration budget')
        for field in ('true_relative','kahan_relative','reported_relative'):
            v=float(r[field]);
            if not math.isfinite(v) or v<0:raise ValueError('invalid residual')
        if r['status']=='Converged' and max(float(r['true_relative']),float(r['kahan_relative']))>1e-10*(1+1e-12):
            raise ValueError('false convergence')
        m[key]=r
    if set(m)!={(c,a) for c in CASES for a in ARMS}:raise ValueError('incomplete trial grid')
    return m

def compare(a:dict,b:dict)->list[dict]:
    out=[]
    for case in CASES:
        for arm in ARMS:
            p=a[(case,arm)];q=b[(case,arm)]
            if (p['rhs_hash'],p['baseline_T_status'],p['baseline_T_iterations'])!=(q['rhs_hash'],q['baseline_T_status'],q['baseline_T_iterations']):
                raise ValueError('RHS or T baseline mismatch for '+str((case,arm)))
            # coarse-only and direct-T arms must not have local factors; adaptive can.
            if arm!='tplus_h_adaptive' and any(int(q[x])!=0 for x in ('local_regions','local_bytes','escalations')):
                raise ValueError('non-adaptive arm used local correction')
            out.append({'case':case,'arm':arm,'off_status':p['status'],'off_iters':int(p['iterations']),
                'on_status':q['status'],'on_iters':int(q['iterations']),
                'off_true':float(p['true_relative']),'on_true':float(q['true_relative']),
                'on_kahan':float(q['kahan_relative']),
                'local_on':int(q['local_regions'])>0})
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior_off_csv',type=Path)
    parser.add_argument('new_on_csv',type=Path)
    args=parser.parse_args()
    x=compare(read(args.prior_off_csv,PRIOR_SHA),read(args.new_on_csv))
    print('C1-G5I-P2 opt-in vs default, 12 matched conditions (diagnostic; independent full binaries)')
    print('case | arm | off (status/iter/resid) | on (status/iter/resid) | local-on')
    for r in x:
        print(f"{r['case']} | {r['arm']} | {r['off_status']}/{r['off_iters']}/{r['off_true']:.3e} | {r['on_status']}/{r['on_iters']}/{r['on_true']:.3e} | {r['local_on']}")
    print('P2 comparison PASS: statuses preserved, 12 matches; no inference of model promotion')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as e:
        raise SystemExit('C1-G5I-P2 COMPARE ERROR: '+str(e))
