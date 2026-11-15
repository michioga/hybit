#!/usr/bin/env python3
"""Read-only post-run C1-G5H outcome review for exactly 24 matched T baselines.

Revalidates the owner's 48 raw logs through the repository's C1G5H evidence
reconstructor BEFORE analyzing outcomes. This is not a trained selector or
production policy and never modifies HyBIT source or raw evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import sys

CASES=(
 'scalar_island_shell','scalar_multi_patch','scalar_rotated_anisotropy',
 'elastic_uniform','elastic_stiff_lens','elastic_shear_layers')
MODES=('ones','index_smooth','seeded_random','point_load')
ARMS=('tplus_h_adaptive','tplus_h_static')
EXPECTED_HEAD='f8d5b6993b159a9d7ce66b47c2eb9f817227bdde'
MATCHED_FIELDS=('case','family','rhs_mode','arm','t_status','h_status','t_iterations','h_iterations',
 't_wall_ms','h_wall_ms','time_ratio_h_over_t','local_used','local_regions',
 'local_bytes','coarse_bytes','h_peak_rss_kib','t_peak_rss_kib','matched_outcome',
 'rhs_hash','matrix_sha256')


def strict_csv(data:bytes)->list[dict[str,str]]:
    s=data.decode('utf-8')
    reader=csv.DictReader(io.StringIO(s,newline=''))
    if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):
        raise ValueError('missing/duplicate CSV columns')
    result=list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in result):
        raise ValueError('ragged CSV rows')
    return result


def positive_float(s:str,name:str)->float:
    v=float(s)
    if not math.isfinite(v) or v<0:
        raise ValueError('nonfinite or negative '+name)
    return v


def status_outcome(t_success:bool,h_success:bool,local:bool)->str:
    if t_success and h_success:return 'BOTH_CONVERGED'
    if t_success and not h_success:return 'TPLUS_H_REGRESSION'
    if not t_success and not h_success:return 'BOTH_NOT_CONVERGED'
    return 'TPLUS_H_RESCUE' if local else 'COARSE_ONLY_RECOVERY_NOT_LOCAL_RESCUE'


def inspect(rows:list[dict[str,str]],base:dict[tuple[str,str],dict[str,str]]):
    wanted={(case,mode,arm) for case in CASES for mode in MODES for arm in ARMS}
    indexed={}
    for row in rows:
        key=(row.get('case'),row.get('rhs_mode'),row.get('arm'))
        if key not in wanted or key in indexed:
            raise ValueError('unexpected or duplicate result '+repr(key))
        indexed[key]=row
    if set(indexed)!=wanted or set(base)!={(c,m) for c in CASES for m in MODES}:
        raise ValueError('incomplete matched grid')
    detail=[]
    for case in CASES:
        for mode in MODES:
            t=base[(case,mode)]
            t_success=t['status']=='Converged'
            if t['status'] not in ('Converged','MaxIterations','Breakdown'):
                raise ValueError('unexpected baseline status')
            for arm in ARMS:
                h=indexed[(case,mode,arm)]
                if h['record_status']!='MEASURED' or h['process_exit_status']!='0':
                    raise ValueError('process errors cannot be silently excluded')
                h_success=h['status']=='Converged'
                if h['status'] not in ('Converged','MaxIterations','Breakdown'):
                    raise ValueError('unexpected candidate status')
                if h['matrix_sha256']!=t['matrix_sha256'] or h['rhs_hash']!=t['rhs_hash']:
                    raise ValueError('T/T+H matrix or RHS mismatch')
                local=h['local_used']=='true'
                if h['local_used'] not in ('true','false'):
                    raise ValueError('invalid local usage')
                if arm=='tplus_h_static' and not local:
                    raise ValueError('static T+H local correction was not used')
                outcome=status_outcome(t_success,h_success,local)
                if h['matched_outcome']!=outcome:
                    raise ValueError('matched outcome disagrees with certification')
                t_wall=positive_float(t['inprocess_total_wall_ms'],'T wall')
                h_wall=positive_float(h['inprocess_total_wall_ms'],'T+H wall')
                ratio=h_wall/t_wall if t_success and h_success and t_wall>0 else None
                d=dict(case=case,family=h['family'],rhs_mode=mode,arm=arm,
                    t_status=t['status'],h_status=h['status'],
                    t_iterations=t['iterations'],h_iterations=h['iterations'],
                    t_wall_ms=t_wall,h_wall_ms=h_wall,time_ratio_h_over_t=ratio,
                    local_used=local,local_regions=int(h['local_regions']),
                    local_bytes=int(h['local_bytes']),coarse_bytes=int(h['coarse_bytes']),
                    h_peak_rss_kib=int(h['peak_rss_kib']),
                    t_peak_rss_kib=int(t['peak_rss_kib']),
                    matched_outcome=outcome,rhs_hash=t['rhs_hash'],
                    matrix_sha256=t['matrix_sha256'])
                if d['local_bytes']<0 or d['coarse_bytes']<=0 or d['local_regions']<0:
                    raise ValueError('invalid factor bytes')
                detail.append(d)
    return detail


def group_report(detail:list[dict]):
    family_names=('scalar','elasticity')
    groups={}
    for family in ('all',*family_names):
        for arm in ARMS:
            rows=[d for d in detail if d['arm']==arm and (family=='all' or d['family']==family)]
            if not rows:raise ValueError('missing family/arm')
            outcomes=Counter(d['matched_outcome'] for d in rows)
            matched_ratios=[d['time_ratio_h_over_t'] for d in rows if d['time_ratio_h_over_t'] is not None]
            groups[f'{family}:{arm}']={
                'groups':len(rows),'t_converged':sum(d['t_status']=='Converged' for d in rows),
                'h_converged':sum(d['h_status']=='Converged' for d in rows),
                'local_used':sum(d['local_used'] for d in rows),
                'rescue_local':outcomes['TPLUS_H_RESCUE'],
                'recovery_coarse_only':outcomes['COARSE_ONLY_RECOVERY_NOT_LOCAL_RESCUE'],
                'regression':outcomes['TPLUS_H_REGRESSION'],
                'both_converged':outcomes['BOTH_CONVERGED'],
                'both_failed':outcomes['BOTH_NOT_CONVERGED'],
                'matched_converged_pairs':len(matched_ratios),
                'median_wall_ratio_converged_pairs':statistics.median(matched_ratios) if matched_ratios else None,
                'median_h_wall_ms_converged_only':statistics.median(d['h_wall_ms'] for d in rows if d['h_status']=='Converged') if any(d['h_status']=='Converged' for d in rows) else None,
                'max_local_factor_bytes':max(d['local_bytes'] for d in rows),
                'max_h_peak_rss_kib':max(d['h_peak_rss_kib'] for d in rows),
            }
    per_matrix=[]
    for case in CASES:
        b=[d for d in detail if d['case']==case]
        for arm in ARMS:
            x=[d for d in b if d['arm']==arm]
            per_matrix.append({'case':case,'family':x[0]['family'],'arm':arm,
              't_converged':sum(d['t_status']=='Converged' for d in x),
              'h_converged':sum(d['h_status']=='Converged' for d in x),
              'local_used':sum(d['local_used'] for d in x),
              'local_rescue':sum(d['matched_outcome']=='TPLUS_H_RESCUE' for d in x),
              'coarse_only_recovery':sum(d['matched_outcome']=='COARSE_ONLY_RECOVERY_NOT_LOCAL_RESCUE' for d in x),
              'regression':sum(d['matched_outcome']=='TPLUS_H_REGRESSION' for d in x)})
    return groups,per_matrix


def fmt(val,place=3):
    return 'n/a' if val is None else f'{val:.{place}f}'


def render(groups,per_matrix,detail)->str:
    lines=['# C1-G5H — verified T-only vs T+H outcome review','',
      'Research-only descriptive evaluation; no model training or production routing.',
      f'Baseline/current work HEAD: `{EXPECTED_HEAD}`. Single measurement per arm; T-only was measured in earlier checkpoints.',
      '', '## 24 matched groups per candidate policy','',
      '| Family | Policy | T converged | T+H converged | Local used | Local rescue | Coarse-only recovery | Regression | Both failed | Median T+H/T wall ratio* |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fam in ('scalar','elasticity','all'):
        for arm in ARMS:
            o=groups[f'{fam}:{arm}']
            lines.append(f'| {fam} | {arm.removeprefix("tplus_h_")} | {o["t_converged"]}/{o["groups"]} | {o["h_converged"]}/{o["groups"]} | {o["local_used"]} | {o["rescue_local"]} | {o["recovery_coarse_only"]} | {o["regression"]} | {o["both_failed"]} | {fmt(o["median_wall_ratio_converged_pairs"])} |')
    lines+=['','*Single-shot ratio only where both approaches independently converged; timings from different commits/dates are **not** statistically controlled, and this is not a predictor of speed or production regret.*','',
       '## Matrix-level outcomes (four RHS per row)','',
       '| Matrix | Policy | T converged | T+H converged | Local used | Local rescue | Coarse-only recovery | Regression |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
    for d in per_matrix:
        lines.append(f'| {d["case"]} | {d["arm"].removeprefix("tplus_h_")} | {d["t_converged"]}/4 | {d["h_converged"]}/4 | {d["local_used"]} | {d["local_rescue"]} | {d["coarse_only_recovery"]} | {d["regression"]} |')
    lines+=['','## Cost and memory (descriptive)','',
       '| Family | Policy | Median successful T+H wall (ms) | Matched successful pairs | Max local factor bytes | Max process RSS (KiB) |',
       '|---|---|---:|---:|---:|---:|']
    for fam in ('scalar','elasticity'):
        for arm in ARMS:
            o=groups[f'{fam}:{arm}']
            lines.append(f'| {fam} | {arm.removeprefix("tplus_h_")} | {fmt(o["median_h_wall_ms_converged_only"])} | {o["matched_converged_pairs"]} | {o["max_local_factor_bytes"]} | {o["max_h_peak_rss_kib"]} |')
    lines+=['','## 24 matched RHS conditions','',
       '| Case | RHS | T | Adaptive | Adaptive local | Adaptive outcome | Static | Static outcome |',
       '|---|---|---|---|:---:|---|---|---|']
    ix={(d['case'],d['rhs_mode'],d['arm']):d for d in detail}
    for c in CASES:
        for r in MODES:
            a=ix[(c,r,'tplus_h_adaptive')];s=ix[(c,r,'tplus_h_static')]
            lines.append(f'| {c} | {r} | {a["t_status"]} | {a["h_status"]} | {"yes" if a["local_used"] else "no"} | {a["matched_outcome"]} | {s["h_status"]} | {s["matched_outcome"]} |')
    lines+=['','## Guardrails','',
      '- 48 original raw processes, exact saved CSV/JSON, binary SHA, six matrix SHA-256s, T baselines, and certified true/Kahan residuals were checked via C1-G5H evidence reconstructor before this report.',
      '- A no-local adaptive convergence cannot be called a local-Cholesky rescue. Static windows are preselected index-space controls, not discovered local hard patches.',
      '- Coarse and local factor bytes are not process RSS. End-to-end measured timing is one run per arm and across different baseline/candidate run sessions.',
      '- The two synthetic FEM families differ in DOF and physics; do not compare their absolute wall times. No external held-out family, runtime trained selector, or production promotion.',
      '- Nonconverged statuses are preserved; no fastest-infeasible substitution. True-residual-certified success and forward-error guarantees are distinct.','']
    return '\n'.join(lines)


def canonical_csv(detail):
    sio=io.StringIO(newline='')
    writer=csv.DictWriter(sio,fieldnames=MATCHED_FIELDS,lineterminator='\n',extrasaction='raise')
    writer.writeheader()
    for d in detail:
        row=d.copy()
        row['local_used']='true' if row['local_used'] else 'false'
        row['time_ratio_h_over_t']='' if row['time_ratio_h_over_t'] is None else f'{row["time_ratio_h_over_t"]:.12e}'
        writer.writerow(row)
    return sio.getvalue().encode('utf-8')


def verify(repo:Path,scalar:Path,elastic:Path,evidence:Path):
    source=repo/'benchmarks/scripts/C1G5H_tplus_h_evidence.py'
    if not source.is_file() or source.is_symlink():
        raise ValueError('C1-G5H installed evidence parser not found')
    spec=importlib.util.spec_from_file_location('c1g5h_verified_evidence',source)
    if spec is None or spec.loader is None:raise ValueError('failed to load parser')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.HEAD!=EXPECTED_HEAD:raise ValueError('unexpected baseline source HEAD')
    paths,base=mod.preflight(repo,scalar,elastic)
    csv_b,json_b,summ=mod.reconstruct(repo,evidence,base,paths)
    if (evidence/'results.csv').read_bytes()!=csv_b or (evidence/'summary.json').read_bytes()!=json_b:
        raise ValueError('saved results drift from 48 original raw logs')
    if (summ['processes'],summ['process_errors'])!=(48,0):
        raise ValueError('incomplete experimental cohort')
    return strict_csv(csv_b),base,summ


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('mode',choices=('print','write'))
    pa.add_argument('repo',type=Path)
    pa.add_argument('scalar',type=Path)
    pa.add_argument('elastic',type=Path)
    pa.add_argument('evidence',type=Path)
    pa.add_argument('outdir',type=Path,nargs='?')
    args=pa.parse_args()
    try:
        if (args.mode=='print' and args.outdir is not None) or (args.mode=='write' and args.outdir is None):
            raise ValueError('outdir is required for --write and forbidden for --print')
        repo=args.repo.resolve(strict=True);scalar=args.scalar.resolve(strict=True)
        elastic=args.elastic.resolve(strict=True);evidence=args.evidence.resolve(strict=True)
        raw,base,summary=verify(repo,scalar,elastic,evidence)
        detail=inspect(raw,base)
        groups,per_matrix=group_report(detail)
        report=render(groups,per_matrix,detail)
        if args.mode=='print':
            print(report)
        else:
            # Atomic output directory creation. All file names are predetermined;
            # no overwrite, symlink target, or write into the Git repository.
            dest=args.outdir.absolute()
            if dest==repo or repo in dest.parents:
                raise ValueError('review output must be outside Git repository')
            dest.mkdir(parents=False,exist_ok=False)
            try:
                blobs={
                    'C1G5H_outcome_review.md':report.encode('utf-8'),
                    'C1G5H_matched_48.csv':canonical_csv(detail),
                    'C1G5H_review_summary.json':(json.dumps({
                        'schema':'C1G5H_POSTRUN_DESCRIPTIVE_V1',
                        'source_head':EXPECTED_HEAD,
                        'source_evidence_directory':str(evidence),
                        'source_results_sha256':hashlib.sha256((evidence/'results.csv').read_bytes()).hexdigest(),
                        'verified_48_run_counts':{k:summary[k] for k in ('converged','nonconverged','process_errors','local_used','rescues','regressions')},
                        'group_summary':groups,'matrix_summary':per_matrix,
                        'production_promotion_allowed':False,'trained_model':False,
                    },sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8'),
                }
                for name,data in blobs.items():
                    with (dest/name).open('xb') as f:f.write(data)
            except Exception:
                for name in ('C1G5H_outcome_review.md','C1G5H_matched_48.csv','C1G5H_review_summary.json'):
                    (dest/name).unlink(missing_ok=True)
                dest.rmdir()
                raise
            print(report)
            print(f'C1-G5H OUTCOME REVIEW FILES: {dest}')
        print('C1-G5H VERIFIED OUTCOME REVIEW PASS: 48 raw records + 24 frozen T baselines; no source or production change')
    except Exception as exc:
        pa.exit(1,'C1-G5H REVIEW ERROR: '+str(exc)+'\n')

if __name__=='__main__':main()
