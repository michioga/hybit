#!/usr/bin/env python3
"""C1-G5H: one candidate process per (6 FEM matrices, 4 RHS, 2 T+H policies).

Research-only, fail-closed. Matches committed T-arm baselines but never alters them.
Missing/failed candidates are preserved, not promoted into successes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HEAD='f8d5b6993b159a9d7ce66b47c2eb9f817227bdde'
BINARY='target/release/examples/c1g5h_tplus_hybrid_probe'
MODES=('ones','index_smooth','seeded_random','point_load')
CLI_MODES=('ones','index-smooth','seeded-random','point-load')
ARMS=('tplus_h_static','tplus_h_adaptive')
CLI_ARMS=('tplus-h-static','tplus-h-adaptive')
MATRICES=(
 ('scalar_island_shell','scalar','c1g_p1_island_shell_n144.mtx','2d3f2834e3737610357525a3fdae9abaebb9ee21969d1800b0bd9c15414be45f',20736),
 ('scalar_multi_patch','scalar','c1g_p1_multi_patch_n144.mtx','06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d',20736),
 ('scalar_rotated_anisotropy','scalar','c1g_p1_rotated_anisotropy_n144.mtx','03abf3e7d98f840dbb4a2ed09282d7abfbf28c928cff18837f0087af4fb51c42',20736),
 ('elastic_uniform','elasticity','elastic_uniform.mtx','7a594bd286172a8787db65bde0b9e74bd6bf887473371ce12f7523245f8a23ab',3280),
 ('elastic_stiff_lens','elasticity','elastic_stiff_lens.mtx','f8f2da4fb488dded8d0882c2f0e4ede6f3db3e8ad321231ea368437c7cba9905',3280),
 ('elastic_shear_layers','elasticity','elastic_shear_layers.mtx','b6ba04300933286611802515e9dbfe52ddfd532d6927f85202fc12ed19a4319d',3280),
)
LOG_FIELDS=('case','arm','rhs_mode','rhs_hash','status','iterations','tol','max_iterations',
 'reported_relative','verified_true_relative','compensated_true_relative','relative_x_error',
 'factor_bytes','local_regions','escalations','effective_preconditioner','solver_path','rows','nnz',
 'load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms','inprocess_total_wall_ms',
 'coarse_bytes','local_bytes','local_used')
COLS=('case','family','rhs_mode','arm','matrix_sha256','rhs_hash','status','iterations','tol',
 'max_iterations','verified_true_relative','compensated_true_relative','reported_relative',
 'relative_x_error','factor_bytes','local_regions','escalations','effective_preconditioner',
 'solver_path','rows','nnz','load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms',
 'inprocess_total_wall_ms','coarse_bytes','local_bytes','local_used',
 'elapsed_s','peak_rss_kib','process_exit_status','record_status',
 'baseline_two_level_status','baseline_two_level_wall_ms','matched_outcome')


def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(1<<20):h.update(data)
    return h.hexdigest()


def git(repo:Path,*arg:str)->str:
    return subprocess.check_output(('git',*arg),cwd=repo,text=True,stderr=subprocess.PIPE).strip()


def strict_csv(path:Path)->list[dict]:
    with path.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if not r.fieldnames or len(set(r.fieldnames))!=len(r.fieldnames):raise ValueError('CSV header duplicate')
        rows=list(r)
    if any(None in x or any(v is None for v in x.values()) for x in rows):raise ValueError('ragged CSV')
    return rows


def baseline(repo:Path)->dict:
    out={}
    for path,family,case_names in (
        ('benchmarks/evidence/C1G5G_20261114_results.csv','scalar',MATRICES[:3]),
        ('benchmarks/evidence/C1G5E_20261114_results.csv','elasticity',MATRICES[3:])):
        rows=strict_csv(repo/path)
        for case,_,_,matrix_sha,_ in case_names:
            for rhs in MODES:
                k=(case,rhs)
                choices=[r for r in rows if r['case']==case and r['rhs_mode']==rhs and r['arm']=='two_level_graph_piecewise']
                if len(choices)!=1:raise ValueError('missing/duplicate frozen Two-Level baseline: '+repr(k))
                r=choices[0]
                if r['matrix_sha256']!=matrix_sha or r['record_status']!='MEASURED' or r['process_exit_status']!='0':
                    raise ValueError('baseline incomplete/matrix altered')
                if r['status'] not in ('Converged','MaxIterations','Breakdown'):
                    raise ValueError('baseline status invalid')
                out[k]=r
    if len(out)!=24:raise ValueError('expected 24 frozen T records')
    return out


def check_repo(repo:Path):
    if git(repo,'rev-parse','HEAD')!=HEAD or git(repo,'rev-parse','origin/develop/0.10.0')!=HEAD:
        raise ValueError('repository HEAD/origin mismatch; expected f8d5b69')
    if git(repo,'branch','--show-current')!='develop/0.10.0':raise ValueError('wrong branch')
    if git(repo,'diff','--cached','--name-only'):raise ValueError('staged changes not allowed')
    changed=set(filter(None,git(repo,'diff','--name-only').splitlines()))
    expected={'MANIFEST.txt','SOURCE_SHA256.txt','docs/SESSION_HANDOFF.md','docs/SUMMARY.md'}
    if changed!=expected:raise ValueError('unexpected tracked edits: '+repr(changed))
    subprocess.run(('sha256sum','--check','--strict','--quiet','SOURCE_SHA256.txt'),cwd=repo,check=True)
    if not (repo/'crates/hybit/examples/c1g5h_tplus_hybrid_probe.rs').is_file():
        raise ValueError('C1-G5H research example not installed')


def corpus(scalar:Path,elasticity:Path)->dict:
    result={}
    for case,family,name,digest,n in MATRICES:
        p=(scalar if family=='scalar' else elasticity)/name
        if not p.is_file() or p.is_symlink() or sha(p)!=digest:
            raise ValueError('missing/modified pinned matrix: '+case)
        result[case]=p
    return result


def posfloat(x:str,name:str,strict:bool=False)->float:
    v=float(x)
    if not math.isfinite(v) or v<0 or (strict and v==0):raise ValueError('nonfinite/negative '+name)
    return v


def positive_int(x:str,name:str,zero_ok:bool=True)->int:
    if not re.fullmatch(r'\d+',x):raise ValueError('bad integer '+name)
    v=int(x)
    if not zero_ok and v==0:raise ValueError('zero '+name)
    return v


def parse_log(data:str,case:str,rhs:str,arm:str,n:int,base:dict)->dict:
    lines=[ln.strip() for ln in data.splitlines() if ln.startswith('C1G5H|')]
    if len(lines)!=1:raise ValueError('expected exactly one C1G5H record')
    fields={}
    for item in lines[0].split('|')[1:]:
        k,sep,v=item.partition('=')
        if not sep or k in fields:raise ValueError('duplicate/malformed record field')
        fields[k]=v
    if set(fields)!=set(LOG_FIELDS):raise ValueError('unexpected log field set')
    if (fields['case'],fields['rhs_mode'],fields['arm'])!=(case,rhs,arm):raise ValueError('case/RHS/arm mismatch')
    if fields['rhs_hash']!=base['rhs_hash'] or not re.fullmatch('[0-9a-f]{16}',fields['rhs_hash']):
        raise ValueError('RHS does not match frozen Two-Level baseline')
    if (positive_int(fields['rows'],'rows')!=n or
        positive_int(fields['nnz'],'nnz')<=n or
        (base.get('nnz') is not None and fields['nnz']!=base['nnz']) or
        (base.get('rows') is not None and fields['rows']!=base['rows'])):
        raise ValueError('matrix geometry mismatch')
    if fields['status'] not in ('Converged','MaxIterations','Breakdown'):
        raise ValueError('unsupported solver status')
    its=positive_int(fields['iterations'],'iterations')
    if its>600:raise ValueError('iterations beyond budget')
    if fields['status']=='MaxIterations' and its!=600:raise ValueError('false exhausted budget')
    tol=posfloat(fields['tol'],'tol',True)
    if abs(tol-1e-10)>1e-23 or positive_int(fields['max_iterations'],'max_iterations')!=600:
        raise ValueError('changed tolerance/budget')
    for name in ('reported_relative','verified_true_relative','compensated_true_relative',
        'load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms','inprocess_total_wall_ms'):
        posfloat(fields[name],name)
    if fields['relative_x_error']!='n/a':posfloat(fields['relative_x_error'],'relative_x_error')
    if rhs=='point_load' and fields['relative_x_error']!='n/a':raise ValueError('invented point-load exact solution')
    if fields['status']=='Converged':
        if max(float(fields['verified_true_relative']),float(fields['compensated_true_relative']))>tol*(1+1e-12):
            raise ValueError('false original-CSR/Kahan success')
    factor=positive_int(fields['factor_bytes'],'factor_bytes')
    coarse=positive_int(fields['coarse_bytes'],'coarse_bytes',False)
    local=positive_int(fields['local_bytes'],'local_bytes')
    regions=positive_int(fields['local_regions'],'local_regions')
    switches=positive_int(fields['escalations'],'escalations')
    if coarse+local!=factor or fields['local_used'] not in ('true','false'):
        raise ValueError('memory/use accounting inconsistent')
    if (fields['local_used']=='true') != (local>0 and regions>0):raise ValueError('fake local factor activation')
    if (regions==0)!=(local==0):raise ValueError('unaccounted factors')
    if local>64*1024*1024:raise ValueError('local factor memory policy exceeded')
    if arm=='tplus_h_static':
        if (regions!=8 or local==0 or switches!=0 or
            fields['effective_preconditioner']!='AlgebraicTwoLevelHybridStatic' or
            fields['solver_path']!='C1G5H_STATIC_RESEARCH_COMPOSITE'):
            raise ValueError('static T+H does not use pre-registered local factors')
    else:
        if (switches>1 or fields['effective_preconditioner'] not in
            ('AlgebraicTwoLevelHybridAdaptive','TwoLevelOnlyNoLocal') or
            fields['solver_path']!='HYBIT_AUTO_ALGEBRAIC_COARSE'):
            raise ValueError('adaptive T+H provenance invalid')
        if (regions>0)!=(switches>0):raise ValueError('adaptive escalation/factor disagreement')
        if (regions>0)!=(fields['effective_preconditioner']=='AlgebraicTwoLevelHybridAdaptive'):
            raise ValueError('adaptive label/use mismatch')
    return fields


def parse_time(t:str)->dict:
    fields={}
    for line in t.splitlines():
        if not line.strip():continue
        k,sep,v=line.partition('=')
        if not sep or k in fields:raise ValueError('unexpected time format')
        fields[k]=v.strip()
    if set(fields)!={'elapsed_s','peak_rss_kib','exit_status'}:raise ValueError('GNU time fields missing')
    posfloat(fields['elapsed_s'],'elapsed_s')
    positive_int(fields['peak_rss_kib'],'peak_rss_kib')
    positive_int(fields['exit_status'],'exit_status')
    return fields


def keys():
    return [(c,r,a) for c,_,_,_,_ in MATRICES for r in MODES for a in ARMS]


def reconstruct(repo:Path,ev:Path,base:dict,paths:dict)->tuple[bytes,bytes,dict]:
    manifest=json.loads((ev/'run_manifest.json').read_text(encoding='utf-8'))
    expected={'source_head':HEAD,'binary_sha256':sha(repo/BINARY),
      'matrix_sha256':{c:d for c,_,_,d,_ in MATRICES},
      'arms':list(ARMS),'rhs_modes':list(MODES),'production_promotion_allowed':False}
    for k,v in expected.items():
        if manifest.get(k)!=v:raise ValueError('run manifest drift: '+k)
    all_files={'run_manifest.json','results.csv','summary.json'}
    for c,r,a in keys():all_files.update({f'{c}_{r}_{a}{s}' for s in ('.log','.time','.rc')})
    if {p.name for p in ev.iterdir()}!=all_files or any(p.is_symlink() or not p.is_file() for p in ev.iterdir()):
        raise ValueError('48-run raw file set incomplete or unexpected')
    rows=[];counts={'converged':0,'nonconverged':0,'process_errors':0,'local_used':0,'rescues':0,'regressions':0}
    grid=[]
    for case,family,_,matrix_sha,n in MATRICES:
      for rhs in MODES:
       b=base[(case,rhs)]
       for arm in ARMS:
        stem=f'{case}_{rhs}_{arm}'
        rc_text=(ev/(stem+'.rc')).read_text().strip()
        rc=positive_int(rc_text,'process rc')
        timed=parse_time((ev/(stem+'.time')).read_text())
        if rc!=positive_int(timed['exit_status'],'time exit status'):
            raise ValueError('rc disagrees with time: '+stem)
        line=(ev/(stem+'.log')).read_text()
        if not line:raise ValueError('empty raw log')
        common=dict(case=case,family=family,rhs_mode=rhs,arm=arm,matrix_sha256=matrix_sha,
            elapsed_s=timed['elapsed_s'],peak_rss_kib=timed['peak_rss_kib'],
            process_exit_status=str(rc),baseline_two_level_status=b['status'],
            baseline_two_level_wall_ms=b['inprocess_total_wall_ms'])
        if rc:
            counts['process_errors']+=1
            row=dict(common,status='PROCESS_ERROR',record_status='PROCESS_ERROR',matched_outcome='PROCESS_ERROR')
        else:
            data=parse_log(line,case,rhs,arm,n,b)
            ok=data['status']=='Converged'; b_ok=b['status']=='Converged'
            out=('BOTH_CONVERGED' if ok and b_ok else 'TPLUS_H_RESCUE' if ok else
              'TPLUS_H_REGRESSION' if b_ok else 'BOTH_NOT_CONVERGED')
            if out=='TPLUS_H_RESCUE' and data['local_used']!='true':
                out='COARSE_ONLY_RECOVERY_NOT_LOCAL_RESCUE'
            if out=='TPLUS_H_RESCUE':counts['rescues']+=1
            if out=='TPLUS_H_REGRESSION':counts['regressions']+=1
            counts['converged' if ok else 'nonconverged']+=1
            if data['local_used']=='true':counts['local_used']+=1
            row=dict(common,**data,record_status='MEASURED',matched_outcome=out)
        rows.append({k:str(row.get(k,'')) for k in COLS})
        grid.append({'case':case,'rhs_mode':rhs,'arm':arm,'status':row['status'],
                     'local_used':row.get('local_used',''), 'outcome':row['matched_outcome']})
    summary={'schema':'C1G5H_MATCHED_TPLUS_H_V1', 'source_head':HEAD,'processes':48,
        'matrices':6,'rhs_modes':4,'candidate_arms':2,'complete_groups':24,
        **counts,'results':grid,'production_promotion_allowed':False,
        'training_performed':False,'note':'Static windows are placement controls, not learned/localized physical hard patches. Single-shot cross-commit reference; no statistical timing claims.'}
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=COLS,lineterminator='\n');w.writeheader();w.writerows(rows)
    csv_b=buf.getvalue().encode()
    json_b=(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    return csv_b,json_b,summary


def preflight(repo:Path,scalar:Path,elastic:Path):
    check_repo(repo)
    paths=corpus(scalar,elastic)
    base=baseline(repo)
    binpath=repo/BINARY
    if not binpath.is_file() or binpath.is_symlink():raise ValueError('build release example before running')
    return paths,base


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('mode',choices=('preflight','run','gate'))
    pa.add_argument('repo',type=Path)
    pa.add_argument('scalar',type=Path)
    pa.add_argument('elastic',type=Path)
    pa.add_argument('evidence',type=Path,nargs='?')
    a=pa.parse_args()
    try:
        repo=a.repo.resolve(strict=True);scalar=a.scalar.resolve(strict=True);elastic=a.elastic.resolve(strict=True)
        paths,base=preflight(repo,scalar,elastic)
        if a.mode=='preflight':
            print('C1-G5H PREFLIGHT PASS: 6 pinned matrices, 24 matched T records, 48 T+H processes; no writes');return
        if a.mode=='run':
            if a.evidence is not None:raise ValueError('run creates a unique evidence directory; omit evidence arg')
            root=Path.home()/'c1h11_evidence';root.mkdir(parents=True,exist_ok=True)
            if root.is_symlink():raise ValueError('unsafe evidence root')
            ev=Path(tempfile.mkdtemp(prefix='c1g5h_tplus_h_',dir=root))
            print('C1-G5H EVIDENCE DIRECTORY:',ev,flush=True)
            manifest={'source_head':HEAD,'binary_sha256':sha(repo/BINARY),
                'matrix_sha256':{c:d for c,_,_,d,_ in MATRICES},
                'arms':list(ARMS),'rhs_modes':list(MODES),
                'production_promotion_allowed':False,
                'static_regions':'8 deterministic index-spaced contiguous regions, 128 DOF each; NOT risk-selected',
                'adaptive':'algebraic coarse starts at iteration zero; local escalation requires actual stalled progress'}
            (ev/'run_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
            for case,rhs,arm in keys():
                stem=f'{case}_{rhs}_{arm}'
                family=next(x[1] for x in MATRICES if x[0]==case)
                args=[str(repo/BINARY),'--matrix',str(paths[case]),'--case',case,
                      '--rhs-mode',CLI_MODES[MODES.index(rhs)],'--arm',CLI_ARMS[ARMS.index(arm)]]
                timing=ev/(stem+'.time');log=ev/(stem+'.log')
                with log.open('wb') as out:
                    p=subprocess.run(['/usr/bin/time','-f','elapsed_s=%e\npeak_rss_kib=%M\nexit_status=%x',
                        '-o',str(timing),'timeout','240s',*args],stdout=out,stderr=subprocess.STDOUT,cwd=repo,check=False)
                (ev/(stem+'.rc')).write_text(str(p.returncode)+'\n')
                print('C1-G5H RUN',stem,'rc='+str(p.returncode),flush=True)
            # Results files are part of the audited set; reconstruct first with placeholders.
            (ev/'results.csv').write_bytes(b'pending\n')
            (ev/'summary.json').write_bytes(b'{}\n')
            csv_b,json_b,summary=reconstruct(repo,ev,base,paths)
            (ev/'results.csv').write_bytes(csv_b);(ev/'summary.json').write_bytes(json_b)
            print('C1-G5H RUN PASS:',summary['converged'],'converged,',summary['nonconverged'],
                  'nonconverged,',summary['process_errors'],'process errors,',summary['local_used'],'local actual used')
            print('C1-G5H EVIDENCE COMPLETE:',ev)
        else:
            if a.evidence is None:raise ValueError('gate requires the exact evidence directory')
            ev=a.evidence.resolve(strict=True)
            csv_b,json_b,summary=reconstruct(repo,ev,base,paths)
            if (ev/'results.csv').read_bytes()!=csv_b or (ev/'summary.json').read_bytes()!=json_b:
                raise ValueError('saved CSV/JSON differs from reconstruction')
            print('C1-G5H T+H EVIDENCE GATE PASS:',json.dumps({k:summary[k] for k in ('processes','converged','nonconverged','process_errors','local_used','rescues','regressions')}),
                  'production/model BLOCKED')
    except Exception as e:
        pa.exit(1,'C1-G5H ERROR: '+str(e)+'\n')

if __name__=='__main__':main()
