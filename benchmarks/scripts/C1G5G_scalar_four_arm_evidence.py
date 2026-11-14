#!/usr/bin/env python3
"""C1-G5G: matched scalar family, 4 RHS x 4 arms, independent process evidence.

Research only. No production selector is trained or executed. Original raw artifacts
are kept externally. The complete 48-grid includes failed preparations/timeouts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

HEAD = 'a2df4d84458fdbc4f11a62454aba918556bfde0d'
FAMILY = 'scalar_p1_diffusion'
# Taken from frozen C1-G5F SHA-index, not arbitrary matrix path names.
MATRICES = {
    'scalar_island_shell': ('c1g_p1_island_shell_n144.mtx', '2d3f2834e3737610357525a3fdae9abaebb9ee21969d1800b0bd9c15414be45f', 103104),
    'scalar_multi_patch': ('c1g_p1_multi_patch_n144.mtx', '06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d', 103104),
    'scalar_rotated_anisotropy': ('c1g_p1_rotated_anisotropy_n144.mtx', '03abf3e7d98f840dbb4a2ed09282d7abfbf28c928cff18837f0087af4fb51c42', 144002),
}
RHSS = {'ones': 'ones', 'index_smooth': 'index-smooth', 'seeded_random': 'seeded-random', 'point_load': 'point-load'}
ARMS = {'jacobi_pcg': 'jacobi', 'hybrid_local_cholesky': 'hybrid', 'two_level_graph_piecewise': 'two-level', 'multilevel_graph_piecewise': 'multilevel'}
FIELDS = ('case','arm','rhs_mode','rhs_hash','status','iterations','tol','max_iterations','reported_relative','verified_true_relative','compensated_true_relative','relative_x_error','factor_bytes','local_regions','escalations','effective_preconditioner','solver_path','rows','nnz','load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms','inprocess_total_wall_ms')
FLOATS = ('tol','reported_relative','verified_true_relative','compensated_true_relative','load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms','inprocess_total_wall_ms')
INTS = ('iterations','max_iterations','factor_bytes','local_regions','escalations','rows','nnz')
COLUMNS = ('case','family','rhs_mode','arm','matrix_sha256','rhs_hash','status','iterations','tol','max_iterations','verified_true_relative','compensated_true_relative','reported_relative','relative_x_error','factor_bytes','local_regions','escalations','effective_preconditioner','solver_path','rows','nnz','load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms','inprocess_total_wall_ms','elapsed_s','peak_rss_kib','process_exit_status','record_status')
BINARY = 'target/release/examples/c1g5g_scalar_four_arm_probe'
PROBE = 'crates/hybit/examples/c1g5g_scalar_four_arm_probe.rs'
EXPECTED_FEATURE_FILE = 'benchmarks/evidence/C1G5F_20261114_features.csv'
EXPECTED_ELASTIC_FILE = 'benchmarks/evidence/C1G5E_20261114_results.csv'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return digest(path.read_bytes())


def command(repo: Path, *args: str) -> str:
    return subprocess.check_output(args, cwd=repo, text=True).strip()


def check_repo(repo: Path) -> None:
    if command(repo,'git','rev-parse','HEAD') != HEAD:
        raise ValueError('source HEAD differs from C1-G5G pinned a2df4d8')
    if command(repo,'git','branch','--show-current') != 'develop/0.10.0':
        raise ValueError('unexpected branch')
    subprocess.run(['sha256sum','--check','--strict','--quiet','SOURCE_SHA256.txt'],cwd=repo,check=True)
    if not (repo/PROBE).is_file():
        raise ValueError('C1-G5G research example not installed')
    # Validate independent family hashes from frozen C1-G5F source ledger.
    with (repo/EXPECTED_FEATURE_FILE).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for name, (_, h, nnz) in MATRICES.items():
        candidates = [r for r in rows if r['case']==name and r['family']==FAMILY]
        if len(candidates)!=1 or candidates[0]['matrix_sha256']!=h or int(candidates[0]['nnz'])!=nnz:
            raise ValueError('scalar source provenance mismatch: '+name)
    if not (repo/EXPECTED_ELASTIC_FILE).is_file():
        raise ValueError('frozen elasticity four-arm evidence missing')


def check_corpus(folder: Path) -> dict[str,str]:
    verified={}
    for name,(filename,expected,_) in MATRICES.items():
        p=folder/filename
        if not p.is_file() or sha(p)!=expected:
            raise ValueError('matrix missing or SHA-256 mismatch: '+str(p))
        verified[name]=expected
    return verified


def keys():
    return [(case,rhs,arm) for case in MATRICES for rhs in RHSS for arm in ARMS]


def uinteger(value: str, name: str) -> int:
    if not re.fullmatch(r'0|[1-9][0-9]*',value):
        raise ValueError('invalid unsigned integer: '+name)
    return int(value)


def nonnegative(value: str, name: str, positive: bool=False) -> float:
    try: x=float(value)
    except (ValueError, TypeError) as e: raise ValueError('invalid float: '+name) from e
    if not math.isfinite(x) or x<0 or (positive and x==0):
        raise ValueError('invalid finite nonnegative float: '+name)
    return x


def parse_log(raw: str, case: str, rhs: str, arm: str) -> dict[str,str]:
    records=[x for x in raw.splitlines() if x.startswith('C1G5G|')]
    if len(records)!=1:raise ValueError('missing/duplicate C1G5G record')
    data={}
    for piece in records[0].split('|')[1:]:
        if piece.count('=')!=1:raise ValueError('malformed Rust field')
        k,v=piece.split('=',1)
        if k in data:raise ValueError('duplicate Rust field')
        data[k]=v
    if set(data)!=set(FIELDS):raise ValueError('Rust output field mismatch')
    if (data['case'],data['rhs_mode'],data['arm'])!=(case,rhs,arm):
        raise ValueError('wrong case/RHS/arm record')
    if not re.fullmatch(r'[0-9a-f]{16}',data['rhs_hash']):
        raise ValueError('invalid RHS hash')
    for f in FLOATS:nonnegative(data[f],f,positive=f=='tol')
    for f in INTS:uinteger(data[f],f)
    if data['tol']!='1.000000000000e-10' or data['max_iterations']!='600':
        raise ValueError('wrong tolerance/iteration budget')
    if data['rows']!='20736' or int(data['nnz'])!=MATRICES[case][2]:
        raise ValueError('wrong problem dimension/nonzeros')
    if int(data['iterations'])>600 or data['status'] not in ('Converged','MaxIterations','Breakdown'):
        raise ValueError('invalid solver status/iteration count')
    if data['status']=='MaxIterations' and data['iterations']!='600':
        raise ValueError('premature MaxIterations label')
    if data['status']=='Converged':
        tol=float(data['tol'])*(1+1e-12)
        if float(data['verified_true_relative'])>tol or float(data['compensated_true_relative'])>tol:
            raise ValueError('false convergence in original-CSR/Kahan')
    if rhs=='point_load' and data['relative_x_error']!='n/a':
        raise ValueError('point-load has no manufactured solution')
    if rhs!='point_load':nonnegative(data['relative_x_error'],'relative_x_error')
    if arm!='hybrid_local_cholesky':
        if data['local_regions']!='0' or data['escalations']!='0':
            raise ValueError('unexpected factor statistics for non-Hybrid')
    else:
        activated=int(data['local_regions'])>0 and int(data['factor_bytes'])>0
        if (data['effective_preconditioner']=='Hybrid')!=activated:
            raise ValueError('Hybrid activation mismatch')
        if int(data['factor_bytes'])>64*1024*1024:
            raise ValueError('Hybrid memory budget exceeded')
    return data


def parse_time(raw: str) -> dict[str,str]:
    d={}
    for ln in raw.splitlines():
        k,sep,v=ln.partition('=')
        if not sep:
            # GNU time may prepend a diagnostic for nonzero command status.
            if re.fullmatch(r'Command exited with non-zero status [0-9]+',ln) or re.fullmatch(r'Command terminated by signal [0-9]+',ln):
                continue
            raise ValueError('malformed time record')
        if k in d:raise ValueError('duplicate time field')
        d[k]=v
    if set(d)!={'elapsed_s','peak_rss_kib','exit_status'}:
        raise ValueError('missing GNU time field')
    nonnegative(d['elapsed_s'],'elapsed_s')
    uinteger(d['peak_rss_kib'],'peak_rss_kib')
    uinteger(d['exit_status'],'exit_status')
    return d


def audit(repo: Path, corpus: Path, evidence: Path) -> tuple[list[dict[str,str]],dict]:
    check_repo(repo)
    matrices=check_corpus(corpus)
    if not evidence.is_dir():raise ValueError('evidence directory missing')
    manifest=json.loads((evidence/'run_manifest.json').read_text())
    expected={'source_head': HEAD,'family_id': FAMILY,'matrix_sha256':matrices,
              'binary_sha256':sha(repo/BINARY),'rhs_modes':list(RHSS), 'candidate_arms':list(ARMS)}
    for k,v in expected.items():
        if manifest.get(k)!=v:raise ValueError('manifest mismatch: '+k)
    rows=[];rhs_hashes={}
    for case,rhs,arm in keys():
        stem=f'{case}_{rhs}_{arm}'
        paths=[evidence/(stem+ext) for ext in ('.log','.time','.rc')]
        if any(not p.is_file() for p in paths):raise ValueError('missing raw run: '+stem)
        rc=uinteger(paths[2].read_text().strip(),'process_exit_status')
        timing=parse_time(paths[1].read_text()) if paths[1].stat().st_size else {}
        if timing and (rc==0)!=(timing['exit_status']=='0'):
            raise ValueError('GNU time/process disagrees')
        row={k:'' for k in COLUMNS}
        row.update(case=case,family=FAMILY,rhs_mode=rhs,arm=arm,matrix_sha256=matrices[case],
                   process_exit_status=str(rc),record_status='MEASURED' if rc==0 else 'PROCESS_ERROR')
        if timing:row.update(elapsed_s=timing['elapsed_s'],peak_rss_kib=timing['peak_rss_kib'])
        if rc==0:
            data=parse_log(paths[0].read_text(),case,rhs,arm)
            key=(case,rhs)
            if key in rhs_hashes and rhs_hashes[key]!=data['rhs_hash']:
                raise ValueError('RHS hashes disagree among four arms')
            rhs_hashes[key]=data['rhs_hash']
            row.update(data)
        rows.append(row)
    if len(rows)!=48 or len({(r['case'],r['rhs_mode'],r['arm']) for r in rows})!=48:
        raise ValueError('48-grid incomplete')
    converged=[r for r in rows if r['record_status']=='MEASURED' and r['status']=='Converged']
    groups=[]
    for case in MATRICES:
        for rhs in RHSS:
            group=[r for r in rows if r['case']==case and r['rhs_mode']==rhs]
            feasible=[r for r in group if r['record_status']=='MEASURED' and r['status']=='Converged']
            fastest=min(feasible,key=lambda r:float(r['inprocess_total_wall_ms'])) if feasible else None
            groups.append({'case':case,'rhs_mode':rhs,'complete':all(r['record_status']=='MEASURED' for r in group),
                           'converged_arms':[r['arm'] for r in feasible],
                           'nominal_fastest_converged':fastest['arm'] if fastest else None,
                           'blocked_for_model_training':True})
    summary={'checkpoint':'C1-G5G scalar matched four-arm research','source_head':HEAD,'family_id':FAMILY,
             'matrices':3,'rhs_modes':4,'candidate_arms':4,'processes':48,
             'measured':sum(r['record_status']=='MEASURED' for r in rows),
             'process_errors':sum(r['record_status']!='MEASURED' for r in rows),
             'converged':len(converged),'unconverged_measured':sum(r['record_status']=='MEASURED' and r['status']!='Converged' for r in rows),
             'complete_groups':sum(g['complete'] for g in groups),
             'families_with_matched_four_arm_evidence_if_complete':2 if all(g['complete'] for g in groups) else 1,
             'production_promotion_allowed':False,'trained_selector':False,
             'limitations':'Two constructed families; one candidate run per RHS; no physical external FEM; different DOFs across families; no held-out model or timing repeat; native Multilevel remains research-only; failures retained.',
             'groups':groups}
    return rows,summary


def outputs(rows: list[dict[str,str]],summary: dict) -> dict[str,bytes]:
    f=io.StringIO(newline='')
    w=csv.DictWriter(f,fieldnames=COLUMNS,lineterminator='\n',extrasaction='raise')
    w.writeheader();w.writerows(rows)
    return {'results.csv':f.getvalue().encode(),
            'summary.json':(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()}


def freeze(repo: Path, corpus: Path, evidence: Path, check:bool):
    rows,summary=audit(repo,corpus,evidence)
    for name,data in outputs(rows,summary).items():
        target=evidence/name
        if check:
            if not target.is_file() or target.read_bytes()!=data:
                raise ValueError('evidence result mismatch: '+name)
        else:
            if target.exists() and target.read_bytes()!=data:
                raise ValueError('refusing to overwrite existing differing evidence')
            if not target.exists():target.write_bytes(data)
    print('C1-G5G EVIDENCE PASS:',summary['measured'],'measured,',summary['process_errors'],
          'process errors,',summary['converged'],'converged,',summary['complete_groups'],
          'complete groups; production/model promotion BLOCKED')
    print(json.dumps({k:summary[k] for k in ('processes','measured','process_errors','converged','unconverged_measured','complete_groups','production_promotion_allowed')},indent=2))


def run(repo:Path,corpus:Path,parent:Path):
    check_repo(repo);matrices=check_corpus(corpus)
    binary=repo/BINARY
    if not binary.is_file():raise ValueError('build the scalar research example first')
    parent.mkdir(parents=True,exist_ok=True)
    evidence=Path(tempfile.mkdtemp(prefix='c1g5g_scalar_four_arm_',dir=parent))
    manifest={'source_head':HEAD,'family_id':FAMILY,'matrix_sha256':matrices,
              'binary_sha256':sha(binary),'rhs_modes':list(RHSS),'candidate_arms':list(ARMS),
              'contract':'same RHS generators and 4 arm definitions as C1-G5E; tol 1e-10; max_iters 600; x=0; 1 thread; separate processes; per-run setup+solve',
              'training_allowed':False,'production_promotion_allowed':False}
    (evidence/'run_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print('C1-G5G EVIDENCE DIRECTORY:',evidence,flush=True)
    for case,rhs,arm in keys():
        stem=f'{case}_{rhs}_{arm}'
        print('C1-G5G RUN',stem,flush=True)
        args=['timeout','240','/usr/bin/time','-f','elapsed_s=%e\npeak_rss_kib=%M\nexit_status=%x',
              '-o',str(evidence/(stem+'.time')),'env','OMP_NUM_THREADS=1','RAYON_NUM_THREADS=1',
              str(binary),'--matrix',str(corpus/MATRICES[case][0]),'--case',case,
              '--arm',ARMS[arm],'--rhs-mode',RHSS[rhs]]
        with (evidence/(stem+'.log')).open('wb') as out:
            p=subprocess.run(args,cwd=repo,stdout=out,stderr=subprocess.STDOUT,check=False)
        (evidence/(stem+'.rc')).write_text(str(p.returncode)+'\n')
        if p.returncode:print('  process failure retained:',p.returncode,flush=True)
    freeze(repo,corpus,evidence,False)
    freeze(repo,corpus,evidence,True)
    print('C1-G5G RUN COMPLETE:',evidence)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('preflight','run','gate'),required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--evidence',type=Path)
    p.add_argument('--parent',type=Path)
    a=p.parse_args()
    try:
        repo=a.repo.resolve();corpus=a.corpus.resolve()
        if a.mode=='preflight':
            check_repo(repo);check_corpus(corpus)
            print('C1-G5G PREFLIGHT PASS: 3 pinned scalar matrices, 4 RHS, 4 arms, research-only; no writes')
        elif a.mode=='run':
            if a.parent is None:raise ValueError('--parent required')
            run(repo,corpus,a.parent.resolve())
        else:
            if a.evidence is None:raise ValueError('--evidence required')
            freeze(repo,corpus,a.evidence.resolve(),True)
    except Exception as e:
        p.exit(1,'C1-G5G EVIDENCE ERROR: '+str(e)+'\n')

if __name__=='__main__':main()
