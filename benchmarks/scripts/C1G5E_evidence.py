#!/usr/bin/env python3
"""C1-G5E four-way independent-process experimental evidence.

Does not perform runtime solver selection or train/deploy production models.
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
import sys
import tempfile

import C1G5E_elasticity_corpus as corpus

ARMS = ('jacobi_pcg','hybrid_local_cholesky','two_level_graph_piecewise','multilevel_graph_piecewise')
CLI_ARMS = ('jacobi','hybrid','two-level','multilevel')
RHSS = ('ones','index_smooth','seeded_random','point_load')
CLI_RHSS = ('ones','index-smooth','seeded-random','point-load')
HEAD = '04851d7317ae984e8ec15c81e6d7537d0b7144ec'
COLUMNS = ('case','rhs_mode','arm','matrix_sha256','rhs_hash','status','iterations','tol','max_iterations',
           'verified_true_relative','compensated_true_relative','reported_relative',
           'relative_x_error','factor_bytes','local_regions','escalations','effective_preconditioner',
           'solver_path','rows','nnz','load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms',
           'inprocess_total_wall_ms','elapsed_s','peak_rss_kib','process_exit_status','record_status')
FIELD_NAMES = ('case','arm','rhs_mode','rhs_hash','status','iterations','tol','max_iterations',
               'reported_relative','verified_true_relative','compensated_true_relative',
               'relative_x_error','factor_bytes','local_regions','escalations','effective_preconditioner',
               'solver_path','rows','nnz','load_ms','rhs_ms','setup_ms','solve_ms',
               'arm_work_ms','inprocess_total_wall_ms')
FLOATS = ('tol','reported_relative','verified_true_relative','compensated_true_relative',
          'load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms','inprocess_total_wall_ms')
INTS = ('iterations','max_iterations','factor_bytes','local_regions','escalations','rows','nnz')


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd(repo: Path, *args: str) -> str:
    return subprocess.check_output(args,cwd=repo,text=True).strip()


def check_repo(repo: Path):
    if cmd(repo,'git','rev-parse','HEAD') != HEAD:
        raise ValueError('C1-G5E must run on pinned HEAD 04851d7')
    if cmd(repo,'git','branch','--show-current') != 'develop/0.10.0':
        raise ValueError('wrong branch')
    subprocess.run(['sha256sum','--check','--strict','--quiet','SOURCE_SHA256.txt'],cwd=repo,check=True)
    if not (repo/'crates/hybit/examples/c1g5e_four_arm_probe.rs').exists():
        raise ValueError('C1-G5E example not installed')


def check_corpus(folder: Path) -> dict:
    corpus.generate(folder,True)
    manifest=json.loads((folder/'inventory.json').read_text())
    if set(x['case'] for x in manifest['cases']) != set(corpus.CASES):
        raise ValueError('incomplete corpus inventory')
    if manifest['family_id'] != corpus.FAMILY:
        raise ValueError('wrong family')
    return {r['case']:r['matrix_sha256'] for r in manifest['cases']}


def strict_float(s: str, name: str, *, positive=False) -> float:
    try: x=float(s)
    except Exception as e: raise ValueError(f'invalid {name}: {s!r}') from e
    if not math.isfinite(x) or x<0 or (positive and x<=0):
        raise ValueError(f'invalid {name}: {s!r}')
    return x


def strict_int(s: str,name: str) -> int:
    try: n=int(s)
    except Exception as e: raise ValueError(f'invalid {name}: {s!r}') from e
    if str(n)!=s or n<0:
        raise ValueError(f'invalid {name}: {s!r}')
    return n


def parse_log(log: str, *, case: str, rhs: str, arm: str) -> dict:
    records=[line for line in log.splitlines() if line.startswith('C1G5E|')]
    if len(records)!=1:
        raise ValueError(f'{case}/{rhs}/{arm}: expected one C1G5E record, got {len(records)}')
    fields={}
    for chunk in records[0].split('|')[1:]:
        if chunk.count('=')!=1:raise ValueError('invalid record field')
        k,v=chunk.split('=',1)
        if k in fields:raise ValueError('duplicate record field')
        fields[k]=v
    if set(fields)!=set(FIELD_NAMES):
        raise ValueError(f'missing/extra Rust data {sorted(set(FIELD_NAMES)^set(fields))}')
    if (fields['case'],fields['rhs_mode'],fields['arm'])!=(case,rhs,arm):
        raise ValueError('run key mismatch')
    if not re.fullmatch(r'[0-9a-f]{16}',fields['rhs_hash']):
        raise ValueError('invalid RHS fingerprint')
    for f in FLOATS:strict_float(fields[f],f,positive=f=='tol')
    for f in INTS:strict_int(fields[f],f)
    tol=float(fields['tol'])
    if not math.isclose(tol,1e-10,rel_tol=0.0,abs_tol=1e-22):
        raise ValueError('wrong tolerance')
    if fields['max_iterations']!='600' or int(fields['iterations'])>600:
        raise ValueError('wrong iteration budget')
    if fields['status'] not in ('Converged','MaxIterations','Breakdown'):
        raise ValueError('unexpected solver status')
    if fields['status']=='MaxIterations' and fields['iterations']!='600':
        raise ValueError('MaxIterations under budget')
    if fields['status']=='Converged' and any(float(fields[f])>tol*(1+1e-12)
                                                for f in ('verified_true_relative','compensated_true_relative')):
        raise ValueError('false original-CSR/Kahan convergence')
    if fields['relative_x_error']!='n/a':
        strict_float(fields['relative_x_error'],'relative_x_error')
    if rhs=='point_load' and fields['relative_x_error']!='n/a':
        raise ValueError('invented point-load reference')
    if rhs!='point_load' and fields['relative_x_error']=='n/a':
        raise ValueError('missing manufactured forward error')
    if fields['rows']!='3280':raise ValueError('wrong problem dimension')
    if arm!='hybrid_local_cholesky' and (fields['local_regions']!='0' or fields['escalations']!='0'):
        raise ValueError('non-hybrid fabricated local direct')
    if arm=='hybrid_local_cholesky':
        has_factor=int(fields['local_regions'])>0 and int(fields['factor_bytes'])>0
        if (fields['effective_preconditioner']=='Hybrid') != has_factor:
            raise ValueError('Hybrid effect mismatch')
        if int(fields['factor_bytes'])>64*1024*1024:
            raise ValueError('Hybrid factor budget violated')
    return fields


def parse_time(raw: str) -> dict:
    d={}
    for line in raw.splitlines():
        k,sep,v=line.partition('=')
        if sep:
            if k in d:raise ValueError('duplicate GNU time key')
            d[k]=v
    if set(d)!={'elapsed_s','peak_rss_kib','exit_status'}:
        raise ValueError('incomplete GNU time')
    strict_float(d['elapsed_s'],'elapsed_s')
    strict_int(d['peak_rss_kib'],'peak_rss_kib')
    strict_int(d['exit_status'],'exit_status')
    return d


def expected_keys():
    return [(case,rhs,arm,cli_arm,cli_rhs) for case in corpus.CASES
            for rhs,cli_rhs in zip(RHSS,CLI_RHSS)
            for arm,cli_arm in zip(ARMS,CLI_ARMS)]


def read_audit(repo: Path, folder: Path, evidence: Path):
    check_repo(repo)
    matrices=check_corpus(folder)
    if not evidence.is_dir():raise ValueError('missing evidence directory')
    manifest=json.loads((evidence/'run_manifest.json').read_text())
    if manifest.get('source_head')!=HEAD or manifest.get('matrix_sha256')!=matrices:
        raise ValueError('run manifest does not match source/corpus')
    if manifest.get('binary_sha256')!=sha(repo/'target/release/examples/c1g5e_four_arm_probe'):
        raise ValueError('binary changed after run')
    results=[]
    fingerprints={}
    for case,rhs,arm,_,_ in expected_keys():
        stem=f'{case}_{rhs}_{arm}'
        paths=[evidence/(stem+ext) for ext in ('.log','.time','.rc')]
        if not all(p.is_file() for p in paths):
            raise ValueError('missing raw evidence: '+stem)
        rc=strict_int(paths[2].read_text().strip(),'process_exit_status')
        # Retain both setup errors and timeouts; never relabel as convergence.
        times = parse_time(paths[1].read_text()) if paths[1].stat().st_size else {}
        if rc==0 and times.get('exit_status')!='0':raise ValueError('GNU time disagrees with process')
        if rc!=0 and times.get('exit_status')=='0':raise ValueError('nonzero process with successful GNU time')
        base={c:'' for c in COLUMNS}
        base.update(case=case,rhs_mode=rhs,arm=arm,matrix_sha256=matrices[case],
                    process_exit_status=str(rc),record_status='MEASURED' if rc==0 else 'PROCESS_ERROR')
        if times:
            base.update(elapsed_s=times['elapsed_s'],peak_rss_kib=times['peak_rss_kib'])
        if rc==0:
            data=parse_log(paths[0].read_text(),case=case,rhs=rhs,arm=arm)
            base.update(data)
            fpkey=(case,rhs)
            if fpkey in fingerprints and fingerprints[fpkey]!=data['rhs_hash']:
                raise ValueError('same matrix/RHS, different fingerprints across candidates')
            fingerprints[fpkey]=data['rhs_hash']
        results.append(base)
    if len(results)!=48 or len({(r['case'],r['rhs_mode'],r['arm']) for r in results})!=48:
        raise ValueError('incomplete 48-arm audit')
    # If any arm failed, group must not be used as a complete four-way oracle.
    groups=[]
    for case in corpus.CASES:
        for rhs in RHSS:
            rows=[r for r in results if r['case']==case and r['rhs_mode']==rhs]
            successes=[r for r in rows if r['record_status']=='MEASURED' and r['status']=='Converged']
            groups.append(dict(case=case,rhs_mode=rhs,complete=len(rows)==4 and all(r['record_status']=='MEASURED' for r in rows),
                               converged_candidate_count=len(successes),
                               valid_numerical_arms=[r['arm'] for r in successes],
                               blocked_for_training=True))
    summary=dict(checkpoint='C1-G5E independent elasticity FEM research',source_head=HEAD,
                 family_id=corpus.FAMILY,matrices=3,rhs_modes=4,candidate_policies=4,processes=48,
                 process_errors=sum(r['record_status']!='MEASURED' for r in results),
                 measured=sum(r['record_status']=='MEASURED' for r in results),
                 converged=sum(r['status']=='Converged' for r in results),
                 complete_groups=sum(g['complete'] for g in groups),
                 families_in_new_corpus=1,independent_families_with_C1G4=2,
                 promotion_allowed=False,
                 limitations='Synthetic elastic FEM; no external physical RHS; one run per candidate; A9 native research Multilevel, no production routing; no accuracy or speedup generalization; numerical setup errors retained; no learning/model promotion.',
                 groups=groups)
    return results,summary


def serialized(rows,summary):
    out=io.StringIO(newline='')
    w=csv.DictWriter(out,fieldnames=COLUMNS,lineterminator='\n',extrasaction='raise')
    w.writeheader()
    w.writerows(rows)
    return {'results.csv':out.getvalue().encode(),
            'summary.json':(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()}


def write_or_check(repo,folder,evidence,check):
    rows,summary=read_audit(repo,folder,evidence)
    outputs=serialized(rows,summary)
    for name,content in outputs.items():
        target=evidence/name
        if check:
            if not target.exists() or target.read_bytes()!=content:
                raise ValueError('frozen evidence mismatch: '+name)
        else:
            if target.exists() and target.read_bytes()!=content:
                raise ValueError('would overwrite differing evidence: '+name)
            if not target.exists():target.write_bytes(content)
    print('C1-G5E EVIDENCE PASS:',summary['measured'],'measured,',summary['process_errors'],
          'errors, 48 candidates, 12 groups; production promotion BLOCKED')
    print(json.dumps({k:summary[k] for k in ('processes','measured','process_errors','converged','complete_groups','promotion_allowed')},indent=2))


def execute(repo: Path, folder: Path, parent: Path):
    check_repo(repo)
    matrices=check_corpus(folder)
    binary=repo/'target/release/examples/c1g5e_four_arm_probe'
    if not binary.is_file():raise ValueError('build the C1-G5E example first')
    parent.mkdir(parents=True,exist_ok=True)
    evidence=Path(tempfile.mkdtemp(prefix='c1g5e_four_arm_',dir=parent))
    manifest=dict(source_head=HEAD,binary_sha256=sha(binary),matrix_sha256=matrices,
                  contract='tol=1e-10,max_iterations=600,threads=1,zero x,CSR,one candidate per process',
                  mode='research; no production routing')
    (evidence/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('C1-G5E EVIDENCE DIRECTORY:',evidence,flush=True)
    for case,rhs,arm,cli_arm,cli_rhs in expected_keys():
        stem=f'{case}_{rhs}_{arm}'
        print('C1-G5E RUN',stem,flush=True)
        cmdargs=['timeout','240','/usr/bin/time','-f','elapsed_s=%e\npeak_rss_kib=%M\nexit_status=%x',
                 '-o',str(evidence/(stem+'.time')),'env','OMP_NUM_THREADS=1','RAYON_NUM_THREADS=1',
                 str(binary),'--matrix',str(folder/(case+'.mtx')),
                 '--case',case,'--arm',cli_arm,'--rhs-mode',cli_rhs]
        with (evidence/(stem+'.log')).open('wb') as log:
            p=subprocess.run(cmdargs,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=False)
        (evidence/(stem+'.rc')).write_text(str(p.returncode)+'\n')
        if p.returncode:
            print('  candidate error retained:',p.returncode,flush=True)
    write_or_check(repo,folder,evidence,False)
    write_or_check(repo,folder,evidence,True)
    print('C1-G5E RUN COMPLETE:',evidence)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('preflight','run','gate'),required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--evidence',type=Path)
    p.add_argument('--parent',type=Path)
    a=p.parse_args()
    try:
        repo=a.repo.resolve();folder=a.corpus.resolve()
        if a.mode=='preflight':
            check_repo(repo)
            check_corpus(folder)
            print('C1-G5E PREFLIGHT PASS: 3 FEM matrices, 4 RHS, 4 policies; no writes')
        elif a.mode=='run':
            if a.parent is None:raise ValueError('--parent required for run')
            execute(repo,folder,a.parent.resolve())
        else:
            if a.evidence is None:raise ValueError('--evidence required for gate')
            write_or_check(repo,folder,a.evidence.resolve(),True)
    except Exception as exc:
        print('C1-G5E ERROR:',exc,file=sys.stderr)
        raise SystemExit(1)
if __name__=='__main__':main()
