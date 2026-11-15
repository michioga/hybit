#!/usr/bin/env python3
"""C1-G5I-P2 opt-in certified-restart differential; no tracked source edits."""
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
import shutil
import subprocess
import sys
import tempfile

HEAD = 'b918b4591d27a6c23e3571075eb426babcf080bb'
RUST_SOURCE = 'crates/hybit/examples/c1g5h_tplus_hybrid_probe.rs'
RUST_SOURCE_SHA = '184c0e65fdc69c389c1011d2bc4f0b88accf590920e661fa3d9d8a01b1a79b04'
TEMP_SOURCE = 'crates/hybit/examples/c1g5i_p2_restart_probe.rs'
BINARY = 'target/release/examples/c1g5i_p2_restart_probe'
CASES = (
    ('scalar_multi_patch', 'scalar', 'c1g_p1_multi_patch_n144.mtx', '06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d',20736),
    ('elastic_stiff_lens', 'elasticity', 'elastic_stiff_lens.mtx', 'f8f2da4fb488dded8d0882c2f0e4ede6f3db3e8ad321231ea368437c7cba9905',3280),
    ('elastic_shear_layers', 'elasticity', 'elastic_shear_layers.mtx', 'b6ba04300933286611802515e9dbfe52ddfd532d6927f85202fc12ed19a4319d',3280),
)
ARMS = (
    ('two-level', 'two_level_graph_piecewise'),
    ('tplus-h-adaptive', 'tplus_h_adaptive'),
    ('c1g5i-no-segment', 'c1g5i_coarse_no_segment'),
    ('c1g5i-observe', 'c1g5i_coarse_observe'),
)
KEYS = ('case','arm','rhs_mode','rhs_hash','status','iterations','tol','max_iterations',
        'reported_relative','verified_true_relative','compensated_true_relative','relative_x_error',
        'factor_bytes','local_regions','escalations','effective_preconditioner','solver_path',
        'rows','nnz','load_ms','rhs_ms','setup_ms','solve_ms','arm_work_ms',
        'inprocess_total_wall_ms','coarse_bytes','local_bytes','local_used')
RESULT_FIELDS = ('case','family','arm','status','iterations','true_relative','kahan_relative',
                 'reported_relative','baseline_T_status','baseline_T_iterations','rhs_hash',
                 'local_regions','local_bytes','coarse_bytes','escalations','wall_ms',
                 'elapsed_s','peak_rss_kib','rc')


def digest(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def sha(path:Path)->str: return digest(path.read_bytes())
def swap_once(text:str,old:str,new:str)->str:
    count=text.count(old)
    if count!=1:raise ValueError(f'unsafe Rust anchor count {count}: {old[:90]!r}')
    return text.replace(old,new,1)


def transform(data:bytes)->bytes:
    if digest(data)!=RUST_SOURCE_SHA:raise ValueError('C1-G5H Rust source SHA mismatch; refusing transformation')
    s=data.decode('utf-8')
    s=swap_once(s,'        TPlusHAdaptive,\n    }\n\n    fn parse_e_args()',
      '        TPlusHAdaptive,\n        C1G5INoSegment,\n        C1G5IObserve,\n    }\n\n    fn parse_e_args()')
    s=swap_once(s,'                        "tplus-h-adaptive" => FourArm::TPlusHAdaptive,',
      '                        "tplus-h-adaptive" => FourArm::TPlusHAdaptive,\n                        "c1g5i-no-segment" => FourArm::C1G5INoSegment,\n                        "c1g5i-observe" => FourArm::C1G5IObserve,')
    s=swap_once(s,'            FourArm::TPlusHAdaptive => {',
      '            FourArm::TPlusHAdaptive | FourArm::C1G5INoSegment | FourArm::C1G5IObserve => {')
    start=s.index('            FourArm::TPlusHAdaptive | FourArm::C1G5INoSegment')
    end=s.index('            FourArm::Multilevel => {',start)
    arm=s[start:end]
    arm=swap_once(arm,'                solver.set_hybrid_options(config)?;', '''                // C1-G5I-P2 read-only controller ablations; no library algorithm changes.
                if matches!(arm, FourArm::C1G5INoSegment) {
                    // Keep the coarse base prepared (coarse_requested depends on enabled).
                    // Disable monitoring and make initial/local escalation impossible for
                    // a finite residual ratio; the same coarse PCG remains live.
                    config.enabled = true;
                    config.progress_monitor_interval = 0;
                    config.progress_monitor_rate_limit = None;
                    config.escalation_residual_ratio = f64::MAX;
                } else if matches!(arm, FourArm::C1G5IObserve) {
                    // Keep the same true-residual monitor cadence without any local promotion.
                    config.progress_monitor_observe_only = true;
                }
                solver.set_hybrid_options(config)?;''')
    s=s[:start]+arm+s[end:]
    s=swap_once(s, '            FourArm::TPlusHAdaptive => "tplus_h_adaptive",',
      '            FourArm::TPlusHAdaptive => "tplus_h_adaptive",\n            FourArm::C1G5INoSegment => "c1g5i_coarse_no_segment",\n            FourArm::C1G5IObserve => "c1g5i_coarse_observe",')
    return ('// C1-G5I-P2 opt-in certified restart; source pinned to b918b45.\n'+s).encode('utf-8')


def git(repo:Path,*args:str)->str:
    return subprocess.check_output(('git',*args),cwd=repo,stderr=subprocess.PIPE,text=True).strip()


def check(repo:Path,scalar:Path,elastic:Path,need_free:bool=True):
    if git(repo,'branch','--show-current')!='develop/0.10.0':raise ValueError('wrong branch')
    if git(repo,'rev-parse','HEAD')!=HEAD or git(repo,'rev-parse','origin/develop/0.10.0')!=HEAD:
        raise ValueError('HEAD/origin mismatch: require b918b45')
    if git(repo,'diff','--name-only') or git(repo,'diff','--cached','--name-only'):
        raise ValueError('tracked modifications present; no repository writes permitted')
    if need_free and ((repo/TEMP_SOURCE).exists() or (repo/TEMP_SOURCE).is_symlink()):raise ValueError('temporary research example path already exists; refusing overwrite')
    if sha(repo/RUST_SOURCE)!=RUST_SOURCE_SHA:raise ValueError('pinned Rust source mismatched')
    transform((repo/RUST_SOURCE).read_bytes())
    subprocess.run(('sha256sum','--check','--strict','--quiet','SOURCE_SHA256.txt'),cwd=repo,check=True)
    paths={}
    for case,fam,name,d,n in CASES:
        p=(scalar if fam=='scalar' else elastic)/name
        if not p.is_file() or p.is_symlink() or sha(p)!=d:raise ValueError(f'missing/modified matrix: {case}')
        paths[case]=p
    return paths


def read_csv(path:Path)->list[dict]:
    with path.open(newline='',encoding='utf8') as f:
        r=csv.DictReader(f)
        if not r.fieldnames or len(r.fieldnames)!=len(set(r.fieldnames)):raise ValueError('bad CSV header')
        rows=list(r)
    if any(None in row or None in row.values() for row in rows):raise ValueError('ragged CSV')
    return rows


def baselines(repo:Path)->dict:
    base={}
    scalar=read_csv(repo/'benchmarks/evidence/C1G5G_20261114_results.csv')
    elastic=read_csv(repo/'benchmarks/evidence/C1G5E_20261114_results.csv')
    for case,fam,_,matrix_sha,n in CASES:
        options=[r for r in (scalar if fam=='scalar' else elastic) if r['case']==case and r['rhs_mode']=='ones' and r['arm']=='two_level_graph_piecewise']
        if len(options)!=1:raise ValueError('missing/duplicated frozen T baseline')
        r=options[0]
        if r['matrix_sha256']!=matrix_sha or int(r['rows'])!=n or r['record_status']!='MEASURED' or r['process_exit_status']!='0':raise ValueError('frozen T baseline not valid')
        base[case]=r
    return base


def parse_record(raw:str,case:str,arm:str,base:dict,n:int)->dict:
    lines=[s.strip() for s in raw.splitlines() if s.startswith('C1G5H|')]
    if len(lines)!=1:raise ValueError('missing/multiple C1G5H results')
    f={}
    for item in lines[0].split('|')[1:]:
        key,sep,value=item.partition('=')
        if not sep or key in f:raise ValueError('malformed/duplicate key')
        f[key]=value
    if set(f)!=set(KEYS):raise ValueError(f'unexpected record fields: {set(f)^set(KEYS)}')
    if (f['case'],f['arm'],f['rhs_mode'])!=(case,arm,'ones'):raise ValueError('wrong case/arm/RHS')
    if f['rhs_hash']!=base['rhs_hash']:raise ValueError('RHS fingerprint mismatch')
    if int(f['rows'])!=n or f['nnz']!=base['nnz']:raise ValueError('matrix geometry mismatch')
    if f['status'] not in ('Converged','Breakdown','MaxIterations'):raise ValueError('invalid status')
    if f['max_iterations']!='600' or abs(float(f['tol'])-1e-10)>1e-23:raise ValueError('changed numerical policy')
    if not re.fullmatch(r'\d+',f['iterations']) or int(f['iterations'])>600:raise ValueError('invalid iterations')
    if f['status']=='MaxIterations' and f['iterations']!='600':raise ValueError('max-iterations status before budget')
    for k in ['reported_relative','verified_true_relative','compensated_true_relative','inprocess_total_wall_ms']:
        v=float(f[k]);
        if not math.isfinite(v) or v<0:raise ValueError('nonfinite/negative '+k)
    if f['status']=='Converged' and max(float(f['verified_true_relative']),float(f['compensated_true_relative']))>1e-10*(1+1e-12):raise ValueError('false certified success')
    if f['relative_x_error']=='n/a':raise ValueError('manufactured ones RHS must have reference solution')
    if not math.isfinite(float(f['relative_x_error'])):raise ValueError('nonfinite forward error')
    reg=int(f['local_regions']);lb=int(f['local_bytes']);cb=int(f['coarse_bytes']);fac=int(f['factor_bytes']);esc=int(f['escalations'])
    if min(reg,lb,cb,fac,esc)<0 or lb>64*1024*1024:raise ValueError('invalid factor metrics')
    if arm=='two_level_graph_piecewise':
        if f['local_used']!='false' or reg or lb or esc:raise ValueError('direct T used local')
        # Legacy direct T has coarse_bytes=0 and factor_bytes=the TwoLevel factor
        if cb!=0 or fac==0:raise ValueError('direct T factor accounting')
    else:
        if cb<=0 or lb+cb!=fac:raise ValueError('Auto coarse/factor accounting')
        if (lb>0 or reg>0)!=(f['local_used']=='true'):raise ValueError('local-use label mismatch')
        if arm!='tplus_h_adaptive' and (lb or reg or esc):raise ValueError('no-local control secretly escalated')
    return f


def parse_gnu_time(contents:str,rc:int,stem:str)->dict[str,str]:
    """GNU time emits a diagnostic preface for nonzero exits; never launder it."""
    lines=contents.splitlines()
    diagnostics=[]
    metrics={}
    for line in lines:
        if line.startswith(('Command exited with non-zero status ', 'Command terminated by signal ')):
            diagnostics.append(line)
            continue
        key,sep,val=line.partition('=')
        if not sep or key not in ('elapsed_s','peak_rss_kib','exit_status') or key in metrics:
            raise ValueError(f'{stem}: invalid GNU time line: {line!r}')
        metrics[key]=val
    if set(metrics)!={'elapsed_s','peak_rss_kib','exit_status'}:
        raise ValueError(f'{stem}: missing GNU time metrics')
    if not re.fullmatch(r'\d+',metrics['exit_status']) or int(metrics['exit_status'])!=rc:
        raise ValueError(f'{stem}: GNU time exit_status mismatch (rc={rc})')
    try:
        elapsed=float(metrics['elapsed_s'])
        rss=int(metrics['peak_rss_kib'])
    except ValueError as exc:
        raise ValueError(f'{stem}: invalid GNU time numeric field') from exc
    if not math.isfinite(elapsed) or elapsed<0 or rss<0:
        raise ValueError(f'{stem}: invalid GNU time value')
    if diagnostics and rc==0:
        raise ValueError(f'{stem}: unexpected GNU time process failure diagnostic')
    if len(diagnostics)>1:
        raise ValueError(f'{stem}: multiple GNU time failure diagnostics')
    return metrics


def fileset()->set[str]:
    names={'manifest.json','results.csv','summary.json','generated_source.rs','build.log'}
    for case,*_ in CASES:
        for _,arm in ARMS:
            stem=f'{case}__{arm}'
            names.update({stem+s for s in ('.stdout','.stderr','.time','.rc')})
    return names


def reconstruct(repo:Path,ev:Path,finalized:bool=True)->tuple[bytes,bytes,dict]:
    if not ev.is_dir() or ev.is_symlink():raise ValueError('evidence dir unavailable')
    found={p.name for p in ev.iterdir()}
    expected=fileset() if finalized else fileset()-{'results.csv','summary.json'}
    if found!=expected or any(p.is_symlink() or not p.is_file() for p in ev.iterdir()):raise ValueError('incomplete evidence file set')
    manifest=json.loads((ev/'manifest.json').read_text())
    if manifest.get('source_head')!=HEAD or manifest.get('source_sha256')!=RUST_SOURCE_SHA or manifest.get('generated_sha256')!=sha(ev/'generated_source.rs') or manifest.get('binary_sha256')!=sha(repo/BINARY) or manifest.get('cases')!=[x[0] for x in CASES] or manifest.get('arms')!=[x[1] for x in ARMS] or manifest.get('production_promotion_allowed') is not False or manifest.get('p2_certified_restart_feature') is not True:raise ValueError('evidence provenance invalid')
    base=baselines(repo)
    rows=[];comparisons=[]
    for case,fam,_,d,n in CASES:
        if manifest.get('matrix_sha256',{}).get(case)!=d:raise ValueError('matrix provenance altered')
        for _,arm in ARMS:
            stem=f'{case}__{arm}'
            rc_s=(ev/(stem+'.rc')).read_text().strip()
            if not re.fullmatch(r'\d+',rc_s):raise ValueError('invalid process rc')
            rc=int(rc_s)
            timeval=parse_gnu_time((ev/(stem+'.time')).read_text(),rc,stem)
            if rc!=0:
                stderr=(ev/(stem+'.stderr')).read_text().strip()
                stdout=(ev/(stem+'.stdout')).read_text().strip()
                tail=(stderr or stdout)[-600:]
                raise ValueError(f'{stem}: process error rc={rc}; stderr/stdout tail={tail!r}; raw evidence retained; no success claim')
            f=parse_record((ev/(stem+'.stdout')).read_text(),case,arm,base[case],n)
            r={
                'case':case,'family':fam,'arm':arm,'status':f['status'],'iterations':f['iterations'],
                'true_relative':f['verified_true_relative'],'kahan_relative':f['compensated_true_relative'],
                'reported_relative':f['reported_relative'],'baseline_T_status':base[case]['status'],
                'baseline_T_iterations':base[case]['iterations'],'rhs_hash':f['rhs_hash'],
                'local_regions':f['local_regions'],'local_bytes':f['local_bytes'],
                'coarse_bytes':f['coarse_bytes'],'escalations':f['escalations'],
                'wall_ms':f['inprocess_total_wall_ms'],'elapsed_s':timeval['elapsed_s'],
                'peak_rss_kib':timeval['peak_rss_kib'],'rc':str(rc)
            }
            rows.append(r)
            comparisons.append({'case':case,'arm':arm,'status':f['status'],'iterations':int(f['iterations']),
                'local_used':f['local_used']=='true','true_relative':float(f['verified_true_relative'])})
    sio=io.StringIO(newline='');writer=csv.DictWriter(sio,fieldnames=RESULT_FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary={'schema':'C1G5I_P2_CERTIFIED_RESTART_DIFFERENTIAL_V1','source_head':HEAD,'source_sha256':RUST_SOURCE_SHA,
        'processes':12,'comparisons':comparisons,'p2_certified_restart_feature':True,'production_promotion_allowed':False,
        'hypotheses_only':'Opt-in P2 is compiled into HyBIT Auto. Compare unchanged direct T and three existing Auto controller paths with P2-off prior evidence, without claiming the full numerical root cause.'}
    jb=(json.dumps(summary,indent=2,sort_keys=True)+'\n').encode()
    return sio.getvalue().encode(),jb,summary


def run(repo:Path,scalar:Path,elastic:Path)->Path:
    paths=check(repo,scalar,elastic)
    baselines(repo)
    parent=Path.home()/'c1h11_evidence';parent.mkdir(exist_ok=True)
    ev=Path(tempfile.mkdtemp(prefix='c1g5i_p2_restart_',dir=parent))
    print('C1-G5I-P2 EVIDENCE DIRECTORY (including incomplete runs): '+str(ev),flush=True)
    temp=repo/TEMP_SOURCE
    source=transform((repo/RUST_SOURCE).read_bytes())
    env=os.environ.copy();env.update({'RAYON_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    try:
        with temp.open('xb') as f:f.write(source)
        with (ev/'build.log').open('w') as f:
            for command in [
                ('rustfmt','+1.99.1',str(temp)),
                ('cargo','+1.99.1','test','--release','--locked','-p','hybit','--features','p2-certified-restart','--example','c1g5i_p2_restart_probe'),
                ('cargo','+1.99.1','clippy','--release','--locked','-p','hybit','--features','p2-certified-restart','--example','c1g5i_p2_restart_probe','--','-D','warnings'),
                ('cargo','+1.99.1','build','--release','--locked','-p','hybit','--features','p2-certified-restart','--example','c1g5i_p2_restart_probe'),
            ]:
                f.write('COMMAND '+' '.join(command)+'\n');f.flush()
                subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(temp,ev/'generated_source.rs')
        manifest={'source_head':HEAD,'source_sha256':RUST_SOURCE_SHA,
                  'generated_sha256':sha(temp),'binary_sha256':sha(repo/BINARY),
                  'matrix_sha256':{c:d for c,_,_,d,_ in CASES},
                  'cases':[c for c,*_ in CASES],'arms':[name for _,name in ARMS],
                  'p2_certified_restart_feature':True,'production_promotion_allowed':False}
        (ev/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        for case,fam,_,_,n in CASES:
            for cli,arm in ARMS:
                stem=f'{case}__{arm}'
                timefile=ev/(stem+'.time')
                command=['/usr/bin/time','-f','elapsed_s=%e\npeak_rss_kib=%M\nexit_status=%x',
                    '-o',str(timefile),'timeout','240s',str(repo/BINARY),'--matrix',str(paths[case]),
                    '--case',case,'--rhs-mode','ones','--arm',cli]
                p=subprocess.run(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                (ev/(stem+'.stdout')).write_text(p.stdout)
                (ev/(stem+'.stderr')).write_text(p.stderr)
                (ev/(stem+'.rc')).write_text(str(p.returncode)+'\n')
                print(f'C1-G5I-P2 RUN {case} {arm} rc={p.returncode}',flush=True)
        records,summary,doc=reconstruct(repo,ev,finalized=False)
        (ev/'results.csv').write_bytes(records)
        (ev/'summary.json').write_bytes(summary)
        reconstruct(repo,ev)
        print('C1-G5I-P2 EVIDENCE PASS: 12 current-HEAD trials, no production changes; source example removed after run')
        print('C1-G5I-P2 EVIDENCE DIRECTORY: '+str(ev))
        return ev
    finally:
        temp.unlink(missing_ok=True)


def main()->int:
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('mode',choices=('preflight','run','gate'))
    pa.add_argument('repo',type=Path)
    pa.add_argument('scalar',type=Path)
    pa.add_argument('elastic',type=Path)
    pa.add_argument('evidence',nargs='?',type=Path)
    args=pa.parse_args()
    repo=args.repo.resolve(strict=True)
    scalar=args.scalar.resolve(strict=True)
    elastic=args.elastic.resolve(strict=True)
    try:
        check(repo,scalar,elastic)
        baselines(repo)
        if args.mode=='preflight':
            if args.evidence is not None:raise ValueError('preflight does not take evidence dir')
            print('C1-G5I-P2 PREFLIGHT PASS: pinned b918b45, three matrices, 12 controls, opt-in P2; no writes')
        elif args.mode=='run':
            if args.evidence is not None:raise ValueError('run does not take evidence dir')
            run(repo,scalar,elastic)
        else:
            if args.evidence is None:raise ValueError('gate requires evidence directory')
            raw,summary,doc=reconstruct(repo,args.evidence.resolve(strict=True))
            if (args.evidence/'results.csv').read_bytes()!=raw or (args.evidence/'summary.json').read_bytes()!=summary:
                raise ValueError('saved outcome results altered')
            for case,*_ in CASES:
                vals=[x for x in doc['comparisons'] if x['case']==case]
                print(case+': '+', '.join(f"{v['arm']}={v['status']}/{v['iterations']} local={v['local_used']} true={v['true_relative']:.3e}" for v in vals))
            print('C1-G5I-P2 GATE PASS: 12 feature-on controls; diagnostic only; production BLOCKED')
        return 0
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print(f'C1-G5I-P2 ERROR: {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
