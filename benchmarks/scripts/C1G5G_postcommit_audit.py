#!/usr/bin/env python3
"""Strict read-only, post-commit numerical audit for C1-G5G + comparison with C1-G5E.

The scalar run's manifest legitimately records a2df4d8 (pre-commit). The present
repository is the checked-in 5394cb4. We do NOT replace the run's historical HEAD,
nor mutate its verifier. Every original process and both frozen outputs are re-read.
No production routing, model training or source edits occur.
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
import re
import subprocess
import sys

SOURCE_HEAD = 'a2df4d84458fdbc4f11a62454aba918556bfde0d'
POST_HEAD = '5394cb4083fb04cbaa20acdcf92898ff7e28ede5'
EXPECTED_ELASTIC_CSV_SHA = '9b06a1fc0185822690d35a1b7754a5f40272aa3593a3606c090309b1cb906c2c'
ARMS = ('jacobi_pcg', 'hybrid_local_cholesky', 'two_level_graph_piecewise', 'multilevel_graph_piecewise')
SHORT = dict(zip(ARMS, ('J', 'H', 'T', 'M')))
OUT_COLS = ('case','rhs_mode','jacobi','hybrid','two_level','multilevel','fastest_converged','fastest_wall_ms','hybrid_vs_jacobi','hybrid_local_regions','no_feasible_arm')


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(('git', *args), cwd=repo, text=True).strip()


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError('unable to load audited source module: '+str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_repo(repo: Path):
    # Explicitly distinguish completed source commit from historical run HEAD.
    if run_git(repo,'rev-parse','HEAD') != POST_HEAD or run_git(repo,'branch','--show-current') != 'develop/0.10.0':
        raise ValueError('post-commit repository identity differs from 5394cb4 develop/0.10.0')
    if run_git(repo,'rev-parse','origin/develop/0.10.0') != POST_HEAD:
        raise ValueError('origin tracking branch does not match 5394cb4; refuse mixed provenance')
    if run_git(repo,'rev-parse','HEAD^') != SOURCE_HEAD:
        raise ValueError('unexpected C1-G5G parent commit')
    subprocess.run(('sha256sum','--check','--strict','--quiet','SOURCE_SHA256.txt'),cwd=repo,check=True)
    sha_rows={}
    for line in (repo/'SOURCE_SHA256.txt').read_text().splitlines():
        if not re.fullmatch(r'[0-9a-f]{64}  .+',line):
            raise ValueError('malformed SOURCE_SHA256 entry')
        digest,path=line.split('  ',1)
        if path in sha_rows:raise ValueError('duplicate SOURCE_SHA256 registry path')
        sha_rows[path]=digest
    for key in ('benchmarks/scripts/C1G5G_scalar_four_arm_evidence.py',
                'crates/hybit/examples/c1g5g_scalar_four_arm_probe.rs',
                'benchmarks/evidence/C1G5E_20261114_results.csv',
                'benchmarks/evidence/C1G5E_20261114_summary.json',
                'benchmarks/evidence/C1G5F_20261114_features.csv',
                'benchmarks/scripts/C1G5E_outcome_audit.py'):
        if key not in sha_rows:
            raise ValueError('missing pinned source: '+key)
    if sha_bytes((repo/'benchmarks/evidence/C1G5E_20261114_results.csv').read_bytes())!=EXPECTED_ELASTIC_CSV_SHA:
        raise ValueError('elasticity evidence SHA differs from historical freeze')


def read_exact_csv(raw: bytes, required: tuple[str,...]) -> list[dict[str,str]]:
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8'),newline=''))
    if reader.fieldnames != list(required) or len(set(reader.fieldnames))!=len(required):
        raise ValueError('unexpected or duplicate CSV columns')
    rows=list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('ragged CSV rows')
    return rows


def reconstruct_scalar(repo: Path, scalar: Path, evidence: Path, source):
    """Re-validate historical 48 raw triples; do NOT call source.check_repo (old HEAD)."""
    if source.HEAD != SOURCE_HEAD or source.FAMILY!='scalar_p1_diffusion':
        raise ValueError('unsupported C1-G5G verifier version')
    matrices=source.check_corpus(scalar)
    if not evidence.is_dir():raise ValueError('scalar raw evidence missing')
    manifest=json.loads((evidence/'run_manifest.json').read_text(encoding='utf-8'))
    expected={
        'source_head': SOURCE_HEAD, 'family_id':'scalar_p1_diffusion',
        'matrix_sha256':matrices, 'binary_sha256':source.sha(repo/source.BINARY),
        'rhs_modes':list(source.RHSS), 'candidate_arms':list(source.ARMS),
        'training_allowed':False, 'production_promotion_allowed':False,
    }
    for k,v in expected.items():
        if manifest.get(k)!=v: raise ValueError('scalar run manifest mismatch: '+k)
    keys=source.keys()
    if len(keys)!=48 or len(set(keys))!=48:
        raise ValueError('scalar verifier is not a 48-key grid')
    expected_files={'run_manifest.json','results.csv','summary.json'}
    for case,rhs,arm in keys:
        expected_files.update(f'{case}_{rhs}_{arm}{ext}' for ext in ('.log','.time','.rc'))
    if {p.name for p in evidence.iterdir()}!=expected_files:
        raise ValueError('raw evidence file set differs from exact 147-file contract')
    rows=[];rhs_by_key={}
    for case,rhs,arm in keys:
        stem=f'{case}_{rhs}_{arm}'
        log=evidence/(stem+'.log'); t=evidence/(stem+'.time'); rcfile=evidence/(stem+'.rc')
        if not all(p.is_file() and p.stat().st_size>0 for p in (log,t,rcfile)):
            raise ValueError('missing or empty log/time/rc: '+stem)
        rc=source.uinteger(rcfile.read_text().strip(),'process_exit_status')
        timing=source.parse_time(t.read_text())
        if int(timing['exit_status'])!=rc:
            raise ValueError('GNU time exit status disagrees: '+stem)
        if rc!=0:
            raise ValueError('original C1-G5G owner run had zero process errors; unexpected process failure: '+stem)
        data=source.parse_log(log.read_text(),case,rhs,arm)
        rhs_key=(case,rhs)
        if rhs_key in rhs_by_key and rhs_by_key[rhs_key]!=data['rhs_hash']:
            raise ValueError('RHS fingerprint differs across candidates: '+str(rhs_key))
        rhs_by_key[rhs_key]=data['rhs_hash']
        row={k:'' for k in source.COLUMNS}
        row.update(case=case,family=source.FAMILY,rhs_mode=rhs,arm=arm,matrix_sha256=matrices[case],
                   process_exit_status=str(rc),record_status='MEASURED',
                   elapsed_s=timing['elapsed_s'],peak_rss_kib=timing['peak_rss_kib'])
        row.update(data)
        rows.append(row)
    groups=[]
    for case in source.MATRICES:
        for rhs in source.RHSS:
            g=[r for r in rows if (r['case'],r['rhs_mode'])==(case,rhs)]
            valid=[r for r in g if r['status']=='Converged']
            fastest=min(valid,key=lambda r:float(r['inprocess_total_wall_ms'])) if valid else None
            groups.append({'case':case,'rhs_mode':rhs,'complete':True,
                           'converged_arms':[r['arm'] for r in valid],
                           'nominal_fastest_converged':fastest['arm'] if fastest else None,
                           'blocked_for_model_training':True})
    converged=sum(r['status']=='Converged' for r in rows)
    summary={'checkpoint':'C1-G5G scalar matched four-arm research','source_head':SOURCE_HEAD,
             'family_id':source.FAMILY,'matrices':3,'rhs_modes':4,'candidate_arms':4,'processes':48,
             'measured':48,'process_errors':0,'converged':converged,
             'unconverged_measured':48-converged,'complete_groups':12,
             'families_with_matched_four_arm_evidence_if_complete':2,
             'production_promotion_allowed':False,'trained_selector':False,
             'limitations':'Two constructed families; one candidate run per RHS; no physical external FEM; different DOFs across families; no held-out model or timing repeat; native Multilevel remains research-only; failures retained.',
             'groups':groups}
    expected_output=source.outputs(rows,summary)
    for name,value in expected_output.items():
        if (evidence/name).read_bytes()!=value:
            raise ValueError('frozen C1-G5G '+name+' differs from independently reconstructed raw evidence')
    if (converged,48-converged)!=(30,18):
        raise ValueError('scalar 30/18 measured checkpoint differs from owner acceptance')
    return rows,summary,sha_bytes(expected_output['results.csv']),len(expected_files)


def compare_rows(rows: list[dict], cases: tuple[str,...], rhs_list:tuple[str,...], arms:tuple[str,...]):
    if len(rows)!=len(cases)*len(rhs_list)*len(arms):raise ValueError('wrong matched-grid length')
    grid={}
    fingerprints={}
    for r in rows:
        key=(r['case'],r['rhs_mode'],r['arm'])
        if any(k not in keys for k,keys in zip(key,(cases,rhs_list,arms))) or key in grid:
            raise ValueError('invalid or duplicate matched key: '+repr(key))
        if r['record_status']!='MEASURED' or r['process_exit_status']!='0':
            raise ValueError('unmeasured candidate cannot join comparisons')
        if r['status'] not in ('Converged','MaxIterations','Breakdown'):
            raise ValueError('unrecognized numerical status')
        if r['status']=='Converged':
            true=float(r['verified_true_relative']); kahan=float(r['compensated_true_relative']);tol=float(r['tol'])
            if not all(math.isfinite(x) and x>=0 for x in (true,kahan,tol)) or tol<=0 or max(true,kahan)>tol*(1+1e-12):
                raise ValueError('false original CSR / Kahan convergence: '+repr(key))
        fp=(key[0],key[1]);h=r['rhs_hash']
        if fp in fingerprints and fingerprints[fp]!=h:raise ValueError('RHS mismatch')
        fingerprints[fp]=h
        cost=float(r['inprocess_total_wall_ms'])
        if not math.isfinite(cost) or cost<0:raise ValueError('invalid cost')
        grid[key]=r
    groups=[]
    for case in cases:
        for rhs in rhs_list:
            arr=[grid[(case,rhs,a)] for a in arms]
            valid=[r for r in arr if r['status']=='Converged']
            winner=min(valid,key=lambda r:(float(r['inprocess_total_wall_ms']),arms.index(r['arm']))) if valid else None
            j,h=arr[0],arr[1]
            jok,hok=(j['status']=='Converged'),(h['status']=='Converged')
            hybrid=('HYBRID_REGRESSION' if jok and not hok else
                    'HYBRID_RESCUE' if hok and not jok else
                    'BOTH_CONVERGED' if jok else 'BOTH_NOT_CONVERGED')
            groups.append({'case':case,'rhs_mode':rhs,'converged_arms':[r['arm'] for r in valid],
                           'fastest_converged':winner['arm'] if winner else None,
                           'fastest_wall_ms':float(winner['inprocess_total_wall_ms']) if winner else None,
                           'hybrid_vs_jacobi':hybrid,'hybrid_local_regions':int(h['local_regions']),
                           'no_feasible_arm':not bool(valid)})
    return groups


def median(values):
    values=sorted(values)
    if not values:return None
    mid=len(values)//2
    return values[mid] if len(values)%2 else 0.5*(values[mid-1]+values[mid])


def arm_stats(rows):
    stats=[]
    for arm in ARMS:
        a=[r for r in rows if r['arm']==arm]
        valid=[r for r in a if r['status']=='Converged']
        if len(a)!=12:raise ValueError('expected 12 results per candidate')
        stats.append({'arm':arm,'converged':len(valid),'unconverged':12-len(valid),
                      'hybrid_factors_activated':sum(int(r['local_regions'])>0 for r in a),
                      'median_converged_wall_ms':median([float(r['inprocess_total_wall_ms']) for r in valid]),
                      'median_converged_iterations':median([int(r['iterations']) for r in valid]),
                      'median_converged_peak_rss_kib':median([int(r['peak_rss_kib']) for r in valid])})
    return stats


def analysis(repo:Path, scalar:Path, evidence:Path):
    strict_repo(repo)
    source=import_module(repo/'benchmarks/scripts/C1G5G_scalar_four_arm_evidence.py','c1g5g_original_evidence')
    scalar_rows,scalar_summary,raw_csv_hash,raw_count=reconstruct_scalar(repo,scalar,evidence,source)
    g5e=import_module(repo/'benchmarks/scripts/C1G5E_outcome_audit.py','c1g5e_original_outcome')
    g5e_csv=(repo/'benchmarks/evidence/C1G5E_20261114_results.csv').read_bytes()
    elastic_rows=g5e._read_csv(repo/'benchmarks/evidence/C1G5E_20261114_results.csv')
    elastic_summary=json.loads((repo/'benchmarks/evidence/C1G5E_20261114_summary.json').read_text())
    elastic_report=g5e.analyze(elastic_rows,elastic_summary,sha_bytes(g5e_csv))
    scalar_groups=compare_rows(scalar_rows,tuple(source.MATRICES),tuple(source.RHSS),tuple(source.ARMS))
    if scalar_summary['converged']!=sum(a['converged'] for a in arm_stats(scalar_rows)):
        raise ValueError('scalar summary versus audit disagreement')
    elastic_groups=compare_rows(elastic_rows,tuple(g5e.CASES),tuple(g5e.RHS),tuple(g5e.ARMS))
    if elastic_report['converged']!=sum(a['converged'] for a in arm_stats(elastic_rows)):
        raise ValueError('elastic summary versus audit disagreement')
    feats=read_exact_csv((repo/'benchmarks/evidence/C1G5F_20261114_features.csv').read_bytes(),
                         ('case','family','matrix_sha256','log_sha256','rows','nnz','offdiag','nondominant_rows','nondominant_fraction','positive_fraction','diagonal_log10_spread','max_offdiag_ratio','strong_fraction','rhs_nonzero_fraction','invalid_ratio','audit_ms','shadow_reason','research_rank','production_recommended'))
    if len(feats)!=6 or len({f['case'] for f in feats})!=6:
        raise ValueError('six matrix features not complete')
    matrix_info={}
    for f in feats:
        matrix_info[f['case']]={'family':f['family'], 'diagonal_log10_spread':float(f['diagonal_log10_spread']),
                                'nondominant_fraction':float(f['nondominant_fraction']),
                                'strong_fraction':float(f['strong_fraction']),
                                'positive_fraction':float(f['positive_fraction'])}
        if any(not math.isfinite(x) for x in matrix_info[f['case']].values() if isinstance(x,float)):
            raise ValueError('nonfinite frozen features')
    for r in scalar_rows+elastic_rows:
        if r['case'] not in matrix_info or r['matrix_sha256']!=next(f['matrix_sha256'] for f in feats if f['case']==r['case']):
            raise ValueError('feature matrix fingerprint differs from benchmark')
    report={'schema':'C1G5G_POSTCOMMIT_OUTCOME_AUDIT_V1','analysis_commit':POST_HEAD,
            'scalar_experiment_source_head':SOURCE_HEAD,'scalar_raw_files_checked':raw_count,
            'scalar_results_sha256':raw_csv_hash,'elasticity_results_sha256':sha_bytes(g5e_csv),
            'scopes':{'scalar':{'family':'scalar_p1_diffusion','dof':20736,'cases':3,'groups':scalar_groups,
                                 'per_arm':arm_stats(scalar_rows),'converged':scalar_summary['converged'],
                                 'unconverged':scalar_summary['unconverged_measured']},
                      'elasticity':{'family':'tri_p1_plane_strain_elasticity','dof':3280,'cases':3,'groups':elastic_groups,
                                    'per_arm':arm_stats(elastic_rows),'converged':elastic_report['converged'],
                                    'unconverged':elastic_report['unconverged']}},
            'matrix_features':matrix_info,
            'training_performed':False,'production_promotion_allowed':False,
            'limitations':('Two constructed generator families, no external FEM, no held-out selection model. '
                'One measurement per candidate, no timing significance. RHS sets share names but problem scales differ. '
                'Do not compare wall-time medians across families or treat 24 groups as independent heldout datasets. '
                'The C1G5F features use b=A*ones rather than the four benchmark RHS. '
                'Native Multilevel is research-only, Hybrid setup_ms overlaps solve_ms. '
                'No measured cost should be interpreted as a production-routing guarantee.')}
    return report


def render(report):
    out=['# C1-G5G post-commit matched four-arm outcome audit', '',
         f"**Provenance:** experimental HEAD `{SOURCE_HEAD[:7]}`, current commit `{POST_HEAD[:7]}`; no sources changed.",
         '**State:** descriptive comparisons only; training and production routing BLOCKED.', '']
    for family in ('scalar','elasticity'):
        scope=report['scopes'][family]
        out+=[f"## {family.title()} — {scope['dof']:,} DOF; {scope['converged']}/48 converged",'',
              '| Arm | Converged/12 | Non-converged/12 | Hybrid factors used | Median wall ms (converged only) | Median iterations (converged only) |',
              '|---|---:|---:|---:|---:|---:|']
        for a in scope['per_arm']:
            fmt=lambda v: 'n/a' if v is None else f'{v:.3f}'
            out.append(f"| `{a['arm']}` | {a['converged']} | {a['unconverged']} | {a['hybrid_factors_activated']} | {fmt(a['median_converged_wall_ms'])} | {fmt(a['median_converged_iterations'])} |")
        out+=['','| Matrix | RHS | J | H | T | M | Fastest converged (single run) | Hybrid vs Jacobi |',
              '|---|---|:---:|:---:|:---:|:---:|---|---|']
        for g in scope['groups']:
            winners=[('✓' if a in g['converged_arms'] else '—') for a in ARMS]
            winner=(f"{g['fastest_converged']} ({g['fastest_wall_ms']:.3f} ms)" if g['fastest_converged'] else 'NO_FEASIBLE_ARM')
            out.append('| '+' | '.join((g['case'],g['rhs_mode'],*winners,winner,g['hybrid_vs_jacobi']))+' |')
        out+=['','*Single run per candidate; end-to-end in-process time within that family, among converged candidates only.*','']
    out+=['## Cross-family observations — descriptive only','',
          '| Matrix | Family | DOF | Diagonal spread (log10) | Nondominant rows fraction | Strong link fraction | J/H/T/M converged RHS (of 4) |',
          '|---|---|---:|---:|---:|---:|---|']
    for family in ('scalar','elasticity'):
        scope=report['scopes'][family]
        for case in [g['case'] for g in scope['groups'][::4]]:
            f=report['matrix_features'][case]
            counts=[sum(a in g['converged_arms'] for g in scope['groups'] if g['case']==case) for a in ARMS]
            out.append(f"| {case} | {family} | {scope['dof']} | {f['diagonal_log10_spread']:.3f} | {f['nondominant_fraction']:.3f} | {f['strong_fraction']:.3f} | {'/'.join(map(str,counts))} |")
    out+=['','## Restrictions','',
          '- A process error, Breakdown or MaxIterations is never treated as a successful fastest candidate.',
          '- Relative residuals must pass original CSR and compensated checks; point-load does not invent a reference solution.',
          '- Original scalar logs and time/rc records were revalidated after commit; the historical run HEAD is not rewritten.',
          '- G5F features were measured for `b=A*ones`, not the four benchmark RHS. They are descriptive case properties only.',
          '- Two constructed FEM families are **not** independent held-out model qualification; no external physical cases or timing repetitions.',
          '- Across-family wall-time comparison is invalid: DOF and workloads differ; within-family timing is one-shot.',
          '- This audit does not fit coefficients, alter Rust/FFI, or promote any solver.', '']
    return '\n'.join(out)


def audit_csv(report):
    buff=io.StringIO(newline='')
    writer=csv.DictWriter(buff,fieldnames=('family',*OUT_COLS),lineterminator='\n')
    writer.writeheader()
    for family in ('scalar','elasticity'):
        for g in report['scopes'][family]['groups']:
            r={'family':family,'case':g['case'],'rhs_mode':g['rhs_mode'],
               'fastest_converged':g['fastest_converged'] or 'NO_FEASIBLE_ARM',
               'fastest_wall_ms':'' if g['fastest_wall_ms'] is None else f"{g['fastest_wall_ms']:.6f}",
               'hybrid_vs_jacobi':g['hybrid_vs_jacobi'],
               'hybrid_local_regions':g['hybrid_local_regions'],
               'no_feasible_arm':str(g['no_feasible_arm']).lower()}
            r.update({field:str(arm in g['converged_arms']).lower() for field,arm in zip(('jacobi','hybrid','two_level','multilevel'),ARMS)})
            writer.writerow(r)
    return buff.getvalue()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--scalar',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,help='new directory; existing directories will not be modified')
    a=p.parse_args()
    try:
        result=analysis(a.repo.resolve(strict=True),a.scalar.resolve(strict=True),a.evidence.resolve(strict=True))
        files={'outcome_audit.json':json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',
               'outcome_audit.csv':audit_csv(result),'outcome_audit.md':render(result)}
        print(files['outcome_audit.md'])
        if a.out is not None:
            target=a.out.expanduser().absolute()
            if target.exists():raise ValueError('output destination exists; refusing overwrite')
            target.mkdir(parents=False)
            for name,content in files.items():
                (target/name).write_text(content,encoding='utf-8')
            print('C1-G5G POSTCOMMIT OUTCOME FILES:',str(target))
        print('C1-G5G POSTCOMMIT OUTCOME AUDIT PASS: raw scalar evidence and frozen elasticity checked; no production/model changes')
    except Exception as exc:
        p.exit(1,'C1-G5G POSTCOMMIT OUTCOME AUDIT FAIL: '+str(exc)+'\n')


if __name__=='__main__':main()
