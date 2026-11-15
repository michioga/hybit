#!/usr/bin/env python3
"""C1-G5H fail-closed unit tests: synthetic records are NOT benchmark evidence."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest


def load():
    f=Path(__file__).with_name('C1G5H_tplus_h_evidence.py')
    spec=importlib.util.spec_from_file_location('c1g5h_tests_src',f)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

a=load()


def sample(arm='tplus_h_static',status='Converged'):
    raw={
     'case':'scalar_island_shell','arm':arm,'rhs_mode':'ones','rhs_hash':'0123456789abcdef',
     'status':status,'iterations':'12' if status=='Converged' else '600',
     'tol':'1.000000000000e-10','max_iterations':'600',
     'reported_relative':'9.0e-11','verified_true_relative':'9.0e-11',
     'compensated_true_relative':'9.0e-11','relative_x_error':'1.0e-9',
     'factor_bytes':'250','local_regions':'8' if arm=='tplus_h_static' else '0',
     'escalations':'0','effective_preconditioner':'AlgebraicTwoLevelHybridStatic' if arm=='tplus_h_static' else 'TwoLevelOnlyNoLocal',
     'solver_path':'C1G5H_STATIC_RESEARCH_COMPOSITE' if arm=='tplus_h_static' else 'HYBIT_AUTO_ALGEBRAIC_COARSE','rows':'20736','nnz':'103104',
     'load_ms':'1.2','rhs_ms':'0.2','setup_ms':'2.1','solve_ms':'2.0',
     'arm_work_ms':'4.2','inprocess_total_wall_ms':'5.5',
     'coarse_bytes':'200','local_bytes':'50' if arm=='tplus_h_static' else '0',
     'local_used':'true' if arm=='tplus_h_static' else 'false',
    }
    if arm=='tplus_h_adaptive':raw['factor_bytes']='200'
    line='C1G5H|'+ '|'.join(f'{k}={v}' for k,v in raw.items())
    baseline={'rhs_hash':'0123456789abcdef'}
    return raw,line,baseline


class EvidenceTests(unittest.TestCase):
    def parse(self,arm='tplus_h_static',status='Converged'):
        _,ln,b=sample(arm,status)
        return a.parse_log(ln,'scalar_island_shell','ones',arm,20736,b)
    def test_matrix_grid_exact(self):
        self.assertEqual(len(a.MATRICES),6)
        self.assertEqual(len(a.keys()),48)
        self.assertEqual(len(set(a.keys())),48)
        self.assertEqual(len(a.COLS),len(set(a.COLS)))
    def test_static_good(self):
        self.assertEqual(self.parse()['local_used'],'true')
    def test_adaptive_coarse_only_good(self):
        self.assertEqual(self.parse('tplus_h_adaptive')['local_used'],'false')
    def test_nonconverged_retained(self):
        self.assertEqual(self.parse(status='MaxIterations')['status'],'MaxIterations')
    def test_false_convergence_blocked(self):
        raw,_,b=sample();raw['compensated_true_relative']='3e-4'
        ln='C1G5H|'+'|'.join(f'{k}={v}' for k,v in raw.items())
        with self.assertRaisesRegex(ValueError,'false original-CSR'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_wrong_rhs_hash_blocked(self):
        _,ln,b=sample();b['rhs_hash']='deadbeefdeadbeef'
        with self.assertRaisesRegex(ValueError,'RHS'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_missing_field_blocked(self):
        _,ln,b=sample();ln=ln.replace('|coarse_bytes=200','')
        with self.assertRaisesRegex(ValueError,'field set'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_duplicate_field_blocked(self):
        _,ln,b=sample();ln+='|case=scalar_island_shell'
        with self.assertRaisesRegex(ValueError,'duplicate'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_negative_nonfinite_blocked(self):
        _,ln,b=sample();ln=ln.replace('setup_ms=2.1','setup_ms=nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_unused_static_factor_blocked(self):
        _,ln,b=sample();ln=ln.replace('local_used=true','local_used=false')
        with self.assertRaisesRegex(ValueError,'fake local'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_fake_adaptive_escalation_blocked(self):
        _,ln,b=sample('tplus_h_adaptive');ln=ln.replace('escalations=0','escalations=1')
        with self.assertRaisesRegex(ValueError,'adaptive escalation'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_adaptive',20736,b)
    def test_memory_budget_blocked(self):
        _,ln,b=sample();ln=ln.replace('local_bytes=50','local_bytes=67108865').replace('factor_bytes=250','factor_bytes=67109065')
        with self.assertRaisesRegex(ValueError,'memory policy'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_wrong_dimension_blocked(self):
        _,ln,b=sample();ln=ln.replace('rows=20736','rows=3280')
        with self.assertRaisesRegex(ValueError,'geometry'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_point_load_reference_blocked(self):
        _,ln,b=sample();ln=ln.replace('rhs_mode=ones','rhs_mode=point_load')
        with self.assertRaisesRegex(ValueError,'invented'):a.parse_log(ln,'scalar_island_shell','point_load','tplus_h_static',20736,b)
    def test_time_format(self):
        t=a.parse_time('elapsed_s=0.12\npeak_rss_kib=12345\nexit_status=0\n')
        self.assertEqual(t['peak_rss_kib'],'12345')
        with self.assertRaisesRegex(ValueError,'GNU time'):a.parse_time('elapsed_s=1\n')
    def test_breakdown_and_budget(self):
        self.assertEqual(self.parse(status='Breakdown')['status'],'Breakdown')
        raw,ln,b=sample(status='MaxIterations')
        ln=ln.replace('iterations=600','iterations=5')
        with self.assertRaisesRegex(ValueError,'budget'):a.parse_log(ln,'scalar_island_shell','ones','tplus_h_static',20736,b)
    def test_full_48_grid_reconstruction_and_failed_candidate_retention(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';ev=root/'evidence';repo.mkdir();ev.mkdir()
            binary=repo/a.BINARY;binary.parent.mkdir(parents=True)
            binary.write_bytes(b'fixed synthetic binary for test, not real FEM')
            manifest={'source_head':a.HEAD,'binary_sha256':a.sha(binary),
                'matrix_sha256':{c:d for c,_,_,d,_ in a.MATRICES},
                'arms':list(a.ARMS),'rhs_modes':list(a.MODES),
                'production_promotion_allowed':False}
            (ev/'run_manifest.json').write_text(json.dumps(manifest))
            baselines={};original={};paths={}
            for case,fam,name,digest,n in a.MATRICES:
                nnz='103104' if fam=='scalar' else '44546'
                for rhs in a.MODES:
                    baselines[(case,rhs)]={'rhs_hash':'0123456789abcdef',
                        'status':'Converged','rows':str(n),'nnz':nnz,
                        'inprocess_total_wall_ms':'3.0'}
                    for arm in a.ARMS:
                        raw,_,b=sample(arm)
                        raw.update(case=case,rhs_mode=rhs,rows=str(n),nnz=nnz)
                        if rhs=='point_load':raw['relative_x_error']='n/a'
                        if arm=='tplus_h_adaptive':raw['factor_bytes']='200'
                        line='C1G5H|'+'|'.join(f'{k}={v}' for k,v in raw.items())+'\n'
                        stem=f'{case}_{rhs}_{arm}'
                        (ev/(stem+'.log')).write_text(line)
                        (ev/(stem+'.time')).write_text('elapsed_s=0.01\npeak_rss_kib=5000\nexit_status=0\n')
                        (ev/(stem+'.rc')).write_text('0\n')
            (ev/'results.csv').write_bytes(b'placeholder')
            (ev/'summary.json').write_bytes(b'{}')
            csv_bytes,json_bytes,summary=a.reconstruct(repo,ev,baselines,paths)
            self.assertEqual((summary['converged'],summary['processes'],summary['local_used']),(48,48,24))
            self.assertFalse(summary['production_promotion_allowed'])
            (ev/'results.csv').write_bytes(csv_bytes);(ev/'summary.json').write_bytes(json_bytes)
            self.assertEqual(a.reconstruct(repo,ev,baselines,paths)[:2],(csv_bytes,json_bytes))
            failing='elastic_uniform_ones_tplus_h_static'
            (ev/(failing+'.rc')).write_text('1\n')
            (ev/(failing+'.time')).write_text('elapsed_s=0.01\npeak_rss_kib=5000\nexit_status=1\n')
            _,_,partial=a.reconstruct(repo,ev,baselines,paths)
            self.assertEqual((partial['converged'],partial['process_errors']),(47,1))
            self.assertEqual(sum(r['outcome']=='PROCESS_ERROR' for r in partial['results']),1)
            (ev/(failing+'.time')).write_text('elapsed_s=0.01\npeak_rss_kib=5000\nexit_status=0\n')
            with self.assertRaisesRegex(ValueError,'rc disagrees'):
                a.reconstruct(repo,ev,baselines,paths)

    def test_production_blocked(self):
        self.assertNotIn('production',a.ARMS)
        self.assertIn('production_promotion_allowed',a.reconstruct.__code__.co_consts.__repr__())

if __name__=='__main__':unittest.main()
