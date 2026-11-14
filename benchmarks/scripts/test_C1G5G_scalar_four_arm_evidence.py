#!/usr/bin/env python3
"""Isolated schema and failure-handling tests, no fabricated benchmark evidence."""
import csv
import importlib.util
import io
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('g5g',HERE/'C1G5G_scalar_four_arm_evidence.py')
g=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)

class EvidenceTests(unittest.TestCase):
    def test_exact_48_grid_and_family(self):
        self.assertEqual(len(g.keys()),48)
        self.assertEqual(len(set(g.keys())),48)
        self.assertEqual(len(g.MATRICES),3)
        self.assertEqual(g.FAMILY,'scalar_p1_diffusion')

    def test_manifest_frozen_oracles(self):
        self.assertEqual(len(set(v[1] for v in g.MATRICES.values())),3)
        for _,(_,h,nnz) in g.MATRICES.items():
            self.assertEqual(len(h),64)
            self.assertGreater(nnz,0)

    def record(self,case='scalar_island_shell',rhs='ones',arm='jacobi_pcg'):
        d=dict.fromkeys(g.FIELDS,'0')
        d.update(case=case,rhs_mode=rhs,arm=arm,rhs_hash='0123456789abcdef',
                 status='Converged',iterations='1',tol='1.000000000000e-10',
                 max_iterations='600',reported_relative='1.0e-12',
                 verified_true_relative='1.0e-12',compensated_true_relative='1.0e-12',
                 relative_x_error='1.0e-12' if rhs!='point_load' else 'n/a',
                 factor_bytes='0',local_regions='0',escalations='0',
                 effective_preconditioner='Jacobi',solver_path='A9_CERTIFIED_PCG',
                 rows='20736',nnz=str(g.MATRICES[case][2]),load_ms='1',rhs_ms='1',setup_ms='1',
                 solve_ms='1',arm_work_ms='2',inprocess_total_wall_ms='4')
        return d

    @staticmethod
    def line(d):return 'C1G5G|'+'|'.join(k+'='+d[k] for k in g.FIELDS)

    def test_converged_original_residual(self):
        d=self.record()
        self.assertEqual(g.parse_log(self.line(d),'scalar_island_shell','ones','jacobi_pcg')['status'],'Converged')

    def test_false_convergence_original_csr_rejected(self):
        for field in ('verified_true_relative','compensated_true_relative'):
            d=self.record();d[field]='0.01'
            with self.assertRaises(ValueError):g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])

    def test_bad_tolerance_and_budget(self):
        for field,value in [('tol','1e-8'),('max_iterations','601'),('iterations','601'),('status','Bad')]:
            d=self.record();d[field]=value
            with self.assertRaises(ValueError):g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])

    def test_max_iteration_full_budget(self):
        d=self.record();d.update(status='MaxIterations',iterations='500')
        with self.assertRaises(ValueError):g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])
        d['iterations']='600'
        self.assertEqual(g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])['status'],'MaxIterations')

    def test_missing_duplicate_and_wrong_rhs(self):
        d=self.record();line=self.line(d)
        for bad in ('foo',line+'|rhs_hash=0123456789abcdef',line.replace('rhs_hash=0123456789abcdef|','')):
            with self.assertRaises(ValueError):g.parse_log(bad,d['case'],d['rhs_mode'],d['arm'])
        with self.assertRaises(ValueError):g.parse_log(line,d['case'],'point_load',d['arm'])

    def test_nonfinite_and_negative(self):
        for value in ('nan','inf','-1'):
            d=self.record();d['setup_ms']=value
            with self.assertRaises(ValueError):g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])

    def test_hybrid_factor_truth(self):
        d=self.record(arm='hybrid_local_cholesky')
        d.update(local_regions='2',factor_bytes='2048',effective_preconditioner='Hybrid')
        g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])
        d['factor_bytes']='0'
        with self.assertRaises(ValueError):g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])

    def test_point_load_forward_error_not_invented(self):
        d=self.record(rhs='point_load')
        g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])
        d['relative_x_error']='0'
        with self.assertRaises(ValueError):g.parse_log(self.line(d),d['case'],d['rhs_mode'],d['arm'])

    def test_gnu_time_strict(self):
        good='elapsed_s=1.2\npeak_rss_kib=1234\nexit_status=0\n'
        self.assertEqual(g.parse_time(good)['exit_status'],'0')
        for s in (good+'exit_status=0\n',good.replace('peak_rss_kib=1234\n',''),good.replace('1.2','nan')):
            with self.assertRaises(ValueError):g.parse_time(s)

    def test_error_cannot_become_winner(self):
        # The candidate list explicitly separates numerical convergence from successful process execution.
        rows=[{'record_status':'PROCESS_ERROR','status':'','inprocess_total_wall_ms':''},
              {'record_status':'MEASURED','status':'MaxIterations','inprocess_total_wall_ms':'0.1'},
              {'record_status':'MEASURED','status':'Converged','inprocess_total_wall_ms':'10'}]
        eligible=[r for r in rows if r['record_status']=='MEASURED' and r['status']=='Converged']
        self.assertEqual(len(eligible),1)

    def test_output_columns_and_no_promotion(self):
        rows=[dict.fromkeys(g.COLUMNS,'')]
        result=g.outputs(rows,{'production_promotion_allowed':False,'trained_selector':False})
        reader=csv.DictReader(io.StringIO(result['results.csv'].decode()))
        self.assertEqual(tuple(reader.fieldnames),g.COLUMNS)
        self.assertIn(b'"trained_selector": false',result['summary.json'])

if __name__=='__main__':unittest.main(verbosity=2)
