#!/usr/bin/env python3
"""Negative tests for C1-G5E corpus and audit, no repository mutation."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import C1G5E_elasticity_corpus as c
import C1G5E_evidence as e


def fixture(**overrides):
    d={
        'case':'elastic_uniform','arm':'jacobi_pcg','rhs_mode':'ones',
        'rhs_hash':'0123456789abcdef','status':'Converged','iterations':'17',
        'tol':'1.000000000000e-10','max_iterations':'600',
        'reported_relative':'5.000000000000e-11','verified_true_relative':'5.000000000000e-11',
        'compensated_true_relative':'5.000000000000e-11','relative_x_error':'1.000000000000e-7',
        'factor_bytes':'0','local_regions':'0','escalations':'0',
        'effective_preconditioner':'Jacobi','solver_path':'A9_CERTIFIED_PCG',
        'rows':'3280','nnz':'23958',
        'load_ms':'10.0','rhs_ms':'0.1','setup_ms':'0.2','solve_ms':'2.0',
        'arm_work_ms':'2.4','inprocess_total_wall_ms':'13.0',
    }
    d.update(overrides)
    return 'C1G5E|'+'|'.join(k+'='+v for k,v in d.items())


class CorpusTests(unittest.TestCase):
    def test_family_distinct_from_c1g4(self):
        self.assertIn('elasticity',c.FAMILY)
        self.assertEqual(c.FAMILY,'tri_p1_plane_strain_elasticity_2dof_clamped')
    def test_positive_deterministic_and_distinct(self):
        hashes=set()
        for name in c.CASES:
            a,m=c.assemble(name)
            b,m2=c.assemble(name)
            self.assertEqual(a,b)
            self.assertEqual(m,m2)
            self.assertEqual(m['ndof'],3280)
            self.assertEqual(m['fixed_nodes'],41)
            self.assertGreater(m['min_double_area'],0)
            self.assertGreater(m['min_young'],0)
            hashes.add(m['matrix_sha256'])
        self.assertEqual(len(hashes),3)
    def test_write_once_and_detect_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            c.generate(p,False)
            c.generate(p,True)
            victim=p/'elastic_uniform.mtx'
            victim.write_bytes(victim.read_bytes()+b'% tampered\n')
            with self.assertRaises(ValueError):c.generate(p,True)
            with self.assertRaises(ValueError):c.generate(p,False)
    def test_bad_material_and_case(self):
        for e0,nu in ((0,0.3),(-1,0.3),(1,0.5),(1,-1),(float('nan'),.3)):
            with self.assertRaises(ValueError):c.elasticity_tensor(e0,nu)
        with self.assertRaises(ValueError):c.assemble('nonexistent')
    def test_symmetric_coordinate_and_stiffness(self):
        data,meta=c.assemble('elastic_uniform')
        lines=data.decode().splitlines()
        self.assertEqual(lines[0],'%%MatrixMarket matrix coordinate real symmetric')
        body=[x for x in lines if x and not x.startswith('%')]
        size=list(map(int,body[0].split()))
        self.assertEqual(size[:2],[3280,3280])
        self.assertEqual(len(body)-1,size[2])
        diagonal=[]
        for line in body[1:]:
            i,j,v=line.split();i=int(i);j=int(j)
            self.assertLessEqual(i,j)
            if i==j:diagonal.append(float(v))
        self.assertEqual(len(diagonal),3280)
        self.assertTrue(all(x>0 for x in diagonal))


class AuditTests(unittest.TestCase):
    def test_reference_valid(self):
        row=e.parse_log(fixture(),case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
        self.assertEqual(row['status'],'Converged')
    def test_false_convergence(self):
        for k in ('compensated_true_relative','verified_true_relative'):
            with self.assertRaises(ValueError):e.parse_log(fixture(**{k:'1.5e-10'}),
                case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
    def test_maxiterations_budget(self):
        with self.assertRaises(ValueError):e.parse_log(
            fixture(status='MaxIterations',iterations='500'),
            case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
    def test_no_invented_forward_error(self):
        with self.assertRaises(ValueError):e.parse_log(fixture(rhs_mode='point_load'),
            case='elastic_uniform',rhs='point_load',arm='jacobi_pcg')
    def test_no_missing_forward_error(self):
        with self.assertRaises(ValueError):e.parse_log(fixture(relative_x_error='n/a'),
            case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
    def test_no_duplicate_log_or_field(self):
        log=fixture()
        with self.assertRaises(ValueError):e.parse_log(log+'\n'+log,
            case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
        with self.assertRaises(ValueError):e.parse_log(log+'|tol=1e-10',
            case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
    def test_missing_field(self):
        s=fixture().replace('|rhs_hash=0123456789abcdef','')
        with self.assertRaises(ValueError):e.parse_log(s,
            case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
    def test_hybrid_factor_requires_actual_regions_and_bytes(self):
        for kw in (dict(arm='hybrid_local_cholesky',effective_preconditioner='Hybrid'),
                   dict(arm='hybrid_local_cholesky',local_regions='8',factor_bytes='0',effective_preconditioner='Hybrid'),
                   dict(arm='hybrid_local_cholesky',local_regions='8',factor_bytes=str(65*1024**2),effective_preconditioner='Hybrid')):
            with self.assertRaises(ValueError):e.parse_log(fixture(**kw),
                case='elastic_uniform',rhs='ones',arm='hybrid_local_cholesky')
    def test_bad_tolerance_iteration_nan(self):
        for kw in (dict(tol='1e-8'),dict(iterations='601'),dict(solve_ms='nan'),
                   dict(verified_true_relative='-1e-10')):
            with self.assertRaises(ValueError):e.parse_log(fixture(**kw),
                case='elastic_uniform',rhs='ones',arm='jacobi_pcg')
    def test_gnu_time_duplicate_and_missing(self):
        t='elapsed_s=0.1\npeak_rss_kib=3000\nexit_status=0\n'
        self.assertEqual(e.parse_time(t)['exit_status'],'0')
        for bad in (t+'elapsed_s=0.2\n','elapsed_s=0.1\n',t.replace('peak_rss_kib=3000','peak_rss_kib=-1')):
            with self.assertRaises(ValueError):e.parse_time(bad)
    def test_grid_and_schema(self):
        keys=e.expected_keys()
        self.assertEqual(len(keys),48)
        self.assertEqual(len({x[:3] for x in keys}),48)
        self.assertEqual(len(e.COLUMNS),30)
    def test_wrong_case_and_arm(self):
        with self.assertRaises(ValueError):e.parse_log(fixture(),
            case='elastic_uniform',rhs='ones',arm='multilevel_graph_piecewise')

if __name__=='__main__':unittest.main()
