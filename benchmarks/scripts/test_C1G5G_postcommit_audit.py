#!/usr/bin/env python3
"""Synthetic fixtures for post-commit audit contracts; never claimed as real measurements."""
from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('postcommit',HERE/'C1G5G_postcommit_audit.py')
a=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
SRC=HERE/'C1G5G_scalar_four_arm_evidence.py'


def fixture_rows():
    result=[]
    cases=('c0','c1','c2')
    rhs=('ones','index_smooth','seeded_random','point_load')
    for c in cases:
        for b in rhs:
            for i,arm in enumerate(a.ARMS):
                converged=i in (0,3) if c=='c0' else i==3
                result.append({'case':c,'rhs_mode':b,'arm':arm,
                    'record_status':'MEASURED','process_exit_status':'0','rhs_hash':'0123456789abcdef',
                    'status':'Converged' if converged else 'MaxIterations',
                    'verified_true_relative':'1e-12' if converged else '1e-2',
                    'compensated_true_relative':'1e-12' if converged else '1e-2',
                    'tol':'1e-10','inprocess_total_wall_ms':str(50-i*10),
                    'iterations':'25' if converged else '600',
                    'local_regions':'2' if i==1 else '0','peak_rss_kib':'1000'})
    return result,cases,rhs


class AuditTests(unittest.TestCase):
    def test_full_matched_grid_and_nominal_winner(self):
        rows,cases,rhs=fixture_rows()
        groups=a.compare_rows(rows,cases,rhs,a.ARMS)
        self.assertEqual(len(groups),12)
        self.assertEqual(groups[0]['fastest_converged'],a.ARMS[3])
        self.assertEqual(groups[0]['hybrid_vs_jacobi'],'HYBRID_REGRESSION')
        self.assertEqual(groups[4]['hybrid_vs_jacobi'],'BOTH_NOT_CONVERGED')

    def test_false_convergence_rejected(self):
        rows,cases,rhs=fixture_rows();rows[0]['verified_true_relative']='1e-4'
        with self.assertRaisesRegex(ValueError,'false'):
            a.compare_rows(rows,cases,rhs,a.ARMS)

    def test_kahan_false_convergence_rejected(self):
        rows,cases,rhs=fixture_rows();rows[0]['compensated_true_relative']='5e-4'
        with self.assertRaisesRegex(ValueError,'false'):
            a.compare_rows(rows,cases,rhs,a.ARMS)

    def test_duplicate_key_rejected(self):
        rows,cases,rhs=fixture_rows();rows[1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            a.compare_rows(rows,cases,rhs,a.ARMS)

    def test_missing_candidate_rejected(self):
        rows,cases,rhs=fixture_rows()
        with self.assertRaisesRegex(ValueError,'length'):
            a.compare_rows(rows[:-1],cases,rhs,a.ARMS)

    def test_rhs_fingerprint_mismatch_rejected(self):
        rows,cases,rhs=fixture_rows();rows[1]['rhs_hash']='ffffffffffffffff'
        with self.assertRaisesRegex(ValueError,'RHS mismatch'):
            a.compare_rows(rows,cases,rhs,a.ARMS)

    def test_process_error_rejected(self):
        rows,cases,rhs=fixture_rows();rows[1]['record_status']='PROCESS_ERROR'
        with self.assertRaisesRegex(ValueError,'unmeasured'):
            a.compare_rows(rows,cases,rhs,a.ARMS)

    def test_bad_wall_time_rejected(self):
        rows,cases,rhs=fixture_rows()
        for bad in ('-1','nan','inf'):
            rows[0]['inprocess_total_wall_ms']=bad
            with self.assertRaisesRegex(ValueError,'invalid cost'):
                a.compare_rows(rows,cases,rhs,a.ARMS)

    def test_no_feasible_candidate_explicit(self):
        rows,cases,rhs=fixture_rows()
        for r in rows:
            if (r['case'],r['rhs_mode'])==('c0','ones'):
                r.update(status='MaxIterations',iterations='600')
        groups=a.compare_rows(rows,cases,rhs,a.ARMS)
        self.assertTrue(groups[0]['no_feasible_arm'])
        self.assertIsNone(groups[0]['fastest_converged'])

    def test_medians_are_convergence_conditional(self):
        rows,_,_=fixture_rows()
        result=a.arm_stats(rows)
        self.assertEqual([(x['arm'],x['converged']) for x in result],[(a.ARMS[0],4),(a.ARMS[1],0),(a.ARMS[2],0),(a.ARMS[3],12)])
        self.assertIsNone(result[1]['median_converged_wall_ms'])
        self.assertEqual(a.median([1,2]),1.5)

    def test_bad_csv_headers_rejected(self):
        cols=('a','b')
        with self.assertRaisesRegex(ValueError,'columns'):
            a.read_exact_csv(b'a,a\nx,y\n',cols)
        with self.assertRaisesRegex(ValueError,'ragged'):
            a.read_exact_csv(b'a,b\nx\n',cols)

    def test_audit_outputs_production_blocked(self):
        rows,cases,rhs=fixture_rows()
        group=a.compare_rows(rows,cases,rhs,a.ARMS)
        scope={'dof':100,'groups':group,'converged':16,'unconverged':32,'per_arm':a.arm_stats(rows)}
        report={'scopes':{'scalar':scope,'elasticity':scope},'matrix_features':{c:{'family':'fake','diagonal_log10_spread':1.0,'nondominant_fraction':0.0,'strong_fraction':0.5} for c in cases},'training_performed':False,'production_promotion_allowed':False}
        output=a.audit_csv(report)
        parsed=list(csv.DictReader(io.StringIO(output)))
        self.assertEqual(len(parsed),24)
        rendered=a.render(report)
        self.assertIn('Cross-family observations',rendered)
        self.assertIn('training and production routing BLOCKED',rendered)
        self.assertEqual(parsed[0]['fastest_converged'],a.ARMS[3])
        self.assertEqual(parsed[0]['jacobi'],'true')
        self.assertFalse(report['production_promotion_allowed'])

    def test_historical_vs_current_head_are_different(self):
        self.assertNotEqual(a.POST_HEAD,a.SOURCE_HEAD)
        self.assertEqual(len(a.POST_HEAD),40)

    def test_untrusted_feature_does_not_add_model_promotion(self):
        src=Path(a.__file__).read_text()
        self.assertIn("'production_promotion_allowed':False",src)
        self.assertIn("'training_performed':False",src)
        self.assertNotIn('model.fit(',src)


if __name__=='__main__':unittest.main(verbosity=2)
