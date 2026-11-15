#!/usr/bin/env python3
"""Synthetic fixture tests only; not real numerical evidence."""
from __future__ import annotations
from copy import deepcopy
import unittest
import C1G5H_outcome_review as r


def fixture():
    base={};rows=[]
    for case in r.CASES:
        for k,mode in enumerate(r.MODES):
            rhs='abcd'+format(r.CASES.index(case)*4+k,'012x')
            t_ok=k!=0
            b={'case':case,'rhs_mode':mode,'status':'Converged' if t_ok else 'MaxIterations',
              'matrix_sha256':'deadbeef','rhs_hash':rhs,'inprocess_total_wall_ms':'12.0',
              'iterations':'100','peak_rss_kib':'8096'}
            base[(case,mode)]=b
            for arm in r.ARMS:
                local=arm=='tplus_h_static' or k==0
                h_ok=arm=='tplus_h_static' or k!=2
                status='Converged' if h_ok else 'MaxIterations'
                outcome=r.status_outcome(t_ok,h_ok,local)
                rows.append({'case':case,'family':'scalar' if case.startswith('scalar') else 'elasticity',
                    'rhs_mode':mode,'arm':arm,'matrix_sha256':'deadbeef','rhs_hash':rhs,
                    'status':status,'iterations':'50','local_used':'true' if local else 'false',
                    'local_regions':'8' if local else '0','local_bytes':'256' if local else '0',
                    'coarse_bytes':'4096','peak_rss_kib':'8192','inprocess_total_wall_ms':'18.0',
                    'record_status':'MEASURED','process_exit_status':'0','matched_outcome':outcome})
    return rows,base


class OutcomeReviewTests(unittest.TestCase):
    def test_case_grid_and_strict_csv(self):
        self.assertEqual(len(r.CASES)*len(r.MODES)*len(r.ARMS),48)
        self.assertEqual(len(set(r.MATCHED_FIELDS)),len(r.MATCHED_FIELDS))
        self.assertEqual(len(r.strict_csv(b'case,arm\na,b\n')),1)
    def test_full_matched_groups(self):
        rows,base=fixture();d=r.inspect(rows,base)
        self.assertEqual(len(d),48)
        gs,per=r.group_report(d)
        self.assertEqual(len(per),12)
        self.assertEqual(gs['all:tplus_h_static']['local_used'],24)
        self.assertEqual(gs['all:tplus_h_static']['rescue_local'],6)
        self.assertEqual(gs['all:tplus_h_static']['h_converged'],24)
        self.assertEqual(gs['all:tplus_h_adaptive']['local_used'],6)
        self.assertEqual(gs['all:tplus_h_adaptive']['regression'],6)
        self.assertEqual(gs['all:tplus_h_adaptive']['rescue_local'],6)
    def test_report_renders_and_exports(self):
        rows,base=fixture();d=r.inspect(rows,base);g,m=r.group_report(d)
        report=r.render(g,m,d)
        self.assertIn('elastic_shear_layers',report)
        self.assertIn('Coarse-only recovery',report)
        self.assertIn('TPLUS_H_RESCUE',report)
        exp=r.strict_csv(r.canonical_csv(d))
        self.assertEqual(len(exp),48)
        self.assertEqual(exp[0]['local_used'],'true')
    def test_duplicate_trial_is_rejected(self):
        rows,base=fixture()
        with self.assertRaisesRegex(ValueError,'duplicate'):r.inspect(rows+[deepcopy(rows[0])],base)
    def test_missing_trial_is_rejected(self):
        rows,base=fixture()
        with self.assertRaisesRegex(ValueError,'incomplete'):r.inspect(rows[:-1],base)
    def test_wrong_hash_is_rejected(self):
        rows,base=fixture();rows[0]['rhs_hash']='bad'
        with self.assertRaisesRegex(ValueError,'matrix or RHS'):r.inspect(rows,base)
    def test_fake_local_static_is_rejected(self):
        rows,base=fixture();t=next(x for x in rows if x['arm']=='tplus_h_static')
        t['local_used']='false'
        with self.assertRaisesRegex(ValueError,r'static T\+H'):r.inspect(rows,base)
    def test_outcome_laundering_rejected(self):
        rows,base=fixture();rows[0]['matched_outcome']='BOTH_CONVERGED'
        with self.assertRaisesRegex(ValueError,'disagrees'):r.inspect(rows,base)
    def test_fake_process_error_rejected(self):
        rows,base=fixture();rows[0]['record_status']='PROCESS_ERROR'
        with self.assertRaisesRegex(ValueError,'process errors'):r.inspect(rows,base)
    def test_nonfinite_time_rejected(self):
        rows,base=fixture();rows[0]['inprocess_total_wall_ms']='nan'
        with self.assertRaisesRegex(ValueError,'nonfinite'):r.inspect(rows,base)
    def test_status_classification(self):
        self.assertEqual(r.status_outcome(False,True,False),'COARSE_ONLY_RECOVERY_NOT_LOCAL_RESCUE')
        self.assertEqual(r.status_outcome(False,True,True),'TPLUS_H_RESCUE')
        self.assertEqual(r.status_outcome(True,False,True),'TPLUS_H_REGRESSION')
        self.assertEqual(r.status_outcome(False,False,True),'BOTH_NOT_CONVERGED')
    def test_nonfinite_baseline_time_rejected(self):
        rows,base=fixture();base[(r.CASES[0],r.MODES[0])]['inprocess_total_wall_ms']='inf'
        with self.assertRaisesRegex(ValueError,'nonfinite'):r.inspect(rows,base)

if __name__=='__main__':unittest.main()
