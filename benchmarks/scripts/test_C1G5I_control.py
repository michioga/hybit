#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import unittest
from unittest.mock import patch
import C1G5I_control as m

FIXTURE='''enum FourArm {
        TPlusHAdaptive,
    }

    fn parse_e_args() {
                        "tplus-h-adaptive" => FourArm::TPlusHAdaptive,
            FourArm::TPlusHAdaptive => {
                solver.set_hybrid_options(config)?;
            }
            FourArm::Multilevel => {
            }
            FourArm::TPlusHAdaptive => "tplus_h_adaptive",
'''

class Tests(unittest.TestCase):
    def test_gnu_time_success(self):
        metrics=m.parse_gnu_time('elapsed_s=0.16\npeak_rss_kib=8128\nexit_status=0\n',0,'case__arm')
        self.assertEqual(metrics['peak_rss_kib'],'8128')

    def test_gnu_time_failure_diagnostic_preserved(self):
        t='Command exited with non-zero status 1\nelapsed_s=0.00\npeak_rss_kib=7000\nexit_status=1\n'
        metrics=m.parse_gnu_time(t,1,'case__arm')
        self.assertEqual(metrics['exit_status'],'1')
        with self.assertRaisesRegex(ValueError,'process failure'):
            m.parse_gnu_time(t.replace('exit_status=1','exit_status=0'),0,'case__arm')

    def test_gnu_time_reject_duplicate_fields(self):
        t='elapsed_s=0.16\npeak_rss_kib=8128\nexit_status=0\nexit_status=0\n'
        with self.assertRaisesRegex(ValueError,'invalid GNU time line'):
            m.parse_gnu_time(t,0,'case__arm')

    def test_no_segment_keeps_coarse_base(self):
        data=FIXTURE.encode()
        with patch.object(m,'RUST_SOURCE_SHA',m.digest(data)):
            out=m.transform(data).decode()
        start=out.index('if matches!(arm, FourArm::C1G5INoSegment)')
        end=out.index('} else if matches!(arm, FourArm::C1G5IObserve)',start)
        piece=out[start:end]
        self.assertIn('config.enabled = true;',piece)
        self.assertIn('config.escalation_residual_ratio = f64::MAX;',piece)
        self.assertNotIn('config.enabled = false;',piece)

    def test_sha_pin_blocks_wrong(self):
        with self.assertRaises(ValueError):m.transform(b'bad')

    def test_transform_control_arms(self):
        data=FIXTURE.encode()
        with patch.object(m,'RUST_SOURCE_SHA',m.digest(data)):
            out=m.transform(data).decode()
        for string in ('C1G5INoSegment','C1G5IObserve','config.enabled = true;',
                       'config.escalation_residual_ratio = f64::MAX;',
                       'config.progress_monitor_interval = 0;',
                       'config.progress_monitor_observe_only = true;',
                       'c1g5i_coarse_no_segment','c1g5i_coarse_observe'):
            self.assertIn(string,out)
        self.assertEqual(out.count('solver.set_hybrid_options(config)?;'),1)
        self.assertNotIn('config.enabled = false;',out)

    def test_transform_missing_or_duplicate_anchor(self):
        for bad in (FIXTURE.replace('FourArm::TPlusHAdaptive => "tplus_h_adaptive",',''),
                    FIXTURE+'            FourArm::TPlusHAdaptive => "tplus_h_adaptive",'):
            data=bad.encode()
            with patch.object(m,'RUST_SOURCE_SHA',m.digest(data)):
                with self.assertRaises(ValueError):m.transform(data)

    def test_fileset_exact(self):
        self.assertEqual(len(m.fileset()),53)
        self.assertIn('elastic_stiff_lens__c1g5i_coarse_observe.rc',m.fileset())

    def base(self):
        return {'rhs_hash':'0123456789abcdef','nnz':'1000'}

    def record(self,arm='c1g5i_coarse_observe',status='Converged',true='9e-11'):
        d=dict.fromkeys(m.KEYS,'0')
        d.update(case='elastic_stiff_lens',arm=arm,rhs_mode='ones',rhs_hash='0123456789abcdef',
                 status=status,iterations='430' if status!='MaxIterations' else '600',
                 tol='1e-10',max_iterations='600',reported_relative='9e-11',
                 verified_true_relative=true,compensated_true_relative=true,
                 relative_x_error='1e-8',rows='3280',nnz='1000',
                 inprocess_total_wall_ms='101.3',factor_bytes='1857456',
                 coarse_bytes='1857456',local_bytes='0',local_regions='0',
                 escalations='0',local_used='false')
        return 'C1G5H|'+'|'.join(f'{key}={d[key]}' for key in m.KEYS)+'\n'

    def test_pinned_success_and_no_local(self):
        x=m.parse_record(self.record(),'elastic_stiff_lens','c1g5i_coarse_observe',self.base(),3280)
        self.assertEqual(x['status'],'Converged')

    def test_false_convergence_rejected(self):
        with self.assertRaisesRegex(ValueError,'false certified success'):
            m.parse_record(self.record(true='1.1e-10'),'elastic_stiff_lens','c1g5i_coarse_observe',self.base(),3280)

    def test_no_local_control_cannot_claim_local(self):
        rec=self.record().replace('local_used=false','local_used=true')
        with self.assertRaises(ValueError):m.parse_record(rec,'elastic_stiff_lens','c1g5i_coarse_observe',self.base(),3280)

    def test_bad_rhs_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'RHS fingerprint'):
            m.parse_record(self.record().replace('0123456789abcdef','eeeeeeeeeeeeeeee'),
                'elastic_stiff_lens','c1g5i_coarse_observe',self.base(),3280)

    def test_incomplete_budget_rejected(self):
        with self.assertRaisesRegex(ValueError,'max-iterations status'):
            m.parse_record(self.record(status='MaxIterations').replace('|iterations=600|','|iterations=599|'),
                'elastic_stiff_lens','c1g5i_coarse_observe',self.base(),3280)

    def test_failures_remain_failures(self):
        rec=self.record(status='Breakdown',true='1e-5')
        x=m.parse_record(rec,'elastic_stiff_lens','c1g5i_coarse_observe',self.base(),3280)
        self.assertEqual(x['status'],'Breakdown')

    def test_unchanged_status_constraint(self):
        rec=self.record().replace('status=Converged','status=Magic')
        with self.assertRaises(ValueError):m.parse_record(rec,'elastic_stiff_lens','c1g5i_coarse_observe',self.base(),3280)

if __name__=='__main__':unittest.main()
