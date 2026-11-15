#!/usr/bin/env python3
import csv,io,tempfile,unittest
from pathlib import Path
import C1G5I_compare_p2 as m
class TestCompare(unittest.TestCase):
    def sample(self):
        # Synthetic complete 12-record grid; never depend on a packager-local path.
        d={}
        for c in m.CASES:
            for a in m.ARMS:
                d[(c,a)]={'case':c,'arm':a,'rhs_hash':c+'_hash','baseline_T_status':'Converged',
                    'baseline_T_iterations':'100','status':'Breakdown','iterations':'99',
                    'true_relative':'1e-9','kahan_relative':'1e-9','reported_relative':'1e-9',
                    'local_regions':'0','local_bytes':'0','escalations':'0'}
        return d
    def test_full_matches(self):
        a=self.sample()
        self.assertEqual(len(m.compare(a,a)),12)
    def test_reject_rhs_mismatch(self):
        a=self.sample();b={k:v.copy() for k,v in a.items()};b[next(iter(b))]['rhs_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'RHS'):
            m.compare(a,b)
    def test_reject_spurious_no_local(self):
        a=self.sample();b={k:v.copy() for k,v in a.items()};b[('scalar_multi_patch','c1g5i_coarse_observe')]['local_regions']='1'
        with self.assertRaisesRegex(ValueError,'local'):
            m.compare(a,b)
    def test_prior_csv_is_pinned(self):
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            m.read(Path(__file__),m.PRIOR_SHA)
if __name__=='__main__':unittest.main()
