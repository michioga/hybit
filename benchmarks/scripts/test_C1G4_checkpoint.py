import unittest
from benchmarks.scripts.C1G4_checkpoint import stats_for, csv_of, md_report


def sample(case='island_shell', arm='graph_piecewise', mode='ones', policy='two_level',
           iterations=10, solve=1.0):
    return dict(case=case, arm=arm, rhs_mode=mode, policy=policy,
                status='Converged', verified_true_relative='1e-11',
                relative_x_error='1e-8', iterations=str(iterations),
                solve_ms=str(solve), setup_ms='2.0', peak_rss_kib='2000')


class CheckpointTests(unittest.TestCase):
    def test_stats_valid(self):
        a=sample(iterations=100, solve=4.0)
        b=sample(policy='multilevel', iterations=20, solve=2.0)
        s=stats_for([a,b],'ones')
        self.assertEqual(s['pairs'],1)
        self.assertEqual(s['paired_solve_speedup_median'],2.0)
        self.assertEqual(s['multilevel_iterations_sum'],20)

    def test_stats_missing_pair(self):
        with self.assertRaisesRegex(ValueError,'missing paired'):
            stats_for([sample()],'ones')

    def test_stats_zero_time(self):
        with self.assertRaisesRegex(ValueError,'invalid solve-time'):
            stats_for([sample(),sample(policy='multilevel',solve=0)],'ones')

    def test_stats_non_ones_filter(self):
        rows=[sample(mode='ones'),sample(mode='ones',policy='multilevel'),
              sample(mode='seeded_random',policy='two_level'),
              sample(mode='seeded_random',policy='multilevel')]
        self.assertEqual(stats_for(rows,'non_ones')['pairs'],1)
        self.assertEqual(stats_for(rows,'all')['pairs'],2)

    def test_csv_schema(self):
        self.assertEqual(csv_of([{'name':'a','count':2}],('name','count')),
                         'name,count\na,2\n')


if __name__ == '__main__':
    unittest.main()
