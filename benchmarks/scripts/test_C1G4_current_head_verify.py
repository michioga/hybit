import tempfile
import unittest
from pathlib import Path
from benchmarks.scripts.C1G4_current_head_verify import (
    ARMS, CASES, MODES, EXPECTED_N, POLICIES, expected_specs,
    matrix_hashes, stem_for, summarize, validate_pair, normalized_record, FIELDS,
)


def fake(case='multi_patch', arm='graph_piecewise', rhs_mode='seeded_random',
         policy='two_level', status='Converged', verified=1e-11, fingerprint='aaa'):
    return dict(case=case, arm=arm, rhs_mode=rhs_mode, policy=policy,
                rhs_hash=fingerprint, status=status, verified_true_relative=verified)


class Tests(unittest.TestCase):
    def test_a9_projector_matches_actual_parser_schema(self):
        spec = ('island_shell', 'graph_piecewise', 'point_load', 'multilevel')
        # A9's parser contains no arm field, and uses native Rust policy spelling.
        parsed = {'case': 'island_shell', 'policy': 'Multilevel',
                  'rhs_mode': 'point_load', 'rhs_hash': '0123456789abcdef',
                  'status': 'Converged', 'verified_true_relative': 9e-11,
                  'relative_x_error': 'n/a', 'iterations': 120,
                  'levels': 4, 'coarse_factor_bytes': 10000,
                  'scratch_bytes': 2048, 'setup_ms': 5.0,
                  'solve_ms': 4.0, 'total_wall_ms': 10.0}
        timing = {'elapsed_s': 0.12, 'peak_rss_kib': 35000, 'exit_status': 0}
        record = normalized_record(spec, parsed, timing)
        self.assertEqual(set(record), set(FIELDS))
        self.assertEqual(record['arm'], 'graph_piecewise')
        self.assertEqual(record['policy'], 'multilevel')
        self.assertEqual(record['rhs_mode'], 'point_load')
        self.assertEqual(record['relative_x_error'], 'n/a')
        self.assertEqual(parsed['policy'], 'Multilevel')  # must not mutate parser output
        self.assertNotIn('arm', parsed)

    def test_a9_projector_both_policies(self):
        for policy, expected in [('two_level', 'TwoLevel'), ('multilevel', 'Multilevel')]:
            parsed = {'case': 'multi_patch', 'rhs_mode': 'ones', 'policy': expected}
            parsed.update({key: 0 for key in FIELDS if key not in
                           {'case', 'arm', 'policy', 'rhs_mode', 'elapsed_s', 'peak_rss_kib'}})
            result = normalized_record(('multi_patch', 'strong_graph_piecewise', 'ones', policy),
                                       parsed, {'elapsed_s': 1., 'peak_rss_kib': 1000})
            self.assertEqual(result['policy'], policy)

    def test_a9_projector_rejects_wrong_native_policy(self):
        spec=('multi_patch', 'graph_piecewise', 'ones', 'two_level')
        parsed={'case':'multi_patch', 'rhs_mode':'ones', 'policy':'Multilevel'}
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            normalized_record(spec, parsed, {'elapsed_s': 1., 'peak_rss_kib': 1000})

    def test_a9_projector_requires_time_fields(self):
        spec=('multi_patch', 'graph_piecewise', 'ones', 'two_level')
        parsed={'case':'multi_patch', 'rhs_mode':'ones', 'policy':'TwoLevel'}
        with self.assertRaisesRegex(ValueError, 'incomplete time'):
            normalized_record(spec, parsed, {'elapsed_s': 1.})

    def test_schedule(self):
        specs=list(expected_specs())
        self.assertEqual(len(specs),72)
        self.assertEqual(len({stem_for(s) for s in specs}),72)
        self.assertEqual(len(CASES)*len(ARMS)*len(MODES)*len(POLICIES), EXPECTED_N)

    def test_pair_both_converged(self):
        self.assertEqual(validate_pair(fake(),fake(policy='multilevel')), 'BOTH_CONVERGED')

    def test_pair_rescue(self):
        self.assertEqual(validate_pair(fake(status='MaxIterations',verified=1e-4),
                                      fake(policy='multilevel')), 'RESCUE')

    def test_pair_regression(self):
        self.assertEqual(validate_pair(fake(),fake(policy='multilevel',status='MaxIterations',verified=1e-3)),
                         'REGRESSION')

    def test_pair_both_not_converged(self):
        self.assertEqual(validate_pair(fake(status='Breakdown',verified=1e-2),
                                      fake(policy='multilevel',status='Breakdown',verified=1e-2)),
                         'BOTH_NOT_CONVERGED')

    def test_pair_wrong_rhs_hash_rejected(self):
        with self.assertRaises(ValueError):
            validate_pair(fake(),fake(policy='multilevel', fingerprint='bbb'))

    def test_pair_wrong_policy_rejected(self):
        with self.assertRaises(ValueError):
            validate_pair(fake(),fake(policy='two_level'))

    def test_pair_wrong_identity_rejected(self):
        with self.assertRaises(ValueError):
            validate_pair(fake(),fake(policy='multilevel',case='island_shell'))

    def test_summary_complete(self):
        rows=[]
        for case,arm,mode,policy in expected_specs():
            rows.append(fake(case,arm,mode,policy))
        report=summarize(rows)
        self.assertEqual(report['pairs'],36)
        self.assertEqual(report['non_ones_pairs'],27)
        self.assertEqual(report['outcomes_total'],{'BOTH_CONVERGED':36})

    def test_summary_missing_rejected(self):
        rows=[fake(*s) for s in list(expected_specs())[:-1]]
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_summary_duplicate_rejected(self):
        specs=list(expected_specs())
        rows=[fake(*s) for s in specs[:-1]] + [fake(*specs[0])]
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_matrix_missing_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                matrix_hashes(Path(td))

    def test_matrix_hashes_different(self):
        with tempfile.TemporaryDirectory() as td:
            for i,case in enumerate(CASES):
                (Path(td)/f'c1g_p1_{case}_n144.mtx').write_text(f'{i} fake\n')
            self.assertEqual(len(matrix_hashes(Path(td))),3)

    def test_matrix_identical_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for case in CASES:
                (Path(td)/f'c1g_p1_{case}_n144.mtx').write_text('fake\n')
            with self.assertRaises(ValueError):
                matrix_hashes(Path(td))


if __name__=='__main__':
    unittest.main()
