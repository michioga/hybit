"""Offline fail-closed schema and leakage tests. No Rust or repo mutation."""
import csv
import io
import json
import unittest
from copy import deepcopy

try:
    from benchmarks.scripts import C1G5A_corpus_ledger as m
except ModuleNotFoundError:
    import C1G5A_corpus_ledger as m


def source_fixture():
    rows = []
    for ci, case in enumerate(m.CASES):
        for ai, arm in enumerate(m.CONFIGS):
            for ri, rhs in enumerate(m.RHS):
                for pi, policy in enumerate(m.POLICIES):
                    iterations = 20 + ci * 7 + ai * 3 + ri * 2 + pi * 10
                    rows.append(dict(case=case, arm=arm, rhs_mode=rhs, policy=policy,
                        rhs_hash=f'{ci+10:02x}{ri+10:02x}'+'a'*12, status='Converged',
                        iterations=str(iterations), verified_true_relative='4e-11',
                        relative_x_error='n/a' if rhs == 'point_load' else '0.01',
                        levels='2' if pi == 0 else '4', coarse_factor_bytes='8000',
                        scratch_bytes='0' if pi == 0 else '400', setup_ms='2.0',
                        solve_ms='10.0' if pi == 0 else '8.0',
                        total_wall_ms='14.0' if pi == 0 else '13.0',
                        elapsed_s='0.02', peak_rss_kib='12000'))
    sums = {policy: (sum(int(x['iterations']) for x in rows if x['policy'] == policy),
                     sum(float(x['solve_ms']) for x in rows if x['policy'] == policy)) for policy in m.POLICIES}
    summary = {
        'source_head': m.SOURCE_HEAD, 'numerical_gate': 'PASS_ORIGINAL_CSR_AND_GRID',
        'matrix_sha256': m.CASE_HASH, 'numerical_policy': {
            'tol': 1e-10, 'max_iterations': 600, 'threads': 1, 'rhs': list(m.RHS)},
        'runs': 72, 'pairs': 36, 'outcomes': {'BOTH_CONVERGED': 36},
        'group_statistics': {'all': {
            'two_level_iterations_sum': sums['two_level'][0],
            'multilevel_iterations_sum': sums['multilevel'][0],
            'two_level_solve_ms_sum': sums['two_level'][1],
            'multilevel_solve_ms_sum': sums['multilevel'][1],
        }},
    }
    return rows, summary


def make_csv(rows):
    return m.csv_bytes(rows, m.INPUT_COLUMNS)


def build(rows=None, summary=None):
    rr, ss = source_fixture()
    return m.make_outputs(make_csv(rr if rows is None else rows),
                          json.dumps(ss if summary is None else summary).encode())


class LedgerTests(unittest.TestCase):
    def test_complete_grid_and_missing_arms(self):
        generated = build()
        ledger = list(csv.DictReader(io.StringIO(generated['benchmarks/evidence/C1G5A_20261114_ledger.csv'].decode())))
        self.assertEqual(len(ledger), 96)
        self.assertEqual(sum(r['measurement_status']=='MEASURED' for r in ledger), 72)
        self.assertEqual(sum(r['measurement_status']=='NOT_RUN' for r in ledger), 24)
        pending = [r for r in ledger if r['measurement_status']=='NOT_RUN']
        self.assertTrue(all(r['verified_true_relative']=='' and r['inprocess_total_wall_ms']=='' for r in pending))
        self.assertEqual(set(r['family_id'] for r in ledger), {'constructed_c1g_n144'})

    def test_matrix_inventory_external_unverified(self):
        inventory = list(csv.DictReader(io.StringIO(build()['benchmarks/evidence/C1G5A_20261114_matrix_inventory.csv'].decode())))
        self.assertEqual(len(inventory), 6)
        external = [r for r in inventory if r['origin']=='external_fem_matrix']
        self.assertEqual(len(external), 3)
        self.assertTrue(all(r['matrix_contract']=='SPD_UNVERIFIED' and r['rhs_availability']=='NO_PAIRED_RHS' for r in external))

    def test_shadow_constant_and_unknown(self):
        self.assertEqual(m.shadow_constant_control({'matrix_contract': 'SPD_CANDIDATE'})[0], m.BASELINE)
        self.assertEqual(m.shadow_constant_control({'matrix_contract': 'SPD_UNVERIFIED'})[0], 'NO_RECOMMENDATION')

    def test_feature_allowlist_prevents_leakage(self):
        mat = m.matrix_inventory()[0]
        f = m.presolve_features(mat, 1e-10, 600)
        self.assertEqual(set(f), {'matrix_contract','nrows','nnz','requested_tol','max_iterations'})
        for forbidden in ('rhs_mode_offline_label', 'iterations', 'solve_ms', 'relative_x_error_offline_only','matrix_sha256'):
            self.assertNotIn(forbidden, f)

    def test_shadow_does_not_peek_at_fastest(self):
        rr, ss = source_fixture()
        o1 = m.shadow_constant_control(m.presolve_features(m.matrix_inventory()[0], 1e-10, 600))
        for row in rr:
            row['total_wall_ms'] = '50' if row['policy']=='two_level' else '100'
        o2 = m.shadow_constant_control(m.presolve_features(m.matrix_inventory()[0], 1e-10, 600))
        self.assertEqual(o1, o2)

    def test_deterministic_output(self):
        self.assertEqual(build(), build())

    def test_decision_schema(self):
        out=build()
        decisions=list(csv.DictReader(io.StringIO(out['benchmarks/evidence/C1G5A_20261114_shadow_audit.csv'].decode())))
        self.assertEqual(len(decisions),12)
        self.assertTrue(all(r['observed_arm_count']=='6' and r['missing_arm_count']=='2' for r in decisions))
        self.assertTrue(all(float(r['shadow_regret_ratio']) >= 1.0 for r in decisions))
        self.assertEqual(json.loads(out['benchmarks/evidence/C1G5A_20261114_summary.json'])['held_out_family_gate'],
                         'BLOCKED_ONLY_ONE_MEASURED_FAMILY')

    def test_forward_error_is_offline_only_and_point_load_missing(self):
        ledger=list(csv.DictReader(io.StringIO(build()['benchmarks/evidence/C1G5A_20261114_ledger.csv'].decode())))
        self.assertTrue(all(r['relative_x_error_offline_only']=='' for r in ledger if r['rhs_mode_offline_label']=='point_load'))

    def test_duplicate_fails(self):
        rr,ss=source_fixture()
        rr[-1]=deepcopy(rr[0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build(rr,ss)

    def test_missing_fails(self):
        rr,ss=source_fixture()
        with self.assertRaisesRegex(ValueError, 'incomplete source grid'):
            build(rr[:-1],ss)

    def test_rhs_mismatch_fails(self):
        rr,ss=source_fixture()
        rr[1]['rhs_hash']='f'*16
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            build(rr,ss)

    def test_false_convergence_fails(self):
        rr,ss=source_fixture()
        rr[0]['verified_true_relative']='2e-8'
        with self.assertRaisesRegex(ValueError,'false convergence'):
            build(rr,ss)

    def test_budget_violation_fails(self):
        rr,ss=source_fixture()
        rr[0]['iterations']='601'
        with self.assertRaisesRegex(ValueError,'iteration budget'):
            build(rr,ss)

    def test_point_load_no_xref(self):
        rr,ss=source_fixture()
        next(x for x in rr if x['rhs_mode']=='point_load')['relative_x_error']='0.1'
        with self.assertRaisesRegex(ValueError, 'point_load'):
            build(rr,ss)

    def test_wrong_head_fails(self):
        rr,ss=source_fixture()
        ss['source_head']='123'
        with self.assertRaisesRegex(ValueError,'provenance'):
            build(rr,ss)

    def test_wrong_family_hash_fails(self):
        rr,ss=source_fixture()
        ss['matrix_sha256']=dict(m.CASE_HASH, island_shell='0'*64)
        with self.assertRaisesRegex(ValueError,'matrix hashes'):
            build(rr,ss)

    def test_bad_frozen_summary_fails(self):
        rr,ss=source_fixture()
        ss['group_statistics']['all']['two_level_iterations_sum']+=1
        with self.assertRaisesRegex(ValueError,'iteration total'):
            build(rr,ss)

    def test_unknown_arm_fails(self):
        rr,ss=source_fixture()
        rr[0]['arm']='not_a_real_method'
        with self.assertRaisesRegex(ValueError,'unexpected case'):
            build(rr,ss)

    def test_nonfinite_time_fails(self):
        rr,ss=source_fixture()
        rr[0]['total_wall_ms']='nan'
        with self.assertRaisesRegex(ValueError,'non-finite'):
            build(rr,ss)

    def test_extra_column_fails(self):
        rr,ss=source_fixture()
        data=make_csv(rr).replace(b'case,arm',b'case,extra,arm',1)
        with self.assertRaisesRegex(ValueError, 'schema'):
            m.make_outputs(data,json.dumps(ss).encode())


if __name__=='__main__':
    unittest.main()
