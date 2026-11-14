import copy
import unittest
from benchmarks.scripts import C1G5E_outcome_audit as a


def fixture():
    records=[]
    groups=[]
    for case in a.CASES:
        for rhs in a.RHS:
            good=[]
            for i, arm in enumerate(a.ARMS):
                # Exactly 32 converged / 16 max-iterations on synthetic fixture
                success=arm!='hybrid_local_cholesky' or rhs in ('ones','point_load')
                if success:good.append(arm)
                records.append(dict(case=case,rhs_mode=rhs,arm=arm,
                    matrix_sha256=a.MATRIX_SHA256[case],rhs_hash='0123456789abcdef',
                    status='Converged' if success else 'MaxIterations',
                    iterations='125' if success else '600',tol='1e-10',max_iterations='600',
                    verified_true_relative='7e-11' if success else '1e-4',
                    compensated_true_relative='8e-11' if success else '1e-4',
                    relative_x_error='n/a' if rhs=='point_load' else '1e-6',
                    factor_bytes='1024' if i==1 else '0',
                    local_regions='1' if i==1 else '0',escalations='1' if i==1 else '0',
                    effective_preconditioner='Hybrid' if i==1 else 'Jacobi',
                    arm_work_ms=str(3+i),inprocess_total_wall_ms=str(4+i),
                    elapsed_s='0.02',peak_rss_kib='5000',process_exit_status='0',
                    record_status='MEASURED'))
            groups.append({'case':case,'rhs_mode':rhs,'valid_numerical_arms':good})
    summary={'source_head':a.HEAD,'processes':48,'process_errors':0,
             'measured':48,'converged':sum(r['status']=='Converged' for r in records),
             'complete_groups':12,'promotion_allowed':False,'groups':groups}
    return records,summary


class AuditTest(unittest.TestCase):
    def test_complete_contract(self):
        rows,summary=fixture()
        report=a.analyze(rows,summary,'a'*64)
        self.assertEqual(report['converged'],42) # 3 always + 1 half = 3.5/group * 12
        self.assertEqual(report['unconverged'],6)
        self.assertEqual(sum(x['converged'] for x in report['per_matrix']),42)
        self.assertEqual(len(report['groups']),12)
        self.assertFalse(report['promotion_allowed'])
        self.assertIn('C1-G5E',a.render(report))
        self.assertEqual(len(a.output_csv(report).splitlines()),13)

    def test_bad_matrix_rejected(self):
        rows,summary=fixture();rows[0]['matrix_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'matrix identity'):
            a.analyze(rows,summary,'a'*64)

    def test_rhs_disagreement_rejected(self):
        rows,summary=fixture();rows[1]['rhs_hash']='f'*16
        with self.assertRaisesRegex(ValueError,'RHS differs'):
            a.analyze(rows,summary,'a'*64)

    def test_false_convergence_rejected(self):
        rows,summary=fixture();rows[0]['verified_true_relative']='5e-8'
        with self.assertRaisesRegex(ValueError,'false convergence'):
            a.analyze(rows,summary,'a'*64)

    def test_nonconvergence_not_winner(self):
        rows,summary=fixture()
        for r in rows:
            if r['case']=='elastic_uniform' and r['rhs_mode']=='ones':
                r['inprocess_total_wall_ms']='0.01' if r['status']!='Converged' else '20'
        report=a.analyze(rows,summary,'a'*64)
        self.assertNotEqual(report['groups'][0]['nominal_fastest_converged_arm'],a.ARMS[1])

    def test_nothing_converged_explicit_no_feasible_arm(self):
        rows,summary=fixture()
        for r in rows:
            if r['case']=='elastic_uniform' and r['rhs_mode']=='ones':
                r['status']='MaxIterations'
                r['iterations']='600'
                r['verified_true_relative']='0.0001'
                r['compensated_true_relative']='0.0001'
        summary['converged']-=4
        summary['groups'][0]['valid_numerical_arms']=[]
        result=a.analyze(rows,summary,'a'*64)
        self.assertTrue(result['groups'][0]['no_feasible_arm'])
        self.assertEqual(result['groups'][0]['nominal_fastest_converged_arm'],'NO_FEASIBLE_ARM')

    def test_summary_disagrees_rejected(self):
        rows,summary=fixture();summary['converged']=48
        with self.assertRaisesRegex(ValueError,'contradicts'):
            a.analyze(rows,summary,'a'*64)

    def test_duplicate_key_rejected(self):
        rows,summary=fixture();rows[1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            a.analyze(rows,summary,'a'*64)

    def test_missing_reference_rejected(self):
        rows,summary=fixture();rows[0]['relative_x_error']='n/a'
        with self.assertRaisesRegex(ValueError,'bad numeric'):
            a.analyze(rows,summary,'a'*64)

    def test_hybrid_factor_use_mismatch_rejected(self):
        rows,summary=fixture();rows[1]['local_regions']='0'
        with self.assertRaisesRegex(ValueError,'factor-use'):
            a.analyze(rows,summary,'a'*64)

    def test_process_error_not_silently_dropped(self):
        rows,summary=fixture();rows[0]['record_status']='PROCESS_ERROR'
        with self.assertRaisesRegex(ValueError,'process failure'):
            a.analyze(rows,summary,'a'*64)

    def test_no_false_production_promotion(self):
        rows,summary=fixture();summary['promotion_allowed']=True
        with self.assertRaisesRegex(ValueError,'blocked-promotion'):
            a.analyze(rows,summary,'a'*64)

if __name__=='__main__': unittest.main()
