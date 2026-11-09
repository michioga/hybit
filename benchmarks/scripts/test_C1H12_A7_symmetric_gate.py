"""Strict A7 experiment evidence gate regression tests."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, CASES
from benchmarks.scripts.C1H12_A7_symmetric_gate import audit, parse_a7
from benchmarks.scripts.test_C1H12_A4_numerical_gate import record
from benchmarks.scripts.test_C1H12_A5_certification_gate import a5


def a7(*, case='island_shell', policy='symmetric', arm='strong_graph_smoothed',
       status='Converged', iterations=100, true_rel='6e-11',
       precond_calls=112, extra_spmv_calls=None):
    _, aggregate, basis = ARMS[arm]
    if extra_spmv_calls is None:
        extra_spmv_calls = precond_calls * 2
    return (
        f'C1H12A7|case={case}|policy={"SymmetricComposite" if policy == "symmetric" else "Additive"}'
        f'|aggregation={aggregate}|basis={basis}|rows=20736|nnz={CASES[case]}'
        f'|coarse_dimension=512|coarse_factor_bytes=200000'
        f'|status={status}|iterations={iterations}|tol=1e-10'
        f'|reported_relative=5e-11|verified_true_relative={true_rel}'
        f'|relative_x_error=1e-8|load_ms=2|setup_ms=20|solve_ms=30'
        f'|total_wall_ms=52|true_checks=1|restarts=0'
        f'|rejected_recursive_convergences=0|precond_calls={precond_calls}'
        f'|extra_spmv_calls={extra_spmv_calls}'
        f'|scratch_bytes={20736 * 5 * 8 if policy == "symmetric" else 0}'
    )


class A7GateTests(unittest.TestCase):
    def test_valid_symmetric_and_additive(self):
        self.assertEqual(
            parse_a7(a7(), 'island_shell', 'strong_graph_smoothed', 'symmetric')['iterations'], 100
        )
        self.assertEqual(parse_a7(
            a7(policy='additive', precond_calls=0),
            'island_shell', 'strong_graph_smoothed', 'additive'
        )['extra_spmv_calls'], 0)

    def test_reject_false_convergence_bad_work_and_schema(self):
        samples = (
            (a7(true_rel='4e-10'), 'false convergence'),
            (a7(extra_spmv_calls=999), 'accounting mismatch'),
            (a7() + '|iterations=600', 'duplicate'),
            (a7().replace('solve_ms=30', 'solve_ms=nan'), 'nonfinite'),
            (a7().replace('rows=20736', 'rows=500'), 'input matrix'),
        )
        for sample, problem in samples:
            with self.subTest(problem=problem), self.assertRaisesRegex(ValueError, problem):
                parse_a7(sample, 'island_shell', 'strong_graph_smoothed', 'symmetric')

    def test_full_36_records_then_detect_missing_or_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path(folder) / 'a6'
            trial = Path(folder) / 'a7'
            previous.mkdir()
            trial.mkdir()
            for base in (previous, trial):
                (base / 'HEAD.txt').write_text('1f3eaa5-full\n')
                (base / 'matrix_sha256.txt').write_text('all-three-original-hashes\n')
            for case, nnz in CASES.items():
                for arm in ARMS:
                    if arm == 'jacobi':
                        continue
                    mode, aggregate, basis = ARMS[arm]
                    old_record = record(mode=mode, aggregation=aggregate,
                                        basis=basis, nnz=nnz, iterations=100)
                    (previous / f'{case}_{arm}_a5_1.log').write_text(
                        old_record + '\n' + a5(true_rel='6e-11', iterations=100,
                                                 checks=1, restarts=0, rejected=0)
                    )
                    for policy in ('additive', 'symmetric'):
                        stem = f'{case}_{arm}_{policy}_1'
                        calls = 0 if policy == 'additive' else 112
                        (trial / f'{stem}.log').write_text(
                            a7(case=case, policy=policy, arm=arm, precond_calls=calls) + '\n'
                        )
                        (trial / f'{stem}.time').write_text(
                            'elapsed_s=0.1 peak_rss_kib=9000 exit_status=0\n'
                        )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 0)
            self.assertEqual(
                len((trial / 'symmetric_composite_comparison.csv').read_text().splitlines()), 37
            )
            (trial / 'matrix_sha256.txt').write_text('changed-hashes\n')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)
            (trial / 'matrix_sha256.txt').write_text('all-three-original-hashes\n')
            (trial / 'multi_patch_graph_smoothed_symmetric_1.log').unlink()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)


if __name__ == '__main__':
    unittest.main()
