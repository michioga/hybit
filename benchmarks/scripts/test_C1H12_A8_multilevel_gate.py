"""Negative and complete-corpus regression tests for native A8 evidence."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, CASES
from benchmarks.scripts.C1H12_A8_multilevel_gate import audit, parse_a8
from benchmarks.scripts.test_C1H12_A7_symmetric_gate import a7


def a8(*, case='island_shell', arm='strong_graph_smoothed', policy='multilevel',
       status='Converged', iterations=100, verified='6e-11', calls=101):
    _, agg, basis = ARMS[arm]
    multilevel = policy == 'multilevel'
    levels = 4 if multilevel else 2
    dimensions = '20736,5184,1296,324' if multilevel else '20736,512'
    nnzs = f'{CASES[case]},50000,10000,2000' if multilevel else f'{CASES[case]},0'
    if not multilevel:
        calls = 0
    data = dict(
        case=case, policy='Multilevel' if multilevel else 'TwoLevel',
        aggregation=agg, basis=basis, rows='20736', nnz=str(CASES[case]),
        levels=str(levels), level_dims=dimensions, level_nnz=nnzs,
        coarse_dimension='324' if multilevel else '512',
        coarse_factor_bytes=str(324 * 324 * 8 if multilevel else 200000),
        symbolic_nnz='65000' if multilevel else '0',
        exact_zero_candidates='0', abtm_words='5000' if multilevel else '0',
        status=status, iterations=str(iterations), tol='1e-10',
        reported_relative='5e-11', verified_true_relative=verified,
        relative_x_error='2e-8', load_ms='8', setup_ms='20', solve_ms='40',
        total_wall_ms='68', true_checks='1', restarts='0',
        rejected_recursive_convergences='0', precond_calls=str(calls),
        extra_spmv_calls=str(2 * (levels - 1) * calls),
        scratch_bytes='1000000' if multilevel else '0',
    )
    return 'C1H12A8|' + '|'.join(f'{k}={v}' for k, v in data.items())


class A8GateTests(unittest.TestCase):
    def test_accept_multilevel_and_frozen_control(self):
        self.assertEqual(
            parse_a8(a8(), 'island_shell', 'strong_graph_smoothed', 'multilevel')['levels'], 4
        )
        self.assertEqual(
            parse_a8(a8(policy='two_level'), 'island_shell', 'strong_graph_smoothed', 'two_level')['levels'], 2
        )

    def test_false_convergence_and_inconsistent_work_are_rejected(self):
        examples = (
            (a8(verified='1e-7'), 'false convergence'),
            (a8().replace('extra_spmv_calls=606', 'extra_spmv_calls=605'), 'SpMV work'),
            (a8().replace('level_dims=20736,5184,1296,324',
                          'level_dims=20736,5184,5184,324'), 'nondecreasing'),
            (a8().replace('symbolic_nnz=65000', 'symbolic_nnz=nan'), 'invalid literal'),
            (a8() + '|iterations=7', 'duplicate'),
            (a8().replace('solve_ms=40', 'solve_ms=inf'), 'nonfinite'),
        )
        for sample, text in examples:
            with self.subTest(problem=text), self.assertRaisesRegex(ValueError, text):
                parse_a8(sample, 'island_shell', 'strong_graph_smoothed', 'multilevel')

    def test_complete_36_and_bad_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path(folder) / 'a7'
            trial = Path(folder) / 'a8'
            previous.mkdir()
            trial.mkdir()
            for base in (previous, trial):
                (base / 'HEAD.txt').write_text('1f3eaa5-full\n')
                (base / 'matrix_sha256.txt').write_text('three hashes\n')
            for case in CASES:
                for arm in ARMS:
                    if arm == 'jacobi':
                        continue
                    (previous / f'{case}_{arm}_additive_1.log').write_text(
                        a7(case=case, arm=arm, policy='additive', precond_calls=0,
                           iterations=100, true_rel='6e-11') + '\n'
                    )
                    for policy in ('two_level', 'multilevel'):
                        stem = f'{case}_{arm}_{policy}_1'
                        (trial / f'{stem}.log').write_text(
                            a8(case=case, arm=arm, policy=policy) + '\n'
                        )
                        (trial / f'{stem}.time').write_text(
                            'elapsed_s=0.1 peak_rss_kib=9000 exit_status=0\n'
                        )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 0)
            self.assertEqual(len((trial / 'native_multilevel_comparison.csv').read_text().splitlines()), 37)
            (trial / 'matrix_sha256.txt').write_text('different hashes\n')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)
            (trial / 'matrix_sha256.txt').write_text('three hashes\n')
            (trial / 'rotated_anisotropy_graph_smoothed_multilevel_1.log').unlink()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)


if __name__ == '__main__':
    unittest.main()
