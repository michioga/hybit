"""Strict schema, true-residual and 54-case replay tests for A10-A."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10A_accuracy_gate import (
    ARMS, CASE, MODES, POLICIES, TOLERANCES, audit, parse_a10,
)
from benchmarks.scripts.test_C1H12_A9_rhs_gate import a9


def a10(arm='strong_graph_smoothed', mode='seeded_random', tol_tag='t10',
        policy='multilevel', *, verified=None):
    if verified is None:
        verified = {'t10': '6e-11', 't12': '6e-13', 't13': '6e-14'}[tol_tag]
    a9_row = a9(case=CASE, arm=arm, mode=mode, policy=policy)
    result = a9_row.replace('C1H12A9|', 'C1H12A10A|', 1)
    if tol_tag != 't10':
        result = result.replace('verified_true_relative=6e-11',
                                f'verified_true_relative={verified}')
        result = result.replace('reported_relative=5e-11',
                                'reported_relative=' + ('5e-13' if tol_tag == 't12' else '5e-14'))
        result = result.replace('tol=1e-10', f'tol={TOLERANCES[tol_tag]:.0e}')
    elif verified != '6e-11':
        result = result.replace('verified_true_relative=6e-11', f'verified_true_relative={verified}')
    return (result + '|max_iterations=1200|compensated_true_relative=' + verified
            + '|error_response_relative=' + verified
            + '|error_rayleigh=0.03|residual_amplification=300')


class A10AccuracyGateTests(unittest.TestCase):
    def test_valid_tolerances_and_controls(self):
        for tol_tag in TOLERANCES:
            for policy in POLICIES:
                with self.subTest(tol_tag=tol_tag, policy=policy):
                    row = parse_a10(a10(tol_tag=tol_tag, policy=policy),
                                    'strong_graph_smoothed', 'seeded_random', tol_tag, policy)
                    self.assertEqual(row['tol'], TOLERANCES[tol_tag])

    def test_false_convergence_and_bad_schema(self):
        examples = (
            (a10(verified='7e-9'), 'false original-CSR convergence'),
            (a10() + '|tol=1e-13', 'duplicate'),
            (a10().replace('error_rayleigh=0.03', 'error_rayleigh=nan'), 'nonfinite'),
            (a10().replace('max_iterations=1200', 'max_iterations=600'), 'iteration budget'),
            (a10().replace('extra_spmv_calls=606', 'extra_spmv_calls=605'), 'V-cycle work'),
            (a10().replace('compensated_true_relative=6e-11',
                           'compensated_true_relative=inf'), 'nonfinite'),
        )
        for sample, reason in examples:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                parse_a10(sample, 'strong_graph_smoothed', 'seeded_random', 't10', 'multilevel')

    def test_complete_54_and_missing_or_changed_frozen_a9(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path(folder) / 'a9'
            trial = Path(folder) / 'a10'
            previous.mkdir()
            trial.mkdir()
            for folder_path in (previous, trial):
                (folder_path / 'HEAD.txt').write_text('1f3eaa5 full\n')
                (folder_path / 'matrix_sha256.txt').write_text('matching input hashes\n')
            for arm in ARMS:
                for mode in MODES:
                    for policy in POLICIES:
                        (previous / f'{CASE}_{arm}_{mode}_{policy}_1.log').write_text(
                            a9(case=CASE, arm=arm, mode=mode, policy=policy) + '\n'
                        )
                        for tol_tag in TOLERANCES:
                            stem = f'{CASE}_{arm}_{mode}_{tol_tag}_{policy}_1'
                            (trial / f'{stem}.log').write_text(
                                a10(arm, mode, tol_tag, policy) + '\n'
                            )
                            (trial / f'{stem}.time').write_text(
                                'elapsed_s=0.1 peak_rss_kib=8000 exit_status=0\n'
                            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 0)
            self.assertEqual(len((trial / 'accuracy_tolerance_comparison.csv').read_text().splitlines()), 55)

            (trial / 'HEAD.txt').write_text('unexpected\n')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)
            (trial / 'HEAD.txt').write_text('1f3eaa5 full\n')
            (trial / 'island_shell_graph_piecewise_ones_t12_multilevel_1.log').unlink()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)


if __name__ == '__main__':
    unittest.main()
