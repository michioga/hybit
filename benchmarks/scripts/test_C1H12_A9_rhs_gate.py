"""Positive, negative and complete 72-case A9 evidence checks."""

import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A4_numerical_gate import CASES
from benchmarks.scripts.C1H12_A9_rhs_gate import ARMS, MODES, POLICIES, audit, parse_a9
from benchmarks.scripts.test_C1H12_A8_multilevel_gate import a8


def a9(case='island_shell', arm='strong_graph_smoothed', mode='index_smooth',
       policy='multilevel', *, verified='6e-11'):
    original = a8(case=case, arm=arm, policy=policy, verified=verified)
    if mode == 'point_load':
        original = original.replace('relative_x_error=2e-8', 'relative_x_error=n/a')
    fingerprint = hashlib.sha256(f'{case}/{mode}'.encode()).hexdigest()[:16]
    return (original.replace('C1H12A8|', 'C1H12A9|', 1)
            + f'|rhs_mode={MODES[mode]}|rhs_hash={fingerprint}|rhs_norm=12.5')


class A9GateTests(unittest.TestCase):
    def test_valid_four_rhs_modes(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                parsed = parse_a9(a9(mode=mode), 'island_shell',
                                  'strong_graph_smoothed', mode, 'multilevel')
                self.assertEqual(parsed['rhs_mode'], mode)
                self.assertEqual(parsed['relative_x_error'] == 'n/a', mode == 'point_load')

    def test_false_convergence_and_bad_schema_rejected(self):
        examples = (
            (a9(verified='5e-8'), 'false convergence'),
            (a9() + '|rhs_hash=0123456789abcdef', 'duplicate'),
            (a9().replace('rhs_norm=12.5', 'rhs_norm=inf'), 'nonfinite'),
            (a9().replace('rhs_hash=', 'rhs_hash=x'), 'bad RHS fingerprint'),
            (a9().replace('rhs_mode=IndexSmooth', 'rhs_mode=PointLoad'), 'RHS mode mismatch'),
            (a9(mode='point_load').replace('relative_x_error=n/a',
                                           'relative_x_error=0.0'), 'point-load claims'),
            (a9().replace('relative_x_error=2e-8', 'relative_x_error=n/a'),
             'manufactured RHS'),
        )
        for sample, message in examples:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_a9(sample, 'island_shell', 'strong_graph_smoothed',
                         'index_smooth' if 'point-load claims' not in message else 'point_load',
                         'multilevel')

    def test_full_72_detects_missing_provenance_and_rhs_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path(folder) / 'a8'
            trial = Path(folder) / 'a9'
            previous.mkdir()
            trial.mkdir()
            for base in (previous, trial):
                (base / 'HEAD.txt').write_text('1f3eaa5-full\n')
                (base / 'matrix_sha256.txt').write_text('three matching input hashes\n')
            for case in CASES:
                for arm in ARMS:
                    for policy in POLICIES:
                        (previous / f'{case}_{arm}_{policy}_1.log').write_text(
                            a8(case=case, arm=arm, policy=policy) + '\n'
                        )
                        for mode in MODES:
                            stem = f'{case}_{arm}_{mode}_{policy}_1'
                            (trial / f'{stem}.log').write_text(
                                a9(case, arm, mode, policy) + '\n'
                            )
                            (trial / f'{stem}.time').write_text(
                                'elapsed_s=0.2 peak_rss_kib=10000 exit_status=0\n'
                            )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 0)
            self.assertEqual(len((trial / 'rhs_diversity_comparison.csv').read_text().splitlines()), 73)

            mismatch = trial / 'multi_patch_graph_piecewise_point_load_multilevel_1.log'
            original = mismatch.read_text()
            mismatch.write_text(original.replace('rhs_hash=', 'rhs_hash=f'))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)
            mismatch.write_text(original)

            (trial / 'HEAD.txt').write_text('wrong revision\n')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)
            (trial / 'HEAD.txt').write_text('1f3eaa5-full\n')

            (trial / 'island_shell_strong_graph_smoothed_ones_two_level_1.time').unlink()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, trial), 1)


if __name__ == '__main__':
    unittest.main()
