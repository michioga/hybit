"""Fast C1-H10 CSV validation tests, no Rust binary required."""
import unittest
from .C1H10_analyze import summarize


def row(mode, count, repetition, warmup=0):
    return {'mode': mode, 'rhs_count': count, 'repeat': repetition, 'warmup': warmup,
            'wall_ms': 5.0 if mode == 'prepared' else 8.0,
            'analysis_ms': 0.0, 'prepare_ms': 0.0, 'diagnostics_ms': 0.2,
            'factor_ms': 0.0 if mode == 'prepared' else 2.0,
            'probe_ms': 0.1, 'solve_ms': 0.1, 'iterations': 14 * count,
            'converged': count, 'promotions': count,
            'reuse_events': count - 1 if mode == 'prepared' else 0,
            'peak_factor_bytes': 40960,
            'max_true_residual': 1e-14, 'max_forward_error': 1e-13}


class C1H10Tests(unittest.TestCase):
    def test_median_and_reuse(self):
        rows = [row(mode, 2, rep) for rep in range(4) for mode in ('plain', 'fresh', 'prepared')]
        # Plain has no local factor.
        for item in rows:
            if item['mode'] == 'plain':
                item['promotions'] = 0
                item['factor_ms'] = 0.0
        result = summarize(rows)
        self.assertEqual(result[0]['rhs_count'], 2)
        self.assertAlmostEqual(result[0]['prepared_over_fresh'], 5 / 8)
        self.assertEqual(result[0]['prepared_reuse_events'], 1)

    def test_missing_arm_rejected(self):
        rows = [row(mode, 2, rep) for rep in range(3) for mode in ('plain', 'fresh', 'prepared')]
        rows.pop()
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_missing_cache_reuse_rejected(self):
        rows = [row(mode, 2, rep) for rep in range(3) for mode in ('plain', 'fresh', 'prepared')]
        for item in rows:
            if item['mode'] == 'prepared':
                item['reuse_events'] = 0
        with self.assertRaises(ValueError):
            summarize(rows)


if __name__ == '__main__':
    unittest.main()
