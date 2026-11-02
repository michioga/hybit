#!/usr/bin/env python3
"""C1-H3 evidence-only offline gate tests (does not require Rust)."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from C1H3_shadow_policy import AuditError, parse_log, assess


def fixture(residuals, *, jaccards=None, persist3=None, energy=0.9, status='MaxIterations'):
    jaccards = jaccards or [None] + [0.7] * (len(residuals)-1)
    persist3 = persist3 or [0, 0] + [60] * (len(residuals)-2)
    lines = ['C1-H2 ABTM residual topology observer (observation-only)',
             'dimensions         : 100', 'seed fraction      : 0.20000',
             'iteration true_residual active energy_fraction jaccard_prev persist2 persist3 components max_component frontier']
    for i, r in enumerate(residuals):
        j = '-' if jaccards[i] is None else str(jaccards[i])
        lines.append(f'{12+64*i} {r:.9e} 80 {energy:.6f} {j} {0 if i==0 else 70} {persist3[i]} 2 70 10')
    lines += [f'observer status    : {status}', 'PCG invariant      : PASS']
    return '\n'.join(lines)


class ShadowTests(unittest.TestCase):
    def test_spike_then_recovery_does_not_propose_factorization(self):
        audit = parse_log('spike', fixture([0.1, 1e-5, 1e-2, 1e-9, 1e-10], status='Converged'))
        events, summary = assess(audit)
        self.assertEqual(summary['shadow_candidates'], 0)
        self.assertEqual(summary['isolated_bad_windows'], 1)
        self.assertEqual(summary['recovered_spike_windows'], 1)
        self.assertEqual(events[2]['action'], 'isolated_bad_window')

    def test_sustained_persistent_stall_is_only_shadow_candidate(self):
        audit = parse_log('stall', fixture([0.1, 0.09, 0.085, 0.083]))
        events, summary = assess(audit)
        self.assertGreaterEqual(summary['shadow_candidates'], 1)
        self.assertEqual(events[2]['action'], 'shadow_candidate')

    def test_stall_with_migrating_mask_fails_spatial_gate(self):
        audit = parse_log('migration', fixture([0.1, 0.09, 0.085, 0.083],
                                               jaccards=[None, 0.01, 0.01, 0.01]))
        _, summary = assess(audit)
        self.assertEqual(summary['shadow_candidates'], 0)

    def test_falling_residual_never_triggers_on_persistent_hotspot(self):
        audit = parse_log('healthy', fixture([0.1, 0.001, 1e-6, 1e-9], status='Converged'))
        _, summary = assess(audit)
        self.assertEqual(summary['shadow_candidates'], 0)

    def test_malformed_invariant_and_nonfinite_rejected(self):
        data = fixture([0.1, 0.01])
        with self.assertRaises(AuditError):
            parse_log('bad', data.replace('PCG invariant      : PASS','PCG invariant      : FAIL'))
        with self.assertRaises(AuditError):
            parse_log('bad', data.replace('1.000000000e-02', 'nan'))

    def test_threshold_contract(self):
        audit = parse_log('baseline', fixture([0.1, 0.09, 0.085]))
        for rate in [-1.0, 0.0, 1.0, float('nan')]:
            with self.assertRaises(AuditError):
                assess(audit, rate_limit=rate)

if __name__ == '__main__':
    unittest.main()
