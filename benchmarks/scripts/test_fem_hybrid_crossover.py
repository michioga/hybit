"""Numerical-evidence guards for the C1-D offline crossover evaluator."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).with_name("fem_hybrid_crossover.py")
SPEC = importlib.util.spec_from_file_location("fem_hybrid_crossover", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def sample_pairs(hybrid_ms, baseline_ms=None, escalations=1):
    if baseline_ms is None:
        baseline_ms = [1000.0] * len(hybrid_ms)
    rows = []
    for k, (a, b) in enumerate(zip(baseline_ms, hybrid_ms), 1):
        rows.extend((
            {"phase": "measure", "pair": k, "arm": "observe", "pass": True,
             "hybit_wall_ms": a, "hybit_escalations": 0},
            {"phase": "measure", "pair": k, "arm": "hybrid", "pass": True,
             "hybit_wall_ms": b, "hybit_escalations": escalations},
        ))
    return rows


class CrossoverTests(unittest.TestCase):
    def test_strong_regression(self):
        out = module.paired_assessment(sample_pairs([1200, 1230, 1190, 1250, 1210, 1220, 1180, 1200, 1240]))
        self.assertEqual(out["classification"], "measured_regression_in_this_matrix")
        self.assertGreater(out["bootstrap_95pct"][0], 0.05)

    def test_strong_benefit(self):
        out = module.paired_assessment(sample_pairs([700, 720, 680, 710, 740, 750, 730, 690, 700]))
        self.assertEqual(out["classification"], "potential_benefit_in_this_matrix")
        self.assertLess(out["bootstrap_95pct"][1], -0.05)

    def test_zero_escalation_cannot_be_benefit(self):
        out = module.paired_assessment(sample_pairs([850] * 9, escalations=0))
        self.assertEqual(out["classification"], "mixed_or_no_activation")

    def test_mixed_outcomes_are_inconclusive(self):
        out = module.paired_assessment(sample_pairs([880, 920, 1080, 1100, 980, 1020, 990, 1010, 1000]))
        self.assertEqual(out["classification"], "inconclusive")

    def test_nonconvergence_invalidates_assessment(self):
        rows = sample_pairs([1200] * 9)
        rows[3]["pass"] = False
        self.assertEqual(module.paired_assessment(rows)["classification"], "insufficient_evidence")

    def test_insufficient_n_never_claims_superiority(self):
        self.assertEqual(module.paired_assessment(sample_pairs([800]*4))["classification"],
                         "insufficient_evidence")

    def test_duplicate_measurement_is_rejected(self):
        rows = sample_pairs([800] * 9)
        with self.assertRaises(ValueError):
            module.paired_assessment(rows + [rows[0]])

    def test_invalid_or_missing_time_is_not_accepted(self):
        rows = sample_pairs([800] * 9)
        rows[0]["hybit_wall_ms"] = float("nan")
        self.assertEqual(module.paired_assessment(rows)["classification"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
