"""Test P3-B status, residual, factor bytes, and observed forward accuracy."""
import unittest
from benchmarks.scripts.P3B_gate import evaluate
from benchmarks.scripts.P3A_gate import EXPECTED, EXPECTED_SCALE


def records(mode, *, bad_case=None, bad_field=None):
    rows = []
    mismatches = {"spd_small_1e16", "spd_congruence", "nonsym_small_1e12"}
    for name, expected in EXPECTED.items():
        observed = expected
        if mode == "off" and name in mismatches:
            observed = "reject" if expected == "accept" else "accept"
        residual = 0.0 if observed == "reject" else 1e-14
        forward = 0.0 if observed == "reject" else 1e-12
        if name == "spd_congruence" and observed == "accept":
            forward = 0.03  # Extremely ill-conditioned forward error is recorded.
        factors = 0 if observed == "reject" else (216 if mode == "off" else 248)
        if name == bad_case:
            if bad_field == "observed":
                observed = "reject" if observed == "accept" else "accept"
                factors = 0 if observed == "reject" else 248
            elif bad_field == "residual":
                residual = 1e-2
            elif bad_field == "memory":
                factors += 8
            elif bad_field == "forward":
                forward = 0.2
        rows.append(
            f"P3A_CASE|name={name}|expected={expected}|observed={observed}|"
            f"relative_residual={residual:.17e}|relative_x_error={forward:.17e}|"
            f"factor_bytes={factors}|mathematical_spd={str(expected == 'accept').lower()}|"
            f"scale={EXPECTED_SCALE[name]:.17e}\n"
        )
    discrepancies = 3 if mode == "off" else (1 if bad_field == "observed" else 0)
    rows.append(f"P3A_SUMMARY|cases=7|contract_discrepancies={discrepancies}\n")
    return "".join(rows)


class P3BGateTests(unittest.TestCase):
    def test_expected_baseline_and_equilibrated_result(self):
        rows, result = evaluate(records("off"), records("on"))
        self.assertEqual(result["feature_off_contract_discrepancies"], 3)
        self.assertEqual(result["feature_on_contract_discrepancies"], 0)
        self.assertEqual(len(rows), 7)
        self.assertEqual(result["ill_conditioned_congruence_forward_error"], 0.03)

    def test_rejects_bad_on_acceptance(self):
        with self.assertRaisesRegex(ValueError, "unresolved classification"):
            evaluate(records("off"), records("on", bad_case="spd_small_1e16", bad_field="observed"))

    def test_rejects_bad_residual(self):
        with self.assertRaisesRegex(ValueError, "true residual"):
            evaluate(records("off"), records("on", bad_case="spd_unit", bad_field="residual"))

    def test_rejects_bad_factor_storage(self):
        with self.assertRaisesRegex(ValueError, "factor bytes"):
            evaluate(records("off"), records("on", bad_case="spd_unit", bad_field="memory"))

    def test_forward_error_observable_but_bounded_on_well_scaled(self):
        with self.assertRaisesRegex(ValueError, "forward error"):
            evaluate(records("off"), records("on", bad_case="spd_unit", bad_field="forward"))

    def test_rejects_default_feature_change(self):
        with self.assertRaisesRegex(ValueError, "baseline changed"):
            evaluate(records("on"), records("on"))


if __name__ == "__main__":
    unittest.main()
