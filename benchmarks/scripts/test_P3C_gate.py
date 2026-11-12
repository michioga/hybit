"""P3-C evidence parser negative tests; never invent measured Rust results."""
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.P3C_gate import EXPECTED, GateError, evaluate, parse


def valid_group(on: bool):
    rows = {}
    for name, expected in EXPECTED.items():
        admitted = expected == "accept" or (not on and name == "nonsym_rot_1e8")
        row = {
            "name": name,
            "expected": expected,
            "observed": "accept" if admitted else "reject",
            "full_relres": "1e-14" if admitted else "0",
            "full_xerror": "1e-7" if admitted else "0",
            "factor_bytes": str(3000 + (192 if on else 0)) if admitted else "0",
            "estimated_bytes": str(3000 + (192 if on else 0)) if admitted else "0",
            "full_positive": "true" if expected == "accept" else "na",
            "overlap_positive": "true" if expected == "accept" else "na",
            "jacobi_status": "Converged" if expected == "accept" else "NA",
            "jacobi_iters": "8" if expected == "accept" else "0",
            "jacobi_true_relres": "1e-12" if expected == "accept" else "0",
            "overlap_status": "Converged" if expected == "accept" else "NA",
            "overlap_iters": "5" if expected == "accept" else "0",
            "overlap_true_relres": "1e-13" if expected == "accept" else "0",
            "full_status": "Converged" if expected == "accept" else "NA",
            "full_iters": "1" if expected == "accept" else "0",
            "full_true_relres": "1e-14" if expected == "accept" else "0",
        }
        rows[name] = row
    return rows


def log_text(rows):
    cases = ["P3C_CASE|" + "|".join(f"{k}={v}" for k, v in row.items()) for row in rows.values()]
    cases.append("P3C_SUMMARY|cases=8|nodes=4|dofs_per_node=6|reference_pcg_max_iters=96|reference_pcg_tol=1e-10")
    return "\n".join(cases) + "\n"


class P3CGateTests(unittest.TestCase):
    def setUp(self):
        self.off = valid_group(on=False)
        self.on = valid_group(on=True)

    def test_accepts_legacy_misclassification_as_observation(self):
        summary = evaluate(self.off, self.on)
        self.assertEqual(summary["feature_off_contract_discrepancies"], 1)
        self.assertEqual(summary["feature_on_contract_discrepancies"], 0)

    def test_rejects_feature_on_false_accept(self):
        self.on["nonsym_rot_1e8"]["observed"] = "accept"
        with self.assertRaises(GateError):
            evaluate(self.off, self.on)

    def test_rejects_false_convergence(self):
        self.on["beam_rot_1e8"]["full_true_relres"] = "0.3"
        with self.assertRaises(GateError):
            evaluate(self.off, self.on)

    def test_rejects_storage_discrepancy(self):
        self.on["balanced_6dof"]["factor_bytes"] = "9999"
        with self.assertRaises(GateError):
            evaluate(self.off, self.on)

    def test_rejects_nonpositive_preconditioner(self):
        self.on["beam_rot_1e4"]["overlap_positive"] = "false"
        with self.assertRaises(GateError):
            evaluate(self.off, self.on)

    def test_rejects_nonfinite(self):
        self.on["beam_rot_1e4"]["full_xerror"] = "nan"
        with self.assertRaises(GateError):
            evaluate(self.off, self.on)

    def test_rejects_iteration_overshoot(self):
        self.on["balanced_6dof"]["jacobi_iters"] = "97"
        with self.assertRaises(GateError):
            evaluate(self.off, self.on)

    def test_parser_accepts_complete_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.log"
            p.write_text(log_text(self.on), encoding="utf-8")
            self.assertEqual(len(parse(p)), 8)

    def test_parser_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.log"
            lines = log_text(self.on).splitlines()
            lines.insert(1, lines[0])
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(GateError):
                parse(p)

    def test_parser_rejects_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.log"
            lines = log_text(self.on).splitlines()
            p.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
            with self.assertRaises(GateError):
                parse(p)


if __name__ == "__main__":
    unittest.main()
