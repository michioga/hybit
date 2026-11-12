import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.P2B_gate import parse_fem, parse_zero_probe, gate


def fem_record(status="MaxIterations", iters=60, residual=1e-7):
    return (f"status             : {status}\n"
            f"preconditioner     : Jacobi\n"
            f"iterations         : {iters}\n"
            f"verified residual  : {residual:.6e}\n"
            f"relative x error   : 1e-4\n")


def zero_record(mode, false_status=False):
    rows = []
    for name in ("zero_rhs_zero_x", "zero_rhs_nonzero_x_rel_only",
                 "zero_rhs_nonzero_x_abs", "manufactured_nonzero_rhs"):
        status = "MaxIterations" if name == "zero_rhs_nonzero_x_rel_only" else "Converged"
        true_norm = 1e-7 if name == "zero_rhs_nonzero_x_rel_only" else 0.0
        target = 1e-10 if name == "zero_rhs_nonzero_x_abs" else 1e-20
        if false_status and name == "zero_rhs_nonzero_x_rel_only":
            status = "Converged"
        rows.append(
            f"P2B_CASE|name={name}|feature={mode}|status={status}|iterations=3|"
            f"max_iters=96|true_norm={true_norm}|target={target}|rhs_norm=0|"
            "initial_norm=1|x_error=1\n"
        )
    return "".join(rows)


class P2BGateTests(unittest.TestCase):
    def test_rejects_false_fem_convergence(self):
        with self.assertRaisesRegex(ValueError, "FALSE CONVERGENCE"):
            parse_fem(fem_record("Converged", 2, 1e-4),
                      "elapsed_s=0.01 peak_rss_kib=2000 exit_status=0", 600, 1e-10)

    def test_zero_rhs_nonzero_x_is_observation_not_success(self):
        results = parse_zero_probe(zero_record("off"), "off")
        self.assertEqual(results["zero_rhs_nonzero_x_rel_only"]["status"], "MaxIterations")
        with self.assertRaisesRegex(ValueError, "FALSE ZERO-RHS"):
            parse_zero_probe(zero_record("on", True), "on")

    def test_paired_gate_counts_only_natural_rescue(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "cases_effective.tsv").write_text("hard\t/a\t/b\t/c\n")
            (p / "hard_off.log").write_text(fem_record())
            (p / "hard_on.log").write_text(fem_record("Converged", 70, 1e-12))
            for mode in ("off", "on"):
                (p / f"hard_{mode}.time").write_text("elapsed_s=0.12 peak_rss_kib=4321 exit_status=0\n")
                (p / f"zero_{mode}.log").write_text(zero_record(mode))
            result = gate(p, 1e-10, 600)
            self.assertEqual(result["natural_rescues"], 1)
            self.assertEqual(result["certified_convergence_regressions"], 0)
            self.assertTrue((p / "p2b_comparison.csv").exists())

    def test_paired_gate_rejects_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "cases_effective.tsv").write_text("hard\t/a\t/b\t/c\n")
            (p / "hard_off.log").write_text(fem_record("Converged", 30, 1e-12))
            (p / "hard_on.log").write_text(fem_record())
            for mode in ("off", "on"):
                (p / f"hard_{mode}.time").write_text("elapsed_s=0.12 peak_rss_kib=4321 exit_status=0\n")
                (p / f"zero_{mode}.log").write_text(zero_record(mode))
            with self.assertRaisesRegex(ValueError, "regressions"):
                gate(p, 1e-10, 600)


if __name__ == "__main__":
    unittest.main()
