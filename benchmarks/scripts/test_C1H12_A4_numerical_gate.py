"""Safety regressions for the native-PCG evidence gate."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A4_numerical_gate import (
    ARMS, CASES, audit, parse_record, parse_time, validate_record,
)


def record(
    *, mode="DirectCoarse", aggregation="StrongGraph", basis="JacobiSmoothed",
    status="Converged", iterations=26, true_rel="6e-11", reported="5e-11",
    coarse_dim=512, coarse_bytes=200000, nnz=103104,
):
    values = dict(
        mode=mode, aggregation=aggregation, basis=basis, transfer_values="F64",
        rows=20736, nnz=nnz, coarse_dimension=coarse_dim,
        coarse_factor_bytes=coarse_bytes, status=status, iterations=iterations,
        tol="1e-10", reported_relative=reported, verified_true_relative=true_rel,
        relative_x_error="1e-8", load_ms="2.0", setup_ms="20.0",
        solve_ms="10.0", total_wall_ms="32.0",
    )
    return "C1H12A4|" + "|".join(f"{key}={value}" for key, value in values.items())


class GateTests(unittest.TestCase):
    def test_valid_converged_and_nonconverged(self):
        data = parse_record(record())
        self.assertEqual(validate_record(data, "island_shell", "strong_graph_smoothed")["status"], "Converged")
        slow = parse_record(record(status="MaxIterations", iterations=600, true_rel="2e-8", reported="2e-8"))
        self.assertEqual(validate_record(slow, "island_shell", "strong_graph_smoothed")["iterations"], 600)

    def test_false_convergence_rejected(self):
        with self.assertRaisesRegex(ValueError, "false convergence"):
            validate_record(parse_record(record(true_rel="1e-7")), "island_shell", "strong_graph_smoothed")

    def test_false_max_iterations_rejected(self):
        with self.assertRaisesRegex(ValueError, "true residual passes"):
            validate_record(parse_record(record(status="MaxIterations", iterations=600)), "island_shell", "strong_graph_smoothed")

    def test_honest_breakdown_kept_as_negative_result(self):
        breakdown = parse_record(record(status="Breakdown", iterations=89,
                                        true_rel="9e-5", reported="1e-4"))
        self.assertEqual(validate_record(breakdown, "island_shell", "strong_graph_smoothed")["status"], "Breakdown")

    def test_duplicate_fields_and_lines_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_record(record() + "|mode=DirectCoarse")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_record(record() + "\n" + record())

    def test_nonfinite_cost_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_record(parse_record(record().replace("setup_ms=20.0", "setup_ms=nan")),
                            "island_shell", "strong_graph_smoothed")

    def test_mismatched_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, "coarse configuration"):
            validate_record(parse_record(record(aggregation="Graph")),
                            "island_shell", "strong_graph_smoothed")

    def test_time_parser(self):
        self.assertEqual(parse_time("elapsed_s=0.03 peak_rss_kib=9000 exit_status=0\n")["peak_rss_kib"], 9000)
        with self.assertRaisesRegex(ValueError, "one GNU time row"):
            parse_time("not a time report")

    def test_jacobi_control(self):
        data = parse_record(record(mode="JacobiOnly", coarse_dim=0, coarse_bytes=0,
                                   status="MaxIterations", iterations=600,
                                   true_rel="1e-8", reported="1e-8"))
        self.assertEqual(validate_record(data, "island_shell", "jacobi")["coarse_dimension"], 0)

    def test_full_corpus_success_and_missing_arm_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for case, nnz in CASES.items():
                for arm, (mode, aggregation, basis) in ARMS.items():
                    if arm == "jacobi":
                        sample = record(mode=mode, status="MaxIterations", iterations=600,
                                        true_rel="1e-8", reported="1e-8", coarse_dim=0,
                                        coarse_bytes=0, nnz=nnz)
                    else:
                        sample = record(mode=mode, aggregation=aggregation,
                                        basis=basis, nnz=nnz)
                    stem = f"{case}_{arm}_1"
                    (folder / f"{stem}.log").write_text(sample)
                    (folder / f"{stem}.time").write_text("elapsed_s=0.08 peak_rss_kib=9000 exit_status=0\n")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(folder, 1), 0)
            self.assertEqual(len((folder / "native_coarse_summary.csv").read_text().splitlines()), 22)
            (folder / "island_shell_jacobi_1.log").unlink()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(folder, 1), 1)


if __name__ == "__main__":
    unittest.main()
