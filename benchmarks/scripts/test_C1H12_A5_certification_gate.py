"""Regression tests: compare A4's honest false positives to A5 certificates."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, CASES
from benchmarks.scripts.C1H12_A5_certification_gate import audit, parse_a5_record
from benchmarks.scripts.test_C1H12_A4_numerical_gate import record


def a5(*, true_rel="4e-11", iterations=100, checks=2, restarts=1, rejected=1):
    return (
        f"C1H12A5|policy=CertifiedRestart|true_checks={checks}|restarts={restarts}"
        f"|rejected_recursive_convergences={rejected}|final_true_relative={true_rel}"
        f"|global_iterations={iterations}|status=Converged"
    )


class A5GateTests(unittest.TestCase):
    def test_valid_restarted_certificate(self):
        value = parse_a5_record(a5())
        self.assertEqual(value["restarts"], 1)
        self.assertEqual(value["true_checks"], 2)

    def test_reject_inconsistent_restarts(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            parse_a5_record(a5(checks=1, restarts=2))

    def test_reject_duplicate_or_nonfinite_certificate(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_a5_record(a5() + "|true_checks=5")
        with self.assertRaisesRegex(ValueError, "invalid final true residual"):
            parse_a5_record(a5(true_rel="nan"))

    def test_a4_failed_original_and_a5_certified_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "a4"
            certified = Path(tmp) / "a5"
            baseline.mkdir()
            certified.mkdir()
            for folder in (baseline, certified):
                (folder / "HEAD.txt").write_text("4593c20\n")
                (folder / "matrix_sha256.txt").write_text("hash-placeholder\n")
            for case, nnz in CASES.items():
                for arm, (mode, aggregation, basis) in ARMS.items():
                    stem = f"{case}_{arm}_1.log"
                    row_kwargs = dict(mode=mode, aggregation=aggregation or "Graph",
                                      basis=basis or "JacobiSmoothed", nnz=nnz,
                                      iterations=100)
                    (baseline / stem).write_text(record(**row_kwargs, true_rel="4e-10"))
                    (certified / stem).write_text(
                        record(**row_kwargs, true_rel="4e-11") + "\n" + a5()
                    )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(baseline, certified), 0)
            self.assertEqual(len((certified / "certified_restart_comparison.csv").read_text().splitlines()), 22)
            (certified / "matrix_sha256.txt").write_text("modified\n")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(baseline, certified), 1)


if __name__ == "__main__":
    unittest.main()
