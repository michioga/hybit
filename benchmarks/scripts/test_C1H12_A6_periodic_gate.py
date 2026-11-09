"""Negative and full-matrix synthetic evidence tests for A6 periodic PCG."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, CASES
from benchmarks.scripts.C1H12_A6_periodic_gate import (
    audit, parse_a6_record, verify_record,
)
from benchmarks.scripts.test_C1H12_A4_numerical_gate import record
from benchmarks.scripts.test_C1H12_A5_certification_gate import a5


def a6(*, period=16, true_rel="6e-11", iterations=26, checks=7, restarts=0,
       proactive=0, rejected=0, gap="1.1", check_ms="0.25"):
    return (
        "C1H12A6|policy=PeriodicCertifiedRestart"
        f"|check_every={period}|trigger_ratio=2.000000|true_checks={checks}"
        f"|periodic_checks={checks}|restarts={restarts}"
        f"|proactive_restarts={proactive}|rejected_recursive_convergences={rejected}"
        f"|peak_gap_ratio={gap}|check_ms={check_ms}|final_true_relative={true_rel}"
        f"|global_iterations={iterations}|status=Converged"
    )


class A6GateTests(unittest.TestCase):
    def test_valid_periodic_and_decomposition(self):
        row = parse_a6_record(a6(restarts=2, proactive=1, rejected=1), 16)
        self.assertEqual(row["restarts"], 2)
        self.assertEqual(row["true_checks"], 7)

    def test_bad_period_duplicate_nan_and_counters(self):
        for log, pattern in (
            (a6(), "interval"),
            (a6() + "|true_checks=8", "duplicate"),
            (a6(gap="nan"), "invalid A6"),
            (a6(check_ms="inf"), "invalid A6"),
            (a6(restarts=1), "restart count"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                parse_a6_record(log, 32 if pattern == "interval" else 16)

    def test_original_csr_certificate_must_match(self):
        log = record() + "\n" + a6()
        timing = "elapsed_s=0.1 peak_rss_kib=9000 exit_status=0\n"
        self.assertEqual(
            verify_record(log, timing, "island_shell", "strong_graph_smoothed", "p16")
            ["detail"]["periodic_checks"], 7,
        )
        with self.assertRaisesRegex(ValueError, "true residual disagree"):
            verify_record(log.replace("final_true_relative=6e-11", "final_true_relative=2e-8"),
                          timing, "island_shell", "strong_graph_smoothed", "p16")
        with self.assertRaisesRegex(ValueError, "false convergence"):
            verify_record(log.replace("verified_true_relative=6e-11", "verified_true_relative=2e-8"),
                          timing, "island_shell", "strong_graph_smoothed", "p16")

    def test_historical_head_can_differ_but_hashes_cannot(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous, current = Path(tmp) / "a5", Path(tmp) / "a6"
            previous.mkdir()
            current.mkdir()
            (previous / "HEAD.txt").write_text("4593c20-full-placeholder\n")
            (current / "HEAD.txt").write_text("1f3eaa5-full-placeholder\n")
            for folder in (previous, current):
                (folder / "matrix_sha256.txt").write_text("verified-hashes\n")
            for case, nnz in CASES.items():
                for arm, (mode, aggregate, basis) in ARMS.items():
                    kwargs = dict(
                        mode=mode, aggregation=aggregate or "Graph",
                        basis=basis or "JacobiSmoothed", nnz=nnz,
                        coarse_dim=0 if arm == "jacobi" else 512,
                        coarse_bytes=0 if arm == "jacobi" else 200000,
                    )
                    for policy in ("a5", "p16", "p32"):
                        stem = f"{case}_{arm}_{policy}_1"
                        description = a5(true_rel="6e-11", iterations=26, checks=1,
                                         restarts=0, rejected=0) if policy == "a5" else a6(
                                             period=int(policy[1:]), true_rel="6e-11",
                                             iterations=26, checks=2,
                                         )
                        (current / f"{stem}.log").write_text(
                            record(**kwargs) + "\n" + description + "\n"
                        )
                        (current / f"{stem}.time").write_text(
                            "elapsed_s=0.1 peak_rss_kib=9000 exit_status=0\n"
                        )
                    (previous / f"{case}_{arm}_1.log").write_text(
                        record(**kwargs) + "\n" + a5(true_rel="6e-11", iterations=26,
                                                       checks=1, restarts=0, rejected=0)
                    )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, current), 0)
            self.assertEqual(len((current / "periodic_pcg_comparison.csv").read_text().splitlines()), 64)
            (current / "matrix_sha256.txt").write_text("incorrect-hashes\n")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, current), 1)
            (current / "matrix_sha256.txt").write_text("verified-hashes\n")
            (current / "island_shell_graph_smoothed_p16_1.log").unlink()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit(previous, current), 1)


if __name__ == "__main__":
    unittest.main()
