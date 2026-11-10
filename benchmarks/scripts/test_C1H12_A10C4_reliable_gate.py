"""Synthetic A10C4 108-way grid, including frozen controls and fail-closed tampering."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10C4_reliable_gate import (
    ARMS, POLICIES, RHS, TOLS, PERIODS, CHECK, FINAL, SESSION, validate,
)


def marker(name: str, fields: dict[str, object]) -> str:
    return name + "|" + "|".join(f"{k}={v}" for k, v in fields.items())


class ReliableGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.a10b, self.a10c3, self.output = (
            self.base / "a10b", self.base / "a10c3", self.base / "a10c4",
        )
        for path in (self.a10b, self.a10c3, self.output):
            path.mkdir()
        old_head = "1f3eaa50612186287a3a0d3e3ce8ed39a74cd755\n"
        new_head = "60104c9f84c099b35605229e2173ca8d1bd7a411\n"
        for path, head in ((self.a10b, old_head), (self.a10c3, new_head), (self.output, new_head)):
            (path / "HEAD.txt").write_text(head)
            (path / "matrix_sha256.txt").write_text("0" * 64 + "  original.mtx\n")
        (self.output / "FROZEN_A10B_HEAD.txt").write_text(old_head)
        (self.output / "FROZEN_A10C3_HEAD.txt").write_text(new_head)
        self._write_frozen()
        self._write_logs()

    @staticmethod
    def hash_rhs(rhs: str) -> str:
        return "cc8b6856aa58926e" if rhs == "Ones" else "a2d63316a180e6b8"

    def _write_frozen(self) -> None:
        fields = ("case", "arm", "policy", "rhs_mode", "rhs_hash", "trace_status",
                  "trace_iterations", "trace_recursive_relative", "trace_true_relative",
                  "trace_compensated_relative", "trace_precond_calls")
        with (self.a10c3 / "trace_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for arm in ARMS:
                for policy in POLICIES.values():
                    for rhs in (*RHS, "IndexSmooth", "SeededRandom"):
                        is_hard = rhs in RHS
                        writer.writerow({"case": "multi_patch", "arm": arm, "policy": policy,
                                         "rhs_mode": rhs, "rhs_hash": self.hash_rhs(rhs),
                                         "trace_status": "FalseConvergence" if is_hard else "Converged",
                                         "trace_iterations": 77 if is_hard else 50,
                                         "trace_recursive_relative": "8e-13",
                                         "trace_true_relative": "3e-10" if is_hard else "8e-13",
                                         "trace_compensated_relative": "3e-10" if is_hard else "8e-13",
                                         "trace_precond_calls": 77 if is_hard else 50})
        (self.a10c3 / "residual_trace.csv").write_text("unchanged frozen checkpoints\n")
        with (self.output / "FROZEN_A10C3_SHA256.txt").open("w") as stream:
            for name in ("trace_comparison.csv", "residual_trace.csv"):
                path = self.a10c3 / name
                stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")

    def _write_logs(self) -> None:
        for arm in ARMS:
            for short_policy, policy in POLICIES.items():
                stem = f"multi_patch_{arm}_{short_policy}_1"
                multi = policy == "Multilevel"
                session = {"case": "multi_patch", "arm": arm, "policy": policy,
                           "rows": 20736, "nnz": 103104, "rhs_count": 2, "tol_count": 3,
                           "period_count": 3, "max_iterations": 1200,
                           "level_dims": "20736,5184,512" if multi else "20736,508",
                           "level_nnz": "103104,22000,4000" if multi else "103104,0",
                           "abtm_words": 100 if multi else 0}
                lines = [marker(SESSION, session)]
                for rhs in RHS:
                    for tol in TOLS:
                        for period in PERIODS:
                            false = period == 0 and tol == 1e-12
                            maximum = 77 if false else 100
                            status = "FalseConvergence" if false else "Converged"
                            terminal_recursive = 8e-13 if false else tol / 2
                            terminal_true = 3e-10 if false else tol / 2
                            fields = {"case": "multi_patch", "arm": arm, "policy": policy,
                                      "rhs_mode": rhs, "rhs_hash": self.hash_rhs(rhs),
                                      "tol": f"{tol:.0e}", "replacement_period": period}
                            steps = sorted({0, 1, maximum, *range(25, maximum + 1, 25)})
                            replaced = 0
                            for k in steps:
                                rec = 1.0 if k == 0 else (terminal_recursive if k == maximum else 1e-3 / (k + 1))
                                true = terminal_true if k == maximum else rec
                                flag = int(period > 0 and k > 0 and k < maximum and k % period == 0)
                                replaced += flag
                                lines.append(marker(CHECK, {**fields, "iteration": k,
                                     "recursive_relative": f"{rec:.15e}",
                                     "true_relative": f"{true:.15e}",
                                     "compensated_relative": f"{true:.15e}",
                                     "gap_ratio": f"{true / rec:.15e}",
                                     "replacement_applied": flag, "precond_calls": k}))
                            lines.append(marker(FINAL, {**fields, "status": status,
                                "iterations": maximum, "recursive_relative": f"{terminal_recursive:.15e}",
                                "true_relative": f"{terminal_true:.15e}",
                                "compensated_relative": f"{terminal_true:.15e}",
                                "precond_calls": maximum, "true_checks": len(steps),
                                "replacements": replaced, "false_events": int(false),
                                "best_true_relative": f"{terminal_true:.15e}",
                                "relative_x_error": "n/a" if rhs == "PointLoad" else "1e-5"}))
                (self.output / f"{stem}.log").write_text("\n".join(lines) + "\n")
                (self.output / f"{stem}.time").write_text("elapsed_s=2.3 peak_rss_kib=12000 exit_status=0\n")

    def test_all_108_complete_with_frozen_no_replacement_controls(self) -> None:
        checks, finals = validate(self.a10c3, self.a10b, self.output)
        self.assertEqual(len(finals), 108)
        self.assertGreater(len(checks), 500)
        self.assertEqual(sum(f["status"] == "FalseConvergence" for f in finals), 12)

    def test_missing_checkpoint_and_bad_replacement_work_rejected(self) -> None:
        path = self.output / "multi_patch_graph_piecewise_two_level_1.log"
        original = path.read_text()
        check = next(x for x in original.splitlines() if x.startswith(CHECK + "|")
                     and "replacement_period=25|" in x and "iteration=25|" in x)
        for changed in (original.replace(check + "\n", "", 1),
                        original.replace(check, check.replace("replacement_applied=1", "replacement_applied=0"), 1)):
            path.write_text(changed)
            with self.assertRaises(ValueError):
                validate(self.a10c3, self.a10b, self.output)
        path.write_text(original)

    def test_forged_success_and_changed_a10c3_control_rejected(self) -> None:
        path = self.output / "multi_patch_graph_piecewise_two_level_1.log"
        original = path.read_text()
        for changed in (original.replace("status=FalseConvergence", "status=Converged", 1),
                        original.replace("iterations=77", "iterations=78", 1)):
            path.write_text(changed)
            with self.assertRaises(ValueError):
                validate(self.a10c3, self.a10b, self.output)

    def test_changed_frozen_reference_and_invalid_time_rejected(self) -> None:
        path = self.a10c3 / "trace_comparison.csv"
        original = path.read_text()
        path.write_text(original.replace("FalseConvergence", "Converged", 1))
        with self.assertRaises(ValueError):
            validate(self.a10c3, self.a10b, self.output)
        path.write_text(original)
        t = self.output / "multi_patch_graph_piecewise_two_level_1.time"
        t.write_text("elapsed_s=2.3 peak_rss_kib=12000 exit_status=1\n")
        with self.assertRaises(ValueError):
            validate(self.a10c3, self.a10b, self.output)


if __name__ == "__main__":
    unittest.main()
