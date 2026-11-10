"""Synthetic 24-RHS traces plus provenance and numerical fail-closed checks."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10C3_trace_gate import (
    ARMS, POLICIES, RHS_NATIVE, SESSION, TRACE, FINAL, TRACE_EVERY,
    MAX_ITER, check_group, frozen_baseline, parse_rows, validate,
)


def marker(prefix: str, fields: dict[str, object]) -> str:
    return prefix + "|" + "|".join(f"{k}={v}" for k, v in fields.items())


class A10C3GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.a10b = self.base / "a10b"
        self.a10c2 = self.base / "a10c2"
        self.evidence = self.base / "a10c3"
        for path in (self.a10b, self.a10c2, self.evidence):
            path.mkdir()
        old_head = "1f3eaa50612186287a3a0d3e3ce8ed39a74cd755\n"
        new_head = "60104c9f84c099b35605229e2173ca8d1bd7a411\n"
        matrix_hash = "0" * 64 + "  c1g_p1_multi_patch_n144.mtx\n"
        (self.a10b / "HEAD.txt").write_text(old_head)
        (self.a10c2 / "HEAD.txt").write_text(new_head)
        (self.evidence / "HEAD.txt").write_text(new_head)
        (self.evidence / "FROZEN_A10B_HEAD.txt").write_text(old_head)
        (self.evidence / "FROZEN_A10C2_HEAD.txt").write_text(new_head)
        for path in (self.a10b, self.a10c2, self.evidence):
            (path / "matrix_sha256.txt").write_text(matrix_hash)
        (self.a10c2 / "spectral_ritz.csv").write_text("frozen spectral data\n")
        (self.a10c2 / "coarse_projection.csv").write_text("frozen projection data\n")
        self._write_frozen_baseline()
        self._write_traces()

    @staticmethod
    def fingerprint(rhs: str) -> str:
        return f"{list(RHS_NATIVE.values()).index(rhs) + 1:016x}"

    @staticmethod
    def outcome(rhs: str) -> tuple[str, int, str]:
        if rhs in ("Ones", "PointLoad"):
            return ("MaxIterations", MAX_ITER, "2e-7")
        return ("Converged", 100, "8e-14")

    def _write_frozen_baseline(self) -> None:
        path = self.a10b / "prepared_reuse_comparison.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("case", "arm", "policy", "rhs_mode", "status",
                                      "rhs_hash", "reuse_mode", "iterations"),
            )
            writer.writeheader()
            for case in ("island_shell", "multi_patch", "rotated_anisotropy"):
                for arm in ARMS:
                    for policy in POLICIES.values():
                        for mode in RHS_NATIVE.values():
                            status, iterations, _ = self.outcome(mode) if case == "multi_patch" else ("Converged", 100, "8e-14")
                            for lifecycle in ("fresh", "reuse"):
                                for _ in range(5):
                                    writer.writerow({"case": case, "arm": arm, "policy": policy,
                                                     "rhs_mode": mode, "status": status,
                                                     "rhs_hash": self.fingerprint(mode),
                                                     "reuse_mode": lifecycle, "iterations": iterations})

    def _write_traces(self) -> None:
        for arm in ARMS:
            for short_policy, policy in POLICIES.items():
                stem = f"multi_patch_{arm}_{short_policy}_1"
                multi = policy == "Multilevel"
                session = {"case": "multi_patch", "arm": arm, "policy": policy,
                           "rows": 20736, "nnz": 103104, "tol": "1e-12",
                           "max_iterations": 1200, "trace_every": TRACE_EVERY,
                           "rhs_count": 4, "levels": 3 if multi else 2,
                           "level_dims": "20736,5184,512" if multi else "20736,508",
                           "level_nnz": "103104,22000,4000" if multi else "103104,0",
                           "abtm_words": 123 if multi else 0}
                lines = [marker(SESSION, session)]
                for mode in RHS_NATIVE.values():
                    outcome, maximum, final_true = self.outcome(mode)
                    rhs_hash = self.fingerprint(mode)
                    common = {"case": "multi_patch", "arm": arm, "policy": policy,
                              "rhs_mode": mode, "rhs_hash": rhs_hash}
                    steps = sorted({0, 1, maximum, *range(TRACE_EVERY, maximum + 1, TRACE_EVERY)})
                    for step in steps:
                        recursive = 1.0 if step == 0 else (float(final_true) if step == maximum else 1e-3 / (step + 1))
                        true = recursive if step in (0, maximum) else recursive * 1.05
                        comp = true
                        lines.append(marker(TRACE, {**common, "iteration": step,
                                                    "recursive_relative": f"{recursive:.15e}",
                                                    "true_relative": f"{true:.15e}",
                                                    "compensated_relative": f"{comp:.15e}",
                                                    "gap_ratio": f"{true / recursive:.15e}",
                                                    "precond_calls": step}))
                    lines.append(marker(FINAL, {
                        **common, "trace_status": outcome, "trace_iterations": maximum,
                        "trace_recursive_relative": f"{float(final_true):.15e}",
                        "trace_true_relative": f"{float(final_true):.15e}",
                        "trace_compensated_relative": f"{float(final_true):.15e}",
                        "trace_precond_calls": maximum,
                        "trace_relative_x_error": "n/a" if mode == "PointLoad" else "1e-5",
                        "baseline_status": outcome, "baseline_iterations": maximum,
                        "baseline_true_relative": final_true,
                        "baseline_reported_relative": final_true,
                        "baseline_compensated_relative": final_true,
                        "baseline_true_checks": 1, "baseline_restarts": 0,
                    }))
                (self.evidence / f"{stem}.log").write_text("\n".join(lines) + "\n")
                (self.evidence / f"{stem}.time").write_text("elapsed_s=2.10 peak_rss_kib=12000 exit_status=0\n")

    def test_complete_24_with_original_csr_checkpoints(self) -> None:
        traces, finals = validate(self.a10b, self.a10c2, self.evidence)
        self.assertEqual(len(finals), 24)
        self.assertGreater(len(traces), 500)
        self.assertEqual(sum(f["baseline_status"] == "Converged" for f in finals), 12)

    def test_reject_missing_duplicate_and_tampered_checkpoints(self) -> None:
        path = self.evidence / "multi_patch_graph_piecewise_two_level_1.log"
        original = path.read_text()
        first = next(line for line in original.splitlines() if line.startswith(TRACE + "|") and "iteration=25|" in line)
        for changed in (original.replace(first + "\n", "", 1), original.replace(first, first + "\n" + first, 1),
                        original.replace(first, first.replace("gap_ratio=1.050000000000000e+00", "gap_ratio=1.000000000000000e+00"), 1)):
            path.write_text(changed)
            with self.assertRaises(ValueError):
                validate(self.a10b, self.a10c2, self.evidence)
        path.write_text(original)

    def test_reject_false_convergence_and_changed_frozen_work(self) -> None:
        path = self.evidence / "multi_patch_graph_piecewise_two_level_1.log"
        original = path.read_text()
        # A trace cannot be labelled converged with an original-CSR residual > tolerance.
        changed = original.replace("trace_status=MaxIterations", "trace_status=Converged", 1)
        path.write_text(changed)
        with self.assertRaises(ValueError):
            validate(self.a10b, self.a10c2, self.evidence)
        changed = original.replace("baseline_iterations=1200", "baseline_iterations=1199", 1)
        path.write_text(changed)
        with self.assertRaises(ValueError):
            validate(self.a10b, self.a10c2, self.evidence)

    def test_reject_provenance_and_nonzero_gnu_time(self) -> None:
        path = self.evidence / "FROZEN_A10C2_HEAD.txt"
        original = path.read_text()
        path.write_text(original.replace("60104c9", "70104c9"))
        with self.assertRaises(ValueError):
            validate(self.a10b, self.a10c2, self.evidence)
        path.write_text(original)
        time_path = self.evidence / "multi_patch_graph_piecewise_two_level_1.time"
        time_path.write_text("elapsed_s=2.1 peak_rss_kib=12000 exit_status=1\n")
        with self.assertRaises(ValueError):
            validate(self.a10b, self.a10c2, self.evidence)


if __name__ == "__main__":
    unittest.main()
