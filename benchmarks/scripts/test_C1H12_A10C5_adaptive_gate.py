"""Synthetic A10C5 72-condition original-CSR cooldown and stagnation evidence."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10C5_adaptive_gate import (
    ARMS, POLICIES, RHS, TOLS, COOLDOWNS, CHECK, FINAL, SESSION, validate,
)


def marker(tag: str, fields: dict[str, object]) -> str:
    return tag + "|" + "|".join(f"{key}={value}" for key, value in fields.items())


class AdaptiveGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.a10b, self.a10c4, self.output = base / "a10b", base / "a10c4", base / "a10c5"
        for path in (self.a10b, self.a10c4, self.output):
            path.mkdir()
        head = "60104c9f84c099b35605229e2173ca8d1bd7a411\n"
        old = "1f3eaa50612186287a3a0d3e3ce8ed39a74cd755\n"
        for directory, value in ((self.a10b, old), (self.a10c4, head), (self.output, head)):
            (directory / "HEAD.txt").write_text(value)
            (directory / "matrix_sha256.txt").write_text("0" * 64 + "  original.mtx\n")
        (self.output / "FROZEN_A10C4_HEAD.txt").write_text(head)
        (self.output / "FROZEN_A10B_HEAD.txt").write_text(old)
        self._frozen()
        self._logs(stalled=False)

    @staticmethod
    def hash_rhs(rhs: str) -> str:
        return "cc8b6856aa58926e" if rhs == "Ones" else "a2d63316a180e6b8"

    def _frozen(self) -> None:
        fields = ("case", "arm", "policy", "rhs_mode", "rhs_hash", "tol",
                  "replacement_period", "status", "iterations", "replacements",
                  "best_true_relative")
        with (self.a10c4 / "reliable_update_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for arm in ARMS:
                for policy in POLICIES.values():
                    for rhs in RHS:
                        for tol in TOLS:
                            for period in (0, *COOLDOWNS):
                                writer.writerow({"case": "multi_patch", "arm": arm, "policy": policy,
                                    "rhs_mode": rhs, "rhs_hash": self.hash_rhs(rhs),
                                    "tol": f"{tol:.0e}", "replacement_period": period,
                                    "status": "MaxIterations", "iterations": 1200,
                                    "replacements": 47, "best_true_relative": "2e-10"})
        (self.a10c4 / "reliable_residual_trace.csv").write_text("original A10C4 checkpoints\n")
        with (self.output / "FROZEN_A10C4_SHA256.txt").open("w") as stream:
            for name in ("reliable_update_comparison.csv", "reliable_residual_trace.csv"):
                p = self.a10c4 / name
                stream.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n")

    def _logs(self, stalled: bool) -> None:
        for arm in ARMS:
            for short_policy, policy in POLICIES.items():
                stem = f"multi_patch_{arm}_{short_policy}_1"
                multi = policy == "Multilevel"
                session = {"case": "multi_patch", "arm": arm, "policy": policy,
                    "rows": 20736, "nnz": 103104, "rhs_count": 2, "tol_count": 3,
                    "cooldown_count": 2, "stagnation_window": 200,
                    "progress_factor": "9.0e-1", "threshold_spacing": 5,
                    "max_iterations": 1200,
                    "level_dims": "20736,5184,512" if multi else "20736,508",
                    "level_nnz": "103104,22000,4000" if multi else "103104,0",
                    "abtm_words": 40 if multi else 0}
                lines = [marker(SESSION, session)]
                for rhs in RHS:
                    for tol in TOLS:
                        for cd in COOLDOWNS:
                            fields = {"case": "multi_patch", "arm": arm, "policy": policy,
                                "rhs_mode": rhs, "rhs_hash": self.hash_rhs(rhs),
                                "tol": f"{tol:.0e}", "cooldown": cd}
                            steps = [0, 1, *range(25, 226, 25)] if stalled else [0, 1, 25, 50]
                            progress, prev_best, last_replacement = 0, 1.0, 0
                            replaced, deferred = 0, 0
                            for k in steps:
                                res = (1.0 if k == 0 else .2 if k == 1 else
                                       .001 if stalled or k == 25 else tol / 2)
                                if k > 0 and res < prev_best * .9:
                                    progress = k
                                    prev_best = res
                                certified = res <= tol
                                stop = stalled and k == steps[-1]
                                wants = k > 0 and k % 25 == 0
                                eligible = k - last_replacement >= cd
                                apply = int(wants and eligible and not certified and not stop)
                                skip = int(wants and not eligible and not certified and not stop)
                                replaced += apply
                                deferred += skip
                                if apply:
                                    last_replacement = k
                                lines.append(marker(CHECK, {**fields, "iteration": k,
                                    "recursive_relative": f"{res:.15e}",
                                    "true_relative": f"{res:.15e}",
                                    "compensated_relative": f"{res:.15e}",
                                    "gap_ratio": "1.0e0", "replacement_applied": apply,
                                    "cooldown_suppressed": skip,
                                    "last_progress_iteration": progress, "precond_calls": k}))
                            lines.append(marker(FINAL, {**fields,
                                "status": "Stagnated" if stalled else "Converged",
                                "iterations": steps[-1], "recursive_relative": f"{res:.15e}",
                                "true_relative": f"{res:.15e}",
                                "compensated_relative": f"{res:.15e}",
                                "precond_calls": steps[-1], "true_checks": len(steps),
                                "replacements": replaced, "false_events": 0,
                                "cooldown_suppressed": deferred,
                                "best_true_relative": f"{res:.15e}",
                                "last_progress_iteration": progress,
                                "relative_x_error": "n/a" if rhs == "PointLoad" else "1e-5"}))
                (self.output / f"{stem}.log").write_text("\n".join(lines) + "\n")
                (self.output / f"{stem}.time").write_text(
                    "elapsed_s=2.2 peak_rss_kib=12000 exit_status=0\n")

    def test_complete_72_certified_convergences(self) -> None:
        checks, finals = validate(self.a10c4, self.a10b, self.output)
        self.assertEqual(len(finals), 72)
        self.assertEqual(len(checks), 288)
        self.assertEqual(sum(int(row["replacements"]) for row in finals), 36)

    def test_complete_72_stagnation_heuristic_markers(self) -> None:
        self._logs(stalled=True)
        checks, finals = validate(self.a10c4, self.a10b, self.output)
        self.assertEqual(len(finals), 72)
        self.assertTrue(all(row["status"] == "Stagnated" for row in finals))
        self.assertTrue(all(int(row["replacements"]) > 0 for row in finals))

    def test_forged_convergence_and_cooldown_violation_rejected(self) -> None:
        p = self.output / "multi_patch_graph_piecewise_two_level_1.log"
        original = p.read_text()
        for changed in (
            original.replace("status=Converged", "status=MaxIterations", 1),
            original.replace("replacement_applied=0|cooldown_suppressed=1",
                             "replacement_applied=1|cooldown_suppressed=0", 1),
            original.replace("last_progress_iteration=25", "last_progress_iteration=1", 1),
        ):
            p.write_text(changed)
            with self.assertRaises(ValueError):
                validate(self.a10c4, self.a10b, self.output)

    def test_tampered_frozen_and_nonzero_gnu_time_rejected(self) -> None:
        p = self.a10c4 / "reliable_update_comparison.csv"
        before = p.read_text()
        p.write_text(before.replace("MaxIterations", "Converged", 1))
        with self.assertRaises(ValueError):
            validate(self.a10c4, self.a10b, self.output)
        p.write_text(before)
        t = self.output / "multi_patch_graph_piecewise_two_level_1.time"
        t.write_text("elapsed_s=2.2 peak_rss_kib=12000 exit_status=1\n")
        with self.assertRaises(ValueError):
            validate(self.a10c4, self.a10b, self.output)


if __name__ == "__main__":
    unittest.main()
