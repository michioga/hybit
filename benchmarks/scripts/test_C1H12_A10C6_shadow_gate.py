"""Synthetic tests for 72 shadow-mode trajectories and frozen A10C5 prefix."""

from __future__ import annotations

import hashlib
from importlib import import_module
import shutil
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import parse_pipe
from benchmarks.scripts.C1H12_A10C5_adaptive_gate import (
    CHECK as OLD_CHECK, CHECK_COLUMNS as OLD_CHECK_COLUMNS,
    FINAL as OLD_FINAL, FINAL_COLUMNS as OLD_FINAL_COLUMNS,
    SESSION as OLD_SESSION, validate as validate_c5, write_csv,
)
from benchmarks.scripts.C1H12_A10C6_shadow_gate import (
    CHECK, FINAL, SESSION, validate,
)


def line(tag: str, fields: dict[str, object]) -> str:
    return tag + "|" + "|".join(f"{k}={v}" for k, v in fields.items())


class ShadowGateTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture_type = import_module(
            "benchmarks.scripts.test_C1H12_A10C5_adaptive_gate"
        ).AdaptiveGateTests
        self.frozen = fixture_type("test_complete_72_certified_convergences")
        self.frozen.setUp()
        self.addCleanup(self.frozen.doCleanups)
        self.a10c5, self.a10c4, self.a10b = (
            self.frozen.output, self.frozen.a10c4, self.frozen.a10b
        )
        self.shadow = self.a10c5.parent / "a10c6"
        self.shadow.mkdir()
        for name in ("HEAD.txt", "matrix_sha256.txt"):
            shutil.copy2(self.a10c5 / name, self.shadow / name)
        for target, origin in (
            ("FROZEN_A10C5_HEAD.txt", self.a10c5),
            ("FROZEN_A10C4_HEAD.txt", self.a10c4),
            ("FROZEN_A10B_HEAD.txt", self.a10b),
        ):
            shutil.copy2(origin / "HEAD.txt", self.shadow / target)
        self.make_shadow(stalled=False)

    def make_shadow(self, stalled: bool, recover: bool = False) -> None:
        self.frozen._logs(stalled=stalled)
        checks, finals = validate_c5(self.a10c4, self.a10b, self.a10c5)
        write_csv(self.a10c5 / "adaptive_residual_trace.csv", OLD_CHECK_COLUMNS, checks)
        write_csv(self.a10c5 / "adaptive_update_comparison.csv", OLD_FINAL_COLUMNS, finals)
        with (self.shadow / "FROZEN_A10C5_SHA256.txt").open("w") as stream:
            for name in ("adaptive_update_comparison.csv", "adaptive_residual_trace.csv"):
                path = self.a10c5 / name
                stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
        for original_log in self.a10c5.glob("*.log"):
            lines = []
            observed = []
            for old_line in original_log.read_text().splitlines():
                if old_line.startswith(OLD_SESSION + "|"):
                    lines.append(old_line.replace(OLD_SESSION, SESSION, 1)
                                 + "|stop_policy=observe_only")
                elif old_line.startswith(OLD_CHECK + "|"):
                    point = parse_pipe(old_line, OLD_CHECK)
                    observed.append(point)
                    candidate = int(stalled and int(point["iteration"]) == 225)
                    lines.append(line(CHECK, {**point, "stagnation_candidate": candidate}))
                elif old_line.startswith(OLD_FINAL + "|"):
                    old = parse_pipe(old_line, OLD_FINAL)
                    if stalled and recover:
                        recovered = f"{float(old['tol']) / 2:.15e}"
                        checkpoint = {**observed[-1], "iteration": 250,
                                      "precond_calls": 250,
                                      "recursive_relative": recovered,
                                      "true_relative": recovered,
                                      "compensated_relative": recovered,
                                      "last_progress_iteration": 250,
                                      "replacement_applied": 0,
                                      "cooldown_suppressed": 0,
                                      "stagnation_candidate": 0}
                        lines.append(line(CHECK, checkpoint))
                        old.update(status="Converged", iterations=250,
                                   recursive_relative=recovered,
                                   true_relative=recovered,
                                   compensated_relative=recovered,
                                   best_true_relative=recovered,
                                   precond_calls=250,
                                   true_checks=len(observed) + 1,
                                   last_progress_iteration=250)
                    elif stalled:
                        last_replace = max(int(p["iteration"]) for p in observed
                                           if p["replacement_applied"] == "1")
                        replace_count = int(old["replacements"])
                        skip_count = int(old["cooldown_suppressed"])
                        for iteration in range(250, 1201, 25):
                            eligible = iteration - last_replace >= int(old["cooldown"])
                            replaced = int(eligible and iteration < 1200)
                            skipped = int(not eligible and iteration < 1200)
                            replace_count += replaced
                            skip_count += skipped
                            if replaced:
                                last_replace = iteration
                            checkpoint = {**observed[-1], "iteration": iteration,
                                          "precond_calls": iteration,
                                          "replacement_applied": replaced,
                                          "cooldown_suppressed": skipped,
                                          "stagnation_candidate": 0}
                            lines.append(line(CHECK, checkpoint))
                        old.update(status="MaxIterations", iterations=1200,
                                   precond_calls=1200,
                                   true_checks=len(observed) + len(range(250, 1201, 25)),
                                   replacements=replace_count, cooldown_suppressed=skip_count)
                    old["first_stagnation_candidate"] = 225 if stalled else 0
                    old["converged_after_candidate"] = int(stalled and recover)
                    lines.append(line(FINAL, old))
                    observed = []
            (self.shadow / original_log.name).write_text("\n".join(lines) + "\n")
            shutil.copy2(original_log.with_suffix(".time"),
                         self.shadow / original_log.with_suffix(".time").name)

    def test_all_72_frozen_convergences_have_identical_prefix(self) -> None:
        checks, finals = validate(self.a10c5, self.a10c4, self.a10b, self.shadow)
        self.assertEqual(len(finals), 72)
        self.assertEqual(len(checks), 288)
        self.assertTrue(all(r["status"] == "Converged" for r in finals))
        self.assertTrue(all(r["first_stagnation_candidate"] == "0" for r in finals))

    def test_72_stagnation_candidates_are_observed_through_budget(self) -> None:
        self.make_shadow(stalled=True)
        checks, finals = validate(self.a10c5, self.a10c4, self.a10b, self.shadow)
        self.assertEqual(len(finals), 72)
        self.assertEqual(len(checks), 72 * (len([0, 1, *range(25, 1201, 25)])))
        self.assertTrue(all(r["status"] == "MaxIterations" for r in finals))
        self.assertTrue(all(r["first_stagnation_candidate"] == "225" for r in finals))

    def test_72_post_candidate_recoveries_are_accepted(self) -> None:
        self.make_shadow(stalled=True, recover=True)
        _, finals = validate(self.a10c5, self.a10c4, self.a10b, self.shadow)
        self.assertEqual(len(finals), 72)
        self.assertTrue(all(r["status"] == "Converged" for r in finals))
        self.assertTrue(all(r["converged_after_candidate"] == "1" for r in finals))

    def test_forged_candidate_or_frozen_prefix_is_rejected(self) -> None:
        self.make_shadow(stalled=True)
        log = self.shadow / "multi_patch_graph_piecewise_two_level_1.log"
        source = log.read_text()
        for altered in (
            source.replace("stagnation_candidate=1", "stagnation_candidate=0", 1),
            source.replace("true_relative=1.000000000000000e-03",
                           "true_relative=2.000000000000000e-03", 1),
            source.replace("status=MaxIterations", "status=Stagnated", 1),
        ):
            log.write_text(altered)
            with self.assertRaises(ValueError):
                validate(self.a10c5, self.a10c4, self.a10b, self.shadow)
        log.write_text(source)

    def test_frozen_csv_tamper_and_nonzero_exit_are_rejected(self) -> None:
        c5 = self.a10c5 / "adaptive_update_comparison.csv"
        source = c5.read_text()
        c5.write_text(source.replace("Converged", "MaxIterations", 1))
        with self.assertRaises(ValueError):
            validate(self.a10c5, self.a10c4, self.a10b, self.shadow)
        c5.write_text(source)
        stamp = self.shadow / "multi_patch_graph_piecewise_two_level_1.time"
        stamp.write_text("elapsed_s=2.2 peak_rss_kib=12000 exit_status=3\n")
        with self.assertRaises(ValueError):
            validate(self.a10c5, self.a10c4, self.a10b, self.shadow)


if __name__ == "__main__":
    unittest.main()
