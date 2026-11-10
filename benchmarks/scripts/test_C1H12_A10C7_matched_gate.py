"""Synthetic fail-closed tests for the 864-sample A10C7 matched timing gate."""

import copy
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10C7_matched_gate import (
    ARMS, COOLDOWNS, POLICIES, REPETITIONS, RHS, SOLVERS, TOLS,
    check_frozen_hashes, parse_pipe, read_time, reference_maps, validate_grid,
    validate_session,
)


def make_grid():
    c4, c6, samples = [], [], []
    setup = {(arm, policy): 123456 for arm in ARMS for policy in POLICIES.values()}
    for arm in ARMS:
        for policy in POLICIES.values():
            for rhs in RHS:
                hash_value = "0123456789abcdef" if rhs == "Ones" else "abcdef0123456789"
                for tol in TOLS:
                    for period in (0, *COOLDOWNS):
                        key = dict(case="multi_patch", arm=arm, policy=policy,
                                   rhs_mode=rhs, rhs_hash=hash_value,
                                   tol=f"{tol:.0e}")
                        status = "Converged" if tol == 1e-10 else "MaxIterations"
                        iterations = "150" if status == "Converged" else "1200"
                        residual = "1e-11" if status == "Converged" else "2e-9"
                        c4row = dict(**key, replacement_period=str(period), status=status,
                                     iterations=iterations, recursive_relative=residual,
                                     true_relative=residual, compensated_relative=residual,
                                     best_true_relative=residual,
                                     precond_calls=iterations, true_checks="8",
                                     replacements="4", false_events="0")
                        c4.append(c4row)
                        if period == 0:
                            continue
                        c6row = dict(**key, cooldown=str(period), status=status,
                                     iterations=iterations, recursive_relative=residual,
                                     true_relative=residual, compensated_relative=residual,
                                     best_true_relative=residual,
                                     precond_calls=iterations, true_checks="8",
                                     replacements="3", false_events="1",
                                     first_stagnation_candidate="0",
                                     a10c4_status=status, a10c4_iterations=iterations,
                                     a10c4_replacements="4",
                                     a10c4_best_true_relative=residual)
                        c6.append(c6row)
                        for repetition in range(1, REPETITIONS + 1):
                            for solver in SOLVERS:
                                old = c4row if solver == "reliable" else c6row
                                samples.append(dict(
                                    **key, cooldown=str(period), solver=solver,
                                    rep=str(repetition),
                                    order="C6C4" if repetition % 2 else "C4C6",
                                    status=old["status"], iterations=old["iterations"],
                                    recursive_relative=old["recursive_relative"],
                                    true_relative=old["true_relative"],
                                    compensated_relative=old["compensated_relative"],
                                    best_true_relative=old["best_true_relative"],
                                    precond_calls=old["precond_calls"],
                                    true_checks=old["true_checks"], replacements=old["replacements"],
                                    false_events=old["false_events"],
                                    first_stagnation_candidate="0", solve_ns="200000",
                                    certificate_ns="30000", replacement_ns="20000",
                                    process_elapsed_s="1.23", process_peak_rss_kib="12000",
                                ))
    return c4, c6, samples, setup


class MatchedGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c4rows, cls.c6rows, cls.samples, cls.setup = make_grid()
        cls.c4, cls.c6 = reference_maps(cls.c4rows, cls.c6rows)

    def test_all_864_matched_samples_and_72_medians(self):
        result = validate_grid(self.samples, self.c4, self.c6, self.setup)
        self.assertEqual(len(result), 72)
        self.assertEqual(len(self.samples), 864)
        self.assertTrue(all(row["performance_status"] == "OBSERVATIONAL_ONLY" for row in result))
        self.assertTrue(all(float(row["paired_median_speed_ratio_c4_over_c6"]) == 1 for row in result))

    def test_missing_duplicate_or_reordered_sample_fails(self):
        for samples in (self.samples[:-1], [*self.samples, self.samples[0]]):
            with self.assertRaises(ValueError):
                validate_grid(samples, self.c4, self.c6, self.setup)
        changed = copy.deepcopy(self.samples)
        changed[0]["order"] = "C4C6" if changed[0]["order"] == "C6C4" else "C6C4"
        with self.assertRaises(ValueError):
            validate_grid(changed, self.c4, self.c6, self.setup)

    def test_forged_convergence_and_numerical_drift_fail(self):
        for field, forged in (("status", "Breakdown"),
                              ("true_relative", "4e-10"),
                              ("precond_calls", "999"),
                              ("rhs_hash", "0000000000000000")):
            changed = copy.deepcopy(self.samples)
            changed[0][field] = forged
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_grid(changed, self.c4, self.c6, self.setup)

    def test_invalid_split_timing_and_replacement_work_fail(self):
        for field, forged in (("solve_ns", "1"),
                              ("certificate_ns", "nan"),
                              ("replacement_ns", "0")):
            changed = copy.deepcopy(self.samples)
            changed[0][field] = forged
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_grid(changed, self.c4, self.c6, self.setup)

    def test_reference_disagreement_and_duplicated_reference_fail(self):
        changed = copy.deepcopy(self.c6rows)
        changed[0]["a10c4_status"] = "Breakdown"
        with self.assertRaises(ValueError):
            reference_maps(self.c4rows, changed)
        with self.assertRaises(ValueError):
            reference_maps([*self.c4rows, self.c4rows[0]], self.c6rows)

    def test_frozen_sha_and_nonzero_gnu_time_fail(self):
        with tempfile.TemporaryDirectory() as root:
            evidence = Path(root) / "evidence"
            frozen = Path(root) / "frozen"
            evidence.mkdir()
            frozen.mkdir()
            (frozen / "data.csv").write_text("frozen\n", encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(b"frozen\n").hexdigest()
            (evidence / "FROZEN_TEST_SHA256.txt").write_text(
                f"{digest}  {frozen / 'data.csv'}\n", encoding="utf-8")
            check_frozen_hashes(evidence, frozen, "TEST", ("data.csv",))
            (frozen / "data.csv").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                check_frozen_hashes(evidence, frozen, "TEST", ("data.csv",))
            timer = evidence / "run.time"
            timer.write_text("elapsed_s=1.25 peak_rss_kib=2000 exit_status=1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_time(timer)

    def test_duplicate_pipe_field_and_session_change_fail(self):
        with self.assertRaises(ValueError):
            parse_pipe("TEST|a=1|a=2", "TEST")
        session = dict(case="multi_patch", arm=ARMS[0], policy="TwoLevel",
                       rows="20736", nnz="103104", rhs_count="2", tol_count="3",
                       cooldown_count="2", warmups="1", measured_repetitions="6",
                       max_iterations="1200", stagnation_window="200",
                       progress_factor="9e-1", threshold_spacing="5",
                       setup_ns="123456", level_dims="20736,1024,128",
                       level_nnz="103104,8000,512", abtm_words="32")
        self.assertEqual(validate_session(session, ARMS[0], "TwoLevel"), 123456)
        session["measured_repetitions"] = "5"
        with self.assertRaises(ValueError):
            validate_session(session, ARMS[0], "TwoLevel")


if __name__ == "__main__":
    unittest.main()
