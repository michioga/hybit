"""Synthetic positive/negative controls for independent A10C8-A phase accounting."""

import copy
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10C8A_profile_gate import (
    PHASES, SAMPLE_FIELDS, SUMMARY_COLUMNS, compare_frozen_sample,
    parse_process, validate_grid, verify_frozen_sha,
)
from benchmarks.scripts.test_C1H12_A10C7_matched_gate import make_grid


def synthetic_profiles():
    _, _, frozen, setup = make_grid()
    profiles = []
    for row in frozen:
        new = {key: row[key] for key in (
            "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown",
            "solver", "rep", "order", "status", "iterations", "recursive_relative",
            "true_relative", "compensated_relative", "precond_calls", "true_checks",
            "replacements", "false_events", "best_true_relative",
            "first_stagnation_candidate",
        )}
        new.update(solve_ns="300000", csr_ns="20000", kahan_ns="30000",
                   precond_ns="60000", spmv_ns="60000", dot_ns="10000",
                   vector_ns="20000", replace_spmv_ns="10000",
                   replace_vector_ns="10000", direction_ns="10000",
                   other_ns="70000", timed_precond_calls=row["precond_calls"],
                   timed_spmv_calls=row["iterations"],
                   timed_csr_calls=str(int(row["true_checks"]) - 1),
                   timed_kahan_calls=str(int(row["true_checks"]) - 1),
                   timed_replacements=row["replacements"])
        profiles.append(new)
    return frozen, profiles, setup


class ProfileGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen, cls.profiles, cls.setup = synthetic_profiles()

    def test_864_profile_samples_144_summaries(self):
        summaries = validate_grid(self.profiles, self.frozen, self.setup)
        self.assertEqual(len(summaries), 144)
        self.assertEqual(tuple(summaries[0]), SUMMARY_COLUMNS)
        self.assertTrue(all(row["phase_sum_is_exclusive"] == "true" for row in summaries))
        self.assertTrue(all(sum(int(row[p]) for p in PHASES) == int(row["solve_ns"])
                            for row in self.profiles))

    def test_overlap_missing_work_and_counter_forgery_fail(self):
        for field, forged in (
            ("other_ns", "70001"), ("csr_ns", "0"),
            ("precond_ns", "0"), ("timed_spmv_calls", "999"),
            ("timed_csr_calls", "0"), ("timed_replacements", "0"),
            ("replace_vector_ns", "0"), ("solve_ns", "1"),
        ):
            changed = copy.deepcopy(self.profiles)
            changed[0][field] = forged
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_grid(changed, self.frozen, self.setup)

    def test_frozen_numerics_reordering_and_incomplete_grid_fail(self):
        for field, forged in (
            ("iterations", "1"), ("true_relative", "1e-4"),
            ("rhs_hash", "0000000000000000"), ("order", "C4C6"),
            ("status", "Breakdown"),
        ):
            changed = copy.deepcopy(self.profiles)
            changed[0][field] = forged
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_grid(changed, self.frozen, self.setup)
        with self.assertRaises(ValueError):
            validate_grid(self.profiles[:-1], self.frozen, self.setup)
        with self.assertRaises(ValueError):
            validate_grid([*self.profiles, self.profiles[0]], self.frozen, self.setup)
        with self.assertRaises(ValueError):
            compare_frozen_sample({**self.profiles[0], "iterations": "999"}, self.frozen[0])

    def test_malformed_profile_logs_and_duplicate_fields_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "run.log"
            log.write_text("C1H12A10C8A_SAMPLE|solver=reliable|solver=shadow\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_process(log)
            self.assertNotEqual(len(SAMPLE_FIELDS), 0)

    def test_frozen_fingerprint_rejects_tampering(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root / "frozen"
            evidence = root / "evidence"
            frozen.mkdir()
            evidence.mkdir()
            files = ("matched_timing_samples.csv", "matched_timing_summary.csv")
            for name in files:
                (frozen / name).write_text("frozen\n", encoding="utf-8")
            (evidence / "FROZEN_A10C7_SHA256.txt").write_text("".join(
                f"{hashlib.sha256((frozen / name).read_bytes()).hexdigest()}  {frozen / name}\n"
                for name in files
            ), encoding="utf-8")
            verify_frozen_sha(evidence, frozen)
            (frozen / files[0]).write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_frozen_sha(evidence, frozen)


if __name__ == "__main__":
    unittest.main()
