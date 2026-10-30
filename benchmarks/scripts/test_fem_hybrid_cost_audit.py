"""Fail-closed C1-E independent crossover auditor tests (stdlib only)."""
import importlib.util
import json
import random
import statistics
from pathlib import Path
import tempfile
import unittest


SRC = Path(__file__).with_name("fem_hybrid_cost_audit.py")
SPEC = importlib.util.spec_from_file_location("audit", SRC)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixtures(n=9, ratio=1.2):
    metadata = {
        "schema": audit.SCHEMA, "matrix_sha256": "a" * 64,
        "binary_sha256": "b" * 64, "git_head": "8" * 40,
        "git_dirty": True, "command": ["fem_bench", "--tol", "1e-8",
                                              "--factor-budget-mib", "64"],
        "pairs": n, "seed": 20261030, "threads": 1,
        "arm_difference": "observe adds --monitor-only", "rhs_kind": "manufactured_A_times_ones",
    }
    rows = []
    for i in range(1, n + 1):
        for arm, position in (("observe", 1 if i % 2 else 2),
                              ("hybrid", 2 if i % 2 else 1)):
            hybrid = arm == "hybrid"
            rows.append(dict(phase="measure", pair=str(i), position=str(position),
                             arm=arm, **{"pass": "True"}, exit_code="0", timed_out="False",
                             hybit_status="Converged", hybit_escalations=str(int(hybrid)),
                             hybit_verified_residual="9e-9",
                             hybit_iterations=str(1305 if hybrid else 1286),
                             hybit_wall_ms=str(900 * ratio if hybrid else 900),
                             hybit_solve_ms=str(880 * ratio if hybrid else 880),
                             hybit_diagnostics_ms=str(32 if hybrid else 12),
                             hybit_local_factor_ms=str(2.6 if hybrid else 0),
                             hybit_factor_mib=str(0.693 if hybrid else 0),
                             peak_rss_kib=str(37200 if hybrid else 26300)))
    rr = [ratio - 1] * n
    rand = random.Random(20261030)
    summary_ci = ([audit.quantile([statistics.median(rand.choices(rr, k=n))
                                  for _ in range(4000)], p) for p in (0.025, 0.975)]
                  if n >= 5 else None)
    summary = dict(measured_pairs=n, valid_pairs=n, escalated_pairs=n,
                   margin_fraction=.05, median_relative_cost=ratio - 1,
                   bootstrap_95pct=summary_ci,
                   classification=("measured_regression_in_this_matrix" if ratio > 1.05
                                   else "potential_benefit_in_this_matrix" if ratio < .95
                                   else "inconclusive"))
    return metadata, rows, summary


class AuditTests(unittest.TestCase):
    def test_detects_strong_regression_and_validates_raw_summary(self):
        m, rows, summary = fixtures()
        out = audit.audit(m, rows, summary)
        self.assertFalse(out["fail_closed"])
        self.assertTrue(out["c1d_summary_consistent"])
        self.assertEqual(out["classification"], "measured_regression_in_this_matrix")
        self.assertAlmostEqual(out["median_relative_cost"], .2)
        self.assertAlmostEqual(out["median_deltas"]["hybit_wall_ms"], 180)
        self.assertAlmostEqual(out["two_sided_sign_p_exploratory"], 2/512)
        self.assertEqual(out["hybrid_first_pairs"], 4)

    def test_detects_strong_benefit_without_production_claim(self):
        m, rows, summary = fixtures(ratio=.8)
        out = audit.audit(m, rows, summary)
        self.assertFalse(out["fail_closed"])
        self.assertEqual(out["classification"], "potential_benefit_in_this_matrix")
        self.assertIn("not production policy", out["limitations"][0])

    def test_detects_incomplete_pair(self):
        m, rows, summary = fixtures()
        rows.pop()
        out = audit.audit(m, rows, summary)
        self.assertTrue(out["fail_closed"])
        self.assertIsNone(out["median_relative_cost"])

    def test_rejects_no_activation(self):
        m, rows, summary = fixtures()
        rows[1]["hybit_escalations"] = "0"
        self.assertTrue(audit.audit(m, rows, summary)["fail_closed"])

    def test_rejects_failed_numerical_status(self):
        m, rows, summary = fixtures()
        rows[2]["hybit_verified_residual"] = "2e-8"
        self.assertTrue(audit.audit(m, rows, summary)["fail_closed"])

    def test_rejects_nonfinite_or_missing_timings(self):
        for value in ("", "nan", "inf", "-1"):
            m, rows, summary = fixtures()
            rows[0]["hybit_wall_ms"] = value
            self.assertTrue(audit.audit(m, rows, summary)["fail_closed"])

    def test_rejects_duplicate_arms_and_invalid_positions(self):
        m, rows, summary = fixtures()
        rows.append(dict(rows[0]))
        self.assertTrue(audit.audit(m, rows, summary)["fail_closed"])
        m, rows, summary = fixtures()
        rows[1]["position"] = rows[0]["position"]
        self.assertTrue(audit.audit(m, rows, summary)["fail_closed"])

    def test_rejects_unrelated_sweep_schema(self):
        m, rows, summary = fixtures()
        m["schema"] = "hybit-c1-fem-region-sweep-v2"
        with self.assertRaisesRegex(ValueError, "do not mix"):
            audit.audit(m, rows, summary)

    def test_rejects_tampered_summary(self):
        m, rows, summary = fixtures()
        summary["median_relative_cost"] = -0.2
        out = audit.audit(m, rows, summary)
        self.assertTrue(out["fail_closed"])
        self.assertEqual(out["classification"], "inconsistent_evidence")

    def test_rejects_tampered_bootstrap_interval(self):
        m, rows, summary = fixtures()
        summary["bootstrap_95pct"][0] = -0.1
        self.assertTrue(audit.audit(m, rows, summary)["fail_closed"])

    def test_small_n_never_claims_superiority(self):
        m, rows, summary = fixtures(n=2)
        summary["classification"] = "insufficient_pairs_for_classification"
        out = audit.audit(m, rows, summary)
        self.assertFalse(out["fail_closed"])
        self.assertEqual(out["classification"], "insufficient_pairs_for_classification")
        self.assertIsNone(out["bootstrap_95pct"])

    def test_cli_rejects_legacy_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            m, rows, s = fixtures()
            m["schema"] = "hybit-c1-fem-region-sweep-v2"
            (path/"metadata.json").write_text(json.dumps(m))
            (path/"summary.json").write_text(json.dumps(s))
            import csv
            with (path/"results.csv").open("w", newline="") as f:
                w=csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            import subprocess, sys
            p = subprocess.run([sys.executable, str(SRC), "--evidence-dir", str(path)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn("do not mix", p.stderr)


if __name__ == "__main__":
    unittest.main()
