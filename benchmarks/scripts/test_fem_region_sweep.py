"""Stdlib-only C1 sweep tests; run with python3 -m unittest discover."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("fem_region_sweep.py")
SPEC = importlib.util.spec_from_file_location("fem_region_sweep", SCRIPT)
assert SPEC and SPEC.loader
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


OUTPUT = """HyBIT 0.10.0 FEM matrix benchmark
matrix             : thermal1.mtx
dimensions         : 82654 x 82654
nnz                : 574458
load time          : 63.607 ms

[1/2] Plain Jacobi-PCG
status             : Converged
iterations         : 1286
verified residual  : 9.840383e-9
setup / solve      : 1.096 / 881.417 ms
relative x error   : 6.150438e-8

[2/2] HyBIT Auto
status             : Converged
backend            : CSR32
preconditioner     : Jacobi
iterations         : 1286
verified residual  : 9.840383e-9
escalations        : 0
monitor checks     : 3
monitor trigger    : none
  monitor iter    12 : true residual 1.000000e0, per-iter ratio baseline, candidate false
  monitor iter    76 : true residual 5.000000e-1, per-iter ratio 0.989228013, candidate false
  monitor iter   140 : true residual 3.000000e-1, per-iter ratio 0.992050000, candidate true
probe              : 12 iters, 9.065 ms
local regions      : 0
largest region     : 0
factor memory      : 0.000 MiB
budget skipped     : 0 regions
budget limited     : false
analysis           : 7.430 ms
prepare            : 0.965 ms
diagnostics        : 0.000 ms
local factor       : 0.000 ms
solver time        : 874.416 ms
total wall         : 894.518 ms
relative x error   : 6.150438e-8

Comparison
iteration ratio    : 1.000 (HyBIT / plain)
"""


class SweepParserTests(unittest.TestCase):
    def test_real_fem_style_log_parses(self):
        parsed = sweep.parse_fem_output(OUTPUT)
        self.assertEqual(parsed["hybit_status"], "Converged")
        self.assertEqual(parsed["hybit_iterations"], 1286)
        self.assertEqual(parsed["plain_iterations"], 1286)
        self.assertEqual(parsed["hybit_probe_iterations"], 12)
        self.assertEqual(parsed["hybit_escalations"], 0)
        self.assertEqual(parsed["hybit_monitor_checks"], 3)
        self.assertIsNone(parsed["hybit_monitor_trigger_iteration"])
        self.assertEqual(parsed["hybit_monitor_candidate_count"], 1)
        self.assertAlmostEqual(parsed["hybit_monitor_last_rate"], 0.99205)
        self.assertAlmostEqual(parsed["hybit_monitor_worst_rate"], 0.99205)
        self.assertEqual(parsed["nnz"], 574458)
        self.assertAlmostEqual(parsed["plain_solve_ms"], 881.417)
        self.assertAlmostEqual(parsed["hybit_wall_ms"], 894.518)
        monitored = sweep.parse_fem_output(OUTPUT.replace(
            "monitor checks     : 3\nmonitor trigger    : none",
            "monitor checks     : 5\nmonitor trigger    : 140"))
        self.assertEqual(monitored["hybit_monitor_checks"], 5)
        self.assertEqual(monitored["hybit_monitor_trigger_iteration"], 140)
        row = dict(parsed, exit_code=0, timed_out=False)
        sweep.numeric_gate(row, 1e-8, True, 64)
        self.assertTrue(row["paired_pass"])
        self.assertEqual(row["failure_reason"], "")

    def test_false_success_never_promoted(self):
        for bad in ["MaxIterations", "Breakdown"]:
            parsed = sweep.parse_fem_output(OUTPUT.replace(
                "[2/2] HyBIT Auto\nstatus             : Converged",
                f"[2/2] HyBIT Auto\nstatus             : {bad}"))
            row = dict(parsed, exit_code=0, timed_out=False)
            sweep.numeric_gate(row, 1e-8, True, 64)
            self.assertFalse(row["hybit_pass"])

    def test_nonfinite_and_tolerance_rejected(self):
        for value in ("inf", "nan", "1.1e-8"):
            parsed = sweep.parse_fem_output(OUTPUT.replace(
                "[2/2] HyBIT Auto\nstatus             : Converged\nbackend",
                "[2/2] HyBIT Auto\nstatus             : Converged\nbackend").replace(
                "verified residual  : 9.840383e-9\nescalations",
                f"verified residual  : {value}\nescalations"))
            row = dict(parsed, exit_code=0, timed_out=False)
            sweep.numeric_gate(row, 1e-8, True, 64)
            self.assertFalse(row["hybit_pass"])

    def test_missing_output_and_nonzero_exit_retained_as_failure(self):
        row = dict(sweep.parse_fem_output("failed to parse matrix"),
                   exit_code=2, timed_out=False)
        sweep.numeric_gate(row, 1e-8, True, 64)
        self.assertFalse(row["paired_pass"])
        self.assertIn("process_exit", row["failure_reason"])

    def test_fixed_factor_budget_check(self):
        parsed = sweep.parse_fem_output(OUTPUT.replace(
            "factor memory      : 0.000 MiB", "factor memory      : 64.200 MiB"))
        row = dict(parsed, exit_code=0, timed_out=False)
        sweep.numeric_gate(row, 1e-8, True, 64)
        self.assertFalse(row["hybit_pass"])

    def test_hybit_only_output(self):
        text = OUTPUT.replace("[1/2] Plain Jacobi-PCG", "discarded").split("discarded")[0]
        text += "Plain Jacobi-PCG   : skipped (--skip-plain)\n\n"
        text += "[1/1] HyBIT Auto" + OUTPUT.split("[2/2] HyBIT Auto", 1)[1]
        parsed = sweep.parse_fem_output(text)
        row = dict(parsed, exit_code=0, timed_out=False)
        sweep.numeric_gate(row, 1e-8, False, 64)
        self.assertTrue(row["hybit_pass"])
        self.assertIsNone(row["plain_pass"])

    def test_non_escalating_result_not_region_evidence(self):
        row = dict(sweep.parse_fem_output(OUTPUT), phase="measure", max_region=128,
                   process_wall_ms=1000., peak_rss_kib=16384,
                   exit_code=0, timed_out=False)
        sweep.numeric_gate(row, 1e-8, True, 64)
        summary = sweep.summarize([row], [128])
        self.assertEqual(summary["128"]["escalated_trials"], 0)
        self.assertFalse(summary["128"]["region_size_informative"])

    def test_nonfinite_value_serializes_to_strict_json_null(self):
        parsed = sweep.parse_fem_output(OUTPUT.replace(
            "verified residual  : 9.840383e-9\nescalations",
            "verified residual  : inf\nescalations"))
        row = dict(parsed, exit_code=0, timed_out=False)
        sweep.numeric_gate(row, 1e-8, True, 64)
        self.assertFalse(row["hybit_pass"])
        safe = sweep.strict_json_row(row)
        self.assertIsNone(safe["hybit_verified_residual"])
        json.dumps(safe, allow_nan=False)

    def test_cpu_ranges_and_process_launch_failures(self):
        self.assertEqual(sweep.parse_cpus("0-2,5,7-8"), [0, 1, 2, 5, 7, 8])
        with self.assertRaises(Exception):
            sweep.parse_cpus("2-1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            code, timed_out, _, peak = sweep.run_process(
                [str(path / "no-such-binary")], path,
                path / "stdout.log", path / "stderr.log", 5., {},
            )
            self.assertEqual(code, 127)
            self.assertFalse(timed_out)
            self.assertEqual(peak, 0)
            self.assertIn("Failed to launch", (path / "stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
