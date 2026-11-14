#!/usr/bin/env python3
"""Schema and negative-path tests; synthetic logs do not prove Rust numerical performance."""
import json
import tempfile
import csv
import hashlib
import shutil
import re
import unittest
from pathlib import Path

from benchmarks.scripts import C1G5B_gate as g

SAMPLE = {
    "case": "island_shell", "arm": "jacobi_pcg", "rhs_mode": "ones",
    "rhs_hash": "7688df0d3e2fd1d3", "status": "Converged", "iterations": "100",
    "tol": "1.000000000000e-10", "max_iterations": "600",
    "reported_relative": "6.000000000000e-11",
    "verified_true_relative": "6.000000000000e-11",
    "compensated_true_relative": "6.000000000000e-11",
    "relative_x_error": "1.000000000000e-7", "escalations": "0",
    "local_regions": "0", "local_factor_bytes": "0",
    "effective_preconditioner": "Jacobi", "rows": "20736", "nnz": "103104",
    "load_ms": "8.1", "rhs_ms": "1.0", "setup_ms": "0.1", "solve_ms": "10.0",
    "arm_work_ms": "10.8", "inprocess_total_wall_ms": "20.1",
}


def log(row=None):
    row = row or SAMPLE
    return "C1G5B|" + "|".join(k + "=" + row[k] for k in g.LOG_FIELDS) + "\n"


class GateTests(unittest.TestCase):
    def read(self, row=None):
        r = row or SAMPLE
        return g.parse_log(log(r), r["case"], r["rhs_mode"], r["arm"], r["rhs_hash"])

    def test_valid_jacobi_record(self):
        self.assertEqual(self.read()["status"], "Converged")

    def test_schema_tracks_rust_output_contract(self):
        root = Path(__file__).parent
        source = ((Path(__file__).resolve().parents[2] / "crates/hybit/examples/c1g5b_jacobi_hybrid_probe.rs")).read_text(encoding="utf-8")
        self.assertIn("C1G5B|case={}|arm={}|rhs_mode={}|rhs_hash={}", source)
        self.assertIn("compensated_true_relative={:.12e}", source)
        fmt = re.search(r'"C1G5B\|([^"\n]*)"', source)
        self.assertIsNotNone(fmt)
        actual = tuple(item.split("=", 1)[0] for item in fmt.group(1).split("|"))
        self.assertEqual(actual, g.LOG_FIELDS)
        self.assertEqual(len(self.read()), len(g.LOG_FIELDS))

    def test_duplicate_key_refused(self):
        with self.assertRaises(ValueError):
            g.parse_log(log() + "C1G5B|" + "|".join(k + "=" + SAMPLE[k] for k in g.LOG_FIELDS) + "\n",
                        "island_shell", "ones", "jacobi_pcg", SAMPLE["rhs_hash"])

    def test_missing_field_refused(self):
        r = SAMPLE.copy()
        r.pop("setup_ms")
        bad = "C1G5B|" + "|".join(k + "=" + str(v) for k, v in r.items())
        with self.assertRaises(ValueError):
            g.parse_log(bad, "island_shell", "ones", "jacobi_pcg", SAMPLE["rhs_hash"])

    def test_false_convergence_rejected(self):
        r = SAMPLE | {"verified_true_relative": "1.01e-10"}
        with self.assertRaisesRegex(ValueError, "false convergence"):
            self.read(r)

    def test_compensated_false_convergence_rejected(self):
        r = SAMPLE | {"compensated_true_relative": "1.01e-10"}
        with self.assertRaisesRegex(ValueError, "false convergence"):
            self.read(r)

    def test_bad_rhs_fingerprint_rejected(self):
        with self.assertRaisesRegex(ValueError, "RHS-fingerprint mismatch"):
            g.parse_log(log(), "island_shell", "ones", "jacobi_pcg", "a" * 16)

    def test_no_invented_point_load_reference(self):
        r = SAMPLE | {"rhs_mode": "point_load", "relative_x_error": "n/a"}
        self.assertEqual(self.read(r)["relative_x_error"], "n/a")
        with self.assertRaisesRegex(ValueError, "point-load"):
            self.read(r | {"relative_x_error": "0"})

    def test_no_explicit_jacobi_factors(self):
        with self.assertRaisesRegex(ValueError, "Jacobi arm"):
            self.read(SAMPLE | {"local_regions": "1"})

    def test_valid_hybrid_promoted(self):
        r = SAMPLE | {"arm": "hybrid_local_cholesky", "escalations": "1",
                      "local_regions": "8", "local_factor_bytes": "602931",
                      "effective_preconditioner": "Hybrid", "status": "MaxIterations",
                      "iterations": "600"}
        self.assertEqual(self.read(r)["effective_preconditioner"], "Hybrid")

    def test_hybrid_without_factor_requires_jacobi_only(self):
        r = SAMPLE | {"arm": "hybrid_local_cholesky", "effective_preconditioner": "JacobiOnly"}
        self.assertEqual(self.read(r)["effective_preconditioner"], "JacobiOnly")
        with self.assertRaisesRegex(ValueError, "nonactivated"):
            self.read(r | {"effective_preconditioner": "Hybrid"})

    def test_memory_budget_enforced(self):
        r = SAMPLE | {"arm": "hybrid_local_cholesky", "local_regions": "9"}
        with self.assertRaisesRegex(ValueError, "out-of-budget"):
            self.read(r)

    def test_max_iterations_requires_full_budget(self):
        r = SAMPLE | {"status": "MaxIterations", "iterations": "599"}
        with self.assertRaisesRegex(ValueError, "MaxIterations must exhaust"):
            self.read(r)

    def test_nan_and_negative_time_rejected(self):
        for value in ("nan", "-0.5", "inf", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read(SAMPLE | {"arm_work_ms": value})

    def test_valid_gnu_time(self):
        self.assertEqual(g.parse_time("elapsed_s=0.14 peak_rss_kib=12000 exit_status=0"),
                         {"elapsed_s": "0.14", "peak_rss_kib": "12000", "exit_status": "0"})

    def test_nonzero_gnu_time_rejected(self):
        with self.assertRaises(ValueError):
            g.parse_time("elapsed_s=0.14 peak_rss_kib=12000 exit_status=124")

    def test_duplicate_gnu_time_rejected(self):
        with self.assertRaises(ValueError):
            g.parse_time("elapsed_s=0.14 peak_rss_kib=12000 exit_status=0 exit_status=0")

    def test_csv_column_set(self):
        row = {k: "" for k in g.CSV_FIELDS}
        self.assertEqual(g.csv_bytes([row], g.CSV_FIELDS).count(b"\n"), 2)
        with self.assertRaises(ValueError):
            g.csv_bytes([row | {"extra": "bad"}], g.CSV_FIELDS)

    def test_write_once_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "f.csv"
            g.write_once(p, b"original")
            with self.assertRaises(FileExistsError):
                g.write_once(p, b"different")
            self.assertEqual(p.read_bytes(), b"original")

    def make_fixture(self, root: Path):
        repo = root / "repo"
        evidence = root / "evidence"
        (repo / "benchmarks/evidence").mkdir(parents=True)
        (repo / "crates/hybit/examples").mkdir(parents=True)
        evidence.mkdir()
        source = (Path(__file__).resolve().parents[2] / "crates/hybit/examples/c1g5b_jacobi_hybrid_probe.rs")
        target = repo / "crates/hybit/examples/c1g5b_jacobi_hybrid_probe.rs"
        shutil.copyfile(source, target)
        (evidence / "HEAD.txt").write_text(g.HEAD + "\n", encoding="utf-8")
        (evidence / "SOURCE.sha256").write_text(g.sha256(target) + "\n", encoding="utf-8")
        (repo / g.SOURCE_SUMMARY).write_text(json.dumps({
            "source_head": "37bc52dfb265123267081afc58bd2e6e7b0b6c97",
            "numerical_gate": "PASS_ORIGINAL_CSR_AND_GRID",
            "runs": 72,
            "matrix_sha256": {c: hashlib.sha256(c.encode()).hexdigest() for c in g.CASES},
        }), encoding="utf-8")
        rows = []
        with (repo / g.SOURCE).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, ["case", "rhs_mode", "rhs_hash"])
            writer.writeheader()
            for case in g.CASES:
                for mode in g.RHS:
                    marker = hashlib.sha256((case + mode).encode()).hexdigest()[:16]
                    for _ in range(6):
                        writer.writerow({"case": case, "rhs_mode": mode, "rhs_hash": marker})
                    for arm in g.ARMS:
                        row = SAMPLE | {
                            "case": case, "rhs_mode": mode, "rhs_hash": marker, "arm": arm,
                            "nnz": "144002" if case == "rotated_anisotropy" else "103104",
                            "relative_x_error": "n/a" if mode == "point_load" else "1e-7",
                            "effective_preconditioner": "Jacobi" if arm == "jacobi_pcg" else "Hybrid",
                            "local_regions": "0" if arm == "jacobi_pcg" else "8",
                            "local_factor_bytes": "0" if arm == "jacobi_pcg" else "602931",
                            "escalations": "0" if arm == "jacobi_pcg" else "1",
                        }
                        stem = f"{case}_{mode}_{arm}"
                        (evidence / f"{stem}.log").write_text(log(row), encoding="utf-8")
                        (evidence / f"{stem}.time").write_text(
                            "elapsed_s=0.12 peak_rss_kib=12000 exit_status=0\n", encoding="utf-8")
        return repo, evidence

    def test_full_24_process_synthetic_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, evidence = self.make_fixture(Path(tmp))
            data, summ, raw = g.audit(repo, evidence)
            self.assertEqual(len(data), 24)
            self.assertEqual(summ["paired_problems"], 12)
            self.assertEqual(summ["outcomes"], {"BOTH_CONVERGED": 12})
            self.assertEqual(raw.count(b"\n"), 49)  # 48 raw files and header

    def test_full_fixture_missing_log_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, evidence = self.make_fixture(Path(tmp))
            (evidence / "island_shell_ones_jacobi_pcg.log").unlink()
            with self.assertRaises(FileNotFoundError):
                g.audit(repo, evidence)

    def test_full_fixture_mismatched_fingerprint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, evidence = self.make_fixture(Path(tmp))
            path = evidence / "island_shell_ones_jacobi_pcg.log"
            path.write_text(path.read_text().replace("rhs_hash=", "rhs_hash=changed"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "RHS-fingerprint mismatch"):
                g.audit(repo, evidence)

    def test_full_fixture_changed_source_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, evidence = self.make_fixture(Path(tmp))
            path = repo / "crates/hybit/examples/c1g5b_jacobi_hybrid_probe.rs"
            path.write_text(path.read_text() + "//changed\n")
            with self.assertRaisesRegex(ValueError, "source hash changed"):
                g.audit(repo, evidence)

    def test_json_no_nan(self):
        with self.assertRaises(ValueError):
            json.dumps({"bad": float("nan")}, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
