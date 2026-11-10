"""Reference/negative tests for A10C1 independent CSR diagnostics."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import CASES, NNZ, RHS
from benchmarks.scripts.C1H12_A10C1_precision import (
    FIELDS, diagnostics, manufactured_x, read_matrix_market, rhs_hash, summarize_frozen,
)
from benchmarks.scripts.C1H12_A10C1_precision_gate import validate


def small_matrix(root: Path, *, asymmetric: bool = False) -> Path:
    path = root / "small.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 7\n"
        "1 1 4\n1 2 -1\n2 1 -1\n2 2 4\n2 3 -1\n"
        + ("3 2 -0.5\n" if asymmetric else "3 2 -1\n")
        + "3 3 4\n",
        encoding="utf-8",
    )
    return path


def gate_fixture(root: Path) -> tuple[Path, Path]:
    frozen = root / "a10b"
    frozen.mkdir()
    hashes = {case: ("a" if case == CASES[0] else "b" if case == CASES[1] else "c") * 64 for case in CASES}
    (frozen / "matrix_sha256.txt").write_text(
        "".join(f"{digest}  /tmp/c1g_p1_{case}_n144.mtx\n" for case, digest in hashes.items()),
        encoding="utf-8",
    )
    rows = []
    for case in CASES:
        for rhs in RHS:
            measured = rhs != "point_load"
            rows.append({
                "case": case, "rhs_mode": rhs, "matrix_sha256": hashes[case],
                "rows": "20736", "nnz": str(NNZ[case]), "max_row_nnz": "5",
                "rhs_norm": "1", "abs_ax_norm": "2" if measured else "n/a",
                "cancellation_ratio": "2" if measured else "n/a",
                "roundoff_bound_relative": "1e-8" if measured else "n/a",
                "spmv_fsum_disagreement": "1e-9" if measured else "n/a",
                "x_rayleigh": "1" if measured else "n/a",
                "diag_scaled_rayleigh": "1" if measured else "n/a",
                "frozen_rhs_hash": "0" * 16,
                "fresh_converged": "30", "fresh_max_iterations": "0",
                "reuse_converged": "30", "reuse_max_iterations": "0",
                "frozen_min_true_relative": "1e-13", "frozen_max_true_relative": "9e-13",
            })
    output = root / "precision.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output, frozen


class A10C1Tests(unittest.TestCase):
    def test_matrix_spd_and_manufactured_rhs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix = read_matrix_market(small_matrix(Path(directory)))
            self.assertEqual(matrix.nnz, 7)
            self.assertEqual(matrix.diag, [4.0, 4.0, 4.0])
            ones = diagnostics(matrix, "ones")
            self.assertGreater(float(ones["roundoff_bound_relative"]), 0.0)
            self.assertLess(float(ones["spmv_fsum_disagreement"]), 1e-12)
            self.assertAlmostEqual(float(ones["x_rayleigh"]), 8.0 / 3.0)
            point = diagnostics(matrix, "point_load")
            self.assertEqual(point["rhs_norm"], "1.000000000000e+00")
            self.assertEqual(point["x_rayleigh"], "n/a")
            self.assertEqual(len(manufactured_x("seeded_random", 3)), 3)
            self.assertEqual(len(rhs_hash([1.0, 0.0])), 16)

    def test_reject_asymmetry_and_invalid_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "asymmetric"):
                read_matrix_market(small_matrix(root, asymmetric=True))
            path = small_matrix(root)
            path.write_text(path.read_text().replace("3 3 4", "3 3 nan"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                read_matrix_market(path)

    def test_frozen_lifecycle_count_and_rhs_consistency(self) -> None:
        rows = [
            {"case": "multi_patch", "rhs_mode": "Ones", "reuse_mode": mode,
             "rhs_hash": "abcdef0123456789", "status": "MaxIterations",
             "verified_true_relative": "1.2e-10"}
            for mode in ("fresh", "reuse") for _ in range(30)
        ]
        outcome = summarize_frozen(rows, "multi_patch", "ones")
        self.assertEqual(outcome["fresh_converged"], "0")
        self.assertEqual(outcome["reuse_max_iterations"], "30")
        rows[0]["rhs_hash"] = "a" * 16
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            summarize_frozen(rows, "multi_patch", "ones")

    def test_gate_accept_and_reject_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, frozen = gate_fixture(Path(directory))
            self.assertEqual(len(validate(output, frozen)), 12)
            original = output.read_text(encoding="utf-8")
            output.write_text(original.replace("1e-9", "1e-4", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "roundoff discrepancy"):
                validate(output, frozen)
            output.write_text(original.replace("20736", "20000", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSR shape"):
                validate(output, frozen)
            output.write_text(original.replace("30,0,30,0", "29,0,30,0", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(output, frozen)
            output.write_text(original, encoding="utf-8")
            (frozen / "matrix_sha256.txt").write_text("bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate(output, frozen)


if __name__ == "__main__":
    unittest.main()
