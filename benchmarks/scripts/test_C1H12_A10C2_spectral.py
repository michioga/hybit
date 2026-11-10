"""Independent A10C2 negative and numerical tests; no FEM corpus required."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.scripts.C1H12_A10C1_precision import Matrix
from benchmarks.scripts.C1H12_A10C2_spectral import (
    ARMS, PROJECTION_FIELDS, SEEDS, SPECTRAL_FIELDS, VECTORS, aggregate,
    d_projection, jacobi_symmetric, lanczos, make_abtm_adjacency,
    prolongation, scaled_operator,
)
from benchmarks.scripts.C1H12_A10C2_spectral_gate import validate

CASES = ("island_shell", "multi_patch", "rotated_anisotropy")


def tridiagonal(n: int) -> Matrix:
    rows = []
    for i in range(n):
        row = []
        if i > 0:
            row.append((i - 1, -1.0))
        row.append((i, 2.0))
        if i + 1 < n:
            row.append((i + 1, -1.0))
        rows.append(row)
    return Matrix(rows=rows, diag=[2.0] * n, nnz=sum(map(len, rows)))


def write_records(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class SpectralTests(unittest.TestCase):
    def test_symmetric_jacobi_eigensystem_and_spd_lanczos(self):
        values, vectors = jacobi_symmetric([[2.0, -1.0], [-1.0, 2.0]])
        self.assertAlmostEqual(values[0], 1.0, places=12)
        self.assertAlmostEqual(values[1], 3.0, places=12)
        self.assertAlmostEqual(sum(x * x for x in vectors[0]), 1.0, places=12)
        matrix = tridiagonal(10)
        result = lanczos(scaled_operator(matrix), [1.0] * 10, 10)
        self.assertTrue(0.0 < result.low <= result.high)
        self.assertLess(result.low_residual, 1e-9)
        self.assertLess(result.high_residual, 1e-9)
        self.assertLess(result.orthogonality, 1e-10)

    def test_abtm_word_boundary_and_scalar_graph_equivalence(self):
        matrix = tridiagonal(70)
        words, count, edges = make_abtm_adjacency(matrix, False)
        self.assertGreater(count, 0)
        self.assertEqual(edges, 69)
        self.assertTrue(words[63][1] & 1)  # 63 -> 64, over 64-bit boundary
        self.assertTrue(words[64][0] & (1 << 63))
        owner, aggregates, _, _ = aggregate(matrix, False)
        self.assertEqual(len(owner), 70)
        self.assertEqual(set(owner), set(range(aggregates)))
        self.assertLessEqual(max(owner.count(j) for j in set(owner)), 4)

    def test_piecewise_exact_constant_and_smoothed_normal_certificate(self):
        matrix = tridiagonal(16)
        owner, n, _, _ = aggregate(matrix, True)
        piecewise, _ = prolongation(matrix, owner, False)
        error, captured, normal, steps = d_projection(
            matrix, piecewise, n, [1.0] * matrix.n, owner)
        self.assertLess(error, 1e-12)
        self.assertAlmostEqual(captured, 1.0, places=12)
        self.assertLess(normal, 1e-12)
        self.assertEqual(steps, 0)
        smoothed, _ = prolongation(matrix, owner, True)
        error, captured, normal, steps = d_projection(
            matrix, smoothed, n, [1.0] * matrix.n)
        self.assertLessEqual(error, 1.0)
        self.assertLessEqual(captured, 1.0)
        self.assertLess(normal, 1e-9)
        self.assertGreater(steps, 0)

    def test_non_spd_ritz_does_not_silently_pass(self):
        matrix = Matrix(rows=[[(0, 1.0), (1, 2.0)], [(0, 2.0), (1, 1.0)]],
                        diag=[1.0, 1.0], nnz=4)
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            lanczos(scaled_operator(matrix), [1.0, 0.0], 2)

    def test_gate_accepts_full_grid_and_rejects_tampering(self):
        fake_hashes = {case: (f"{i + 1:064x}") for i, case in enumerate(CASES)}
        original = [{"case": case, "rhs_mode": "ones", "diag_scaled_rayleigh": "0.2"}
                    for case in CASES]
        with tempfile.TemporaryDirectory() as base:
            folder = Path(base)
            a10b, a10c1, target = (folder / item for item in ("a10b", "a10c1", "target"))
            for directory in (a10b, a10c1, target):
                directory.mkdir()
            older, current = "1" * 40, "2" * 40
            (a10b / "HEAD.txt").write_text(older + "\n")
            (a10c1 / "HEAD.txt").write_text(current + "\n")
            (a10c1 / "precision_floor.csv").write_text("unchanged\n")
            (target / "HEAD.txt").write_text(current + "\n")
            (target / "FROZEN_A10B_HEAD.txt").write_text(older + "\n")
            (target / "FROZEN_A10C1_HEAD.txt").write_text(current + "\n")
            (target / "FROZEN_A10C1_SHA256.txt").write_text(
                hashlib.sha256(b"unchanged\n").hexdigest() + "\n")
            spectra = []
            projection = []
            for case in CASES:
                for seed in SEEDS:
                    spectra.append(dict(case=case, matrix_sha256=fake_hashes[case], seed=seed,
                                        rows="20736", nnz={"island_shell": "103104", "multi_patch": "103104", "rotated_anisotropy": "144002"}[case],
                                        steps="32", ritz_low="0.1", ritz_high="2", ritz_low_residual="0.01",
                                        ritz_high_residual="0.01", low_interval_lower="0.09",
                                        low_interval_upper="0.11", max_orthogonality="1e-14",
                                        lanczos_recurrence_error="1e-14", scaled_start_rayleigh="0.2"))
                for arm in ARMS:
                    for vector in VECTORS:
                        projection.append(dict(
                            case=case, matrix_sha256=fake_hashes[case], arm=arm, vector=vector,
                            rows="20736", nnz={"island_shell": "103104", "multi_patch": "103104", "rotated_anisotropy": "144002"}[case],
                            aggregates="6000", abtm_words="40000", abtm_edges="50000",
                            prolongation_nnz="24000" if arm.endswith("smoothed") else "20736",
                            omega="0.4", projection_error_d="0" if vector == "manufactured_ones" else "0.6",
                            captured_energy_d="1" if vector == "manufactured_ones" else "0.64",
                            normal_relative_residual="1e-12",
                            projection_iterations="2" if arm.endswith("smoothed") else "0",
                            vector_ritz_residual="0" if vector == "manufactured_ones" else "0.01"))
            write_records(target / "spectral_ritz.csv", SPECTRAL_FIELDS, spectra)
            write_records(target / "coarse_projection.csv", PROJECTION_FIELDS, projection)
            with (patch("benchmarks.scripts.C1H12_A10C2_spectral_gate.validate_c1", return_value=original),
                  patch("benchmarks.scripts.C1H12_A10C2_spectral_gate.load_expected_matrix_hashes", return_value=fake_hashes)):
                self.assertEqual(tuple(map(len, validate(target, a10b, a10c1))), (6, 27))
                projection[0]["projection_error_d"] = "1.7"
                write_records(target / "coarse_projection.csv", PROJECTION_FIELDS, projection)
                with self.assertRaisesRegex(ValueError, "invalid D-projection"):
                    validate(target, a10b, a10c1)
                projection[0]["projection_error_d"] = "0"
                write_records(target / "coarse_projection.csv", PROJECTION_FIELDS, projection)
                (a10c1 / "precision_floor.csv").write_text("tampered\n")
                with self.assertRaisesRegex(ValueError, "frozen A10C1 evidence changed"):
                    validate(target, a10b, a10c1)


if __name__ == "__main__":
    unittest.main()
