"""Synthetic, intentionally adversarial tests for the A10B evidence gate."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import (
    ARMS, CASES, NNZ, POLICIES, REPEATS, REUSE, RHS, RHS_NATIVE, read_time, validate,
)


def record(marker: str, **fields: object) -> str:
    return marker + "|" + "|".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def fixture(root: Path) -> tuple[Path, Path, Path]:
    a9, a10a, trial = (root / name for name in ("a9", "a10a", "a10b"))
    for directory in (a9, a10a, trial):
        directory.mkdir()
        (directory / "HEAD.txt").write_text("f" * 40 + "\n", encoding="utf-8")
        (directory / "matrix_sha256.txt").write_text("a" * 64 + "  corpus.mtx\n", encoding="utf-8")
    with (a10a / "accuracy_tolerance_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["case"])
        writer.writeheader()
        writer.writerows([{"case": "island_shell"} for _ in range(54)])
    for case in CASES:
        for rhs in RHS:
            fingerprint = hashlib.sha256(f"{case}:{rhs}".encode()).hexdigest()[:16]
            (a9 / f"{case}_graph_piecewise_{rhs}_two_level_1.log").write_text(
                record("C1H12A9", rhs_mode=RHS_NATIVE[rhs], rhs_hash=fingerprint),
                encoding="utf-8",
            )

    for case in CASES:
        for arm, (aggregation, basis) in ARMS.items():
            for policy, native_policy in POLICIES.items():
                for reuse, native_reuse in REUSE.items():
                    stem = f"{case}_{arm}_{policy}_{reuse}_1"
                    lines = [record(
                        "C1H12A10B_SESSION", case=case, policy=native_policy,
                        aggregation=aggregation, basis=basis, reuse_mode=native_reuse,
                        rows=20736, nnz=NNZ[case], repeats=REPEATS, rhs_count=4,
                        tol="1.000000000000e-12", max_iterations=1200, load_ms="12.0",
                    )]
                    setup_sum = solve_sum = 0.0
                    for repeat in range(REPEATS):
                        for rhs in RHS:
                            setup = 1.0 if reuse == "fresh" else (2.0 if (repeat, rhs) == (0, "ones") else 0.0)
                            solve = 2.0
                            setup_sum += setup
                            solve_sum += solve
                            fingerprint = hashlib.sha256(f"{case}:{rhs}".encode()).hexdigest()[:16]
                            multi = policy == "multilevel"
                            lines.append(record(
                                "C1H12A10B", case=case, policy=native_policy,
                                aggregation=aggregation, basis=basis, reuse_mode=native_reuse,
                                repeat=repeat, rhs_mode=RHS_NATIVE[rhs], rhs_hash=fingerprint,
                                rows=20736, nnz=NNZ[case], levels=4 if multi else 2,
                                level_dims="20736,5192,1308,337" if multi else "20736,508",
                                level_nnz=f"{NNZ[case]},34592,8822,2177" if multi else f"{NNZ[case]},0",
                                coarse_factor_bytes=900_000 if multi else 2_400_000,
                                symbolic_nnz=90_000 if multi else 0,
                                exact_zero_candidates=10 if multi else 0,
                                abtm_words=400 if multi else 0,
                                scratch_bytes=350_000 if multi else 0,
                                status="Converged", iterations=18,
                                tol="1.000000000000e-12", max_iterations=1200,
                                reported_relative="4.000000000000e-13",
                                verified_true_relative="5.000000000000e-13",
                                compensated_true_relative="5.000000000000e-13",
                                relative_x_error="n/a" if rhs == "point_load" else "1.000000000000e-07",
                                error_response_relative="n/a" if rhs == "point_load" else "5.000000000000e-13",
                                error_rayleigh="n/a" if rhs == "point_load" else "4.000000000000e-11",
                                residual_amplification="n/a" if rhs == "point_load" else "2.000000000000e+05",
                                setup_ms=f"{setup:.6f}", solve_ms=f"{solve:.6f}",
                                sample_total_ms=f"{setup + solve:.6f}",
                                true_checks=1, restarts=0, rejected_recursive_convergences=0,
                                precond_calls=18 if multi else 0,
                            ))
                    lines.append(record(
                        "C1H12A10B_SUMMARY", case=case, policy=native_policy,
                        aggregation=aggregation, basis=basis, reuse_mode=native_reuse,
                        samples=4 * REPEATS, setup_ms_total=f"{setup_sum:.6f}",
                        solve_ms_total=f"{solve_sum:.6f}",
                        measured_total_ms=f"{setup_sum + solve_sum:.6f}",
                    ))
                    (trial / f"{stem}.log").write_text("".join(lines), encoding="utf-8")
                    (trial / f"{stem}.time").write_text(
                        "elapsed_s=0.7 peak_rss_kib=12000 exit_status=0\n", encoding="utf-8",
                    )
    return a9, a10a, trial


class PreparedReuseGateTests(unittest.TestCase):
    def test_gnu_time_real_format_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timefile = Path(directory) / "sample.time"
            for text in (
                "elapsed_s=0.7 peak_rss_kib=12000 exit_status=0\n",
                "elapsed_s=0.7\npeak_rss_kib=12000\nexit_status=0\n",
            ):
                timefile.write_text(text, encoding="utf-8")
                self.assertEqual(read_time(timefile), (0.7, 12000))
            for text in (
                "elapsed_s=0.7 peak_rss_kib=12000\n",
                "elapsed_s=0.7 peak_rss_kib=12000 exit_status=0 elapsed_s=0.8\n",
                "elapsed_s=0.7 peak_rss_kib=12000 exit_status=1\n",
                "elapsed_s=0.7 peak_rss_kib=0 exit_status=0\n",
                "elapsed_s=0.7 extra peak_rss_kib=12000 exit_status=0\n",
            ):
                timefile.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_time(timefile)

    def test_complete_720_samples_with_fresh_reuse_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = fixture(Path(directory))
            rows = validate(*expected, write_csv=False)
            self.assertEqual(len(rows), 720)
            self.assertEqual(sum(x["reuse_mode"] == "reuse" for x in rows), 360)

    def test_reject_unverified_convergence_and_pair_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a9, a10a, trial = fixture(Path(directory))
            filename = trial / "island_shell_graph_piecewise_two_level_fresh_1.log"
            original = filename.read_text(encoding="utf-8")
            filename.write_text(
                original.replace("verified_true_relative=5.000000000000e-13",
                                 "verified_true_relative=5.000000000000e-09", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unverified convergence"):
                validate(a9, a10a, trial, write_csv=False)
            filename.write_text(
                original.replace("iterations=18", "iterations=19", 1), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "divergence"):
                validate(a9, a10a, trial, write_csv=False)

    def test_reject_incomplete_or_changed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a9, a10a, trial = fixture(Path(directory))
            (trial / "HEAD.txt").write_text("0" * 40 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate(a9, a10a, trial, write_csv=False)
            (trial / "HEAD.txt").write_bytes((a9 / "HEAD.txt").read_bytes())
            (trial / "multi_patch_graph_piecewise_multilevel_reuse_1.log").unlink()
            with self.assertRaisesRegex(ValueError, "missing A10B run"):
                validate(a9, a10a, trial, write_csv=False)

    def test_reject_timing_or_abtm_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a9, a10a, trial = fixture(Path(directory))
            filename = trial / "rotated_anisotropy_strong_graph_smoothed_multilevel_reuse_1.log"
            original = filename.read_text(encoding="utf-8")
            filename.write_text(original.replace("abtm_words=400", "abtm_words=0", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid multilevel ABTM"):
                validate(a9, a10a, trial, write_csv=False)
            filename.write_text(original.replace("measured_total_ms=42.000000", "measured_total_ms=1.000000"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timing summary"):
                validate(a9, a10a, trial, write_csv=False)


if __name__ == "__main__":
    unittest.main()
