"""Negative/positive gates for the C1-H12-A2–A5 read-only evidence intake."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.scripts.C1H12_A2_A5_evidence_accept import (
    CASES, FALSE_A4, FULL_HEAD, check_source_integrity, evidence_rows,
    parse_unique_line, prepare_outputs,
    symbolic_record, validate_provenance,
)
from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, CASES as NNZ
from benchmarks.scripts.test_C1H12_A4_numerical_gate import record
from benchmarks.scripts.test_C1H12_A5_certification_gate import a5


def a2_line(case: str) -> str:
    return (
        f"C1H12A2|rows=20736|nnz={NNZ[case]}|threshold=0.25|strong_edges=81088"
        "|symbolic_nnz=179220|csr_graph_bytes=829440|abtm_metadata_bytes=825220"
        "|abtm_nonempty_words=61856|strength_ms=1.75|graph_csr_prepare_ms=0.44"
        "|abtm_conversion_ms=1.05|csr_symbolic_ms=2.06|abtm_symbolic_ms=2.23"
        "|repeats=5|symbolic_hash=9d436140fe40de0d|exact_equal=true"
    )


def a3_line(case: str, aggregate: int) -> str:
    return (
        f"C1H12A3|rows=20736|nnz={NNZ[case]}|aggregate_nodes={aggregate}|coarse_n=500"
        "|fine_pattern_edges=72000|symbolic_nnz=25560|numeric_nnz=25560"
        "|exact_zero_candidates=0|csr_adjacency_bytes=910080"
        "|temporary_csr_logical_bytes=946948|abtm_metadata_bytes=608260"
        "|abtm_nonempty_words=43776|graph_build_ms=0.65|graph_csr_prepare_ms=0.47"
        "|abtm_conversion_ms=0.95|numeric_reference_ms=0.55|csr_sort_ms=0.40"
        "|csr_stamp_ms=0.26|abtm_symbolic_ms=0.41|repeats=5"
        "|symbolic_hash=a0e43d8d5b930bd5|exact_equal=true"
    )


class EvidenceIntakeTests(unittest.TestCase):
    def test_generated_manifest_and_hash_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "docs").mkdir()
            handoff = b"Prior handoff content.\n"
            (repo / "docs/SESSION_HANDOFF.md").write_bytes(handoff)
            (repo / "MANIFEST.txt").write_text(
                "MANIFEST.txt\nSOURCE_SHA256.txt\ndocs/SESSION_HANDOFF.md\n"
            )
            (repo / "SOURCE_SHA256.txt").write_text(
                hashlib.sha256(handoff).hexdigest() + "  docs/SESSION_HANDOFF.md\n"
            )
            dirs = {stage: repo / stage for stage in ("A2", "A3", "A4", "A5")}
            symbolic = [{} for _ in range(12)]
            pcg = [
                {"stage": stage, "case": case, "arm": arm,
                 "iterations": 100, "verified_true_relative": 4e-11, "restarts": 0}
                for stage in ("A4", "A5") for case in CASES for arm in ARMS
            ]
            digests = [{"stage": "A2", "file": f"log_{index}", "sha256": "0" * 64} for index in range(108)]
            with (patch("benchmarks.scripts.C1H12_A2_A5_evidence_accept.validate_provenance",
                        return_value=("a" * 64, ["island_shell: `hash`"])),
                  patch("benchmarks.scripts.C1H12_A2_A5_evidence_accept.evidence_rows",
                        return_value=(symbolic, pcg, digests, FALSE_A4)),
                  patch("benchmarks.scripts.C1H12_A2_A5_evidence_accept.run_git",
                        return_value=FULL_HEAD)):
                source = Path(__file__).with_name("C1H12_A2_A5_evidence_accept.py")
                outputs = prepare_outputs(repo, source, dirs)
            self.assertIn("benchmarks/evidence/C1H12_A2_A5_raw_sha256.csv", outputs)
            for name, contents in outputs.items():
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            self.assertEqual(len(check_source_integrity(repo)), 9)

    def test_symbolic_field_and_finite_negative_gates(self):
        from benchmarks.scripts.C1H12_A2_A5_evidence_accept import finite_number

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.log"
            path.write_text(a2_line("island_shell"), encoding="utf-8")
            row = parse_unique_line(path, "C1H12A2")
            self.assertTrue(symbolic_record("A2", "island_shell", None, row)["exact_equal"])
            path.write_text(a2_line("island_shell") + "|rows=20736", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                parse_unique_line(path, "C1H12A2")
            row["abtm_symbolic_ms"] = "nan"
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                symbolic_record("A2", "island_shell", None, row)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                finite_number({"ms": "inf"}, "ms")

    def test_a3_boolean_symbolic_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a3.log"
            path.write_text(a3_line("multi_patch", 16), encoding="utf-8")
            row = parse_unique_line(path, "C1H12A3")
            self.assertEqual(symbolic_record("A3", "multi_patch", 16, row)["numeric_nnz"], 25560)
            row["exact_equal"] = "false"
            with self.assertRaisesRegex(ValueError, "exact-equality"):
                symbolic_record("A3", "multi_patch", 16, row)

    def test_full_54_cases_and_missing_raw_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {stage: Path(tmp) / stage for stage in ("A2", "A3", "A4", "A5")}
            for directory in paths.values():
                directory.mkdir()
            matrix_dir = Path(tmp) / "matrices"
            matrix_dir.mkdir()
            hashes = []
            for case in CASES:
                file = matrix_dir / f"c1g_p1_{case}_n144.mtx"
                file.write_bytes(case.encode())
                hashes.append(hashlib.sha256(file.read_bytes()).hexdigest() + "  " + str(file))
            for directory in paths.values():
                (directory / "HEAD.txt").write_text(FULL_HEAD + "\n")
                (directory / "matrix_sha256.txt").write_text("\n".join(hashes) + "\n")
            manifest, parsed = validate_provenance(paths, FULL_HEAD)
            self.assertEqual(len(manifest), 64)
            self.assertEqual(len(parsed), 3)

            for case in CASES:
                (paths["A2"] / f"{case}.log").write_text(a2_line(case))
                (paths["A2"] / f"{case}.time").write_text("elapsed_s=0.05 peak_rss_kib=6000 exit_status=0\n")
                for aggregate in (4, 16, 40):
                    stem = f"{case}_a{aggregate}"
                    (paths["A3"] / f"{stem}.log").write_text(a3_line(case, aggregate))
                    (paths["A3"] / f"{stem}.time").write_text("elapsed_s=0.05 peak_rss_kib=6000 exit_status=0\n")
                for arm, (mode, agg, basis) in ARMS.items():
                    false_old = f"{case}_{arm}" in FALSE_A4
                    max_iter = arm == "jacobi" and case != "island_shell" or (
                        case == "rotated_anisotropy" and arm == "contiguous_piecewise"
                    )
                    params = dict(mode=mode, aggregation=agg or "Graph", basis=basis or "JacobiSmoothed",
                                  nnz=NNZ[case], coarse_dim=0 if arm == "jacobi" else 512,
                                  coarse_bytes=0 if arm == "jacobi" else 200000,
                                  status="MaxIterations" if max_iter else "Converged",
                                  iterations=600 if max_iter else 100,
                                  true_rel="1e-8" if max_iter else "4e-10" if false_old else "4e-11",
                                  reported="1e-8" if max_iter else "5e-11")
                    stem = f"{case}_{arm}_1"
                    (paths["A4"] / f"{stem}.log").write_text(record(**params))
                    (paths["A4"] / f"{stem}.time").write_text("elapsed_s=0.1 peak_rss_kib=9000 exit_status=0\n")
                    params["iterations"] = 110 if false_old else params["iterations"]
                    params["true_rel"] = "4e-11" if false_old else params["true_rel"]
                    checks, restarts = (2, 1) if false_old else (1, 0)
                    cert_line = a5(
                        true_rel=params["true_rel"], iterations=params["iterations"],
                        checks=checks, restarts=restarts, rejected=restarts
                    )
                    if max_iter:
                        cert_line = cert_line.replace("status=Converged", "status=MaxIterations")
                    (paths["A5"] / f"{stem}.log").write_text(record(**params) + "\n" + cert_line + "\n")
                    (paths["A5"] / f"{stem}.time").write_text("elapsed_s=0.1 peak_rss_kib=9000 exit_status=0\n")

            symbolic, pcg, checksums, observed = evidence_rows(paths)
            self.assertEqual((len(symbolic), len(pcg), len(checksums), len(observed)), (12, 42, 108, 6))
            (paths["A5"] / "multi_patch_graph_smoothed_1.log").unlink()
            with self.assertRaisesRegex(ValueError, "file mismatch"):
                evidence_rows(paths)
            (paths["A4"] / "HEAD.txt").write_text("4593c20\n")
            with self.assertRaisesRegex(ValueError, "HEAD mismatch"):
                validate_provenance(paths, FULL_HEAD)


if __name__ == "__main__":
    unittest.main()
