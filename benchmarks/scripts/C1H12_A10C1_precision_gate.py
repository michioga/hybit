#!/usr/bin/env python3
"""Fail-closed schema/provenance gate for A10C1 independent CSR diagnostics."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import CASES, NNZ, RHS
from benchmarks.scripts.C1H12_A10C1_precision import FIELDS, load_expected_matrix_hashes


def validate(csv_file: Path, frozen_a10b: Path) -> list[dict[str, str]]:
    hashes = load_expected_matrix_hashes(frozen_a10b / "matrix_sha256.txt")
    with csv_file.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("A10C1 CSV schema mismatch")
        rows = list(reader)
    if len(rows) != 12:
        raise ValueError("A10C1 incomplete case/RHS grid")
    seen = set()
    for row in rows:
        if set(row) != set(FIELDS) or any(value is None or not value for value in row.values()):
            raise ValueError("missing or empty A10C1 field")
        case, mode = row["case"], row["rhs_mode"]
        if case not in CASES or mode not in RHS or (case, mode) in seen:
            raise ValueError("duplicate or unknown case/RHS")
        seen.add((case, mode))
        if row["matrix_sha256"] != hashes[case] or not re.fullmatch("[0-9a-f]{16}", row["frozen_rhs_hash"]):
            raise ValueError("bad matrix or RHS provenance")
        if int(row["rows"]) != 20736 or int(row["nnz"]) != NNZ[case]:
            raise ValueError("unexpected CSR shape")
        if not (0 < int(row["max_row_nnz"]) <= int(row["rows"])):
            raise ValueError("bad matrix row work")
        for lifecycle in ("fresh", "reuse"):
            conv = int(row[f"{lifecycle}_converged"])
            max_iter = int(row[f"{lifecycle}_max_iterations"])
            if conv < 0 or max_iter < 0 or conv + max_iter != 30:
                raise ValueError("bad frozen convergence count")
        if any(row[f"fresh_{k}"] != row[f"reuse_{k}"] for k in ("converged", "max_iterations")):
            raise ValueError("fresh/reuse status mismatch")
        min_true, max_true = (float(row[k]) for k in ("frozen_min_true_relative", "frozen_max_true_relative"))
        if not (math.isfinite(min_true) and math.isfinite(max_true) and 0 <= min_true <= max_true):
            raise ValueError("bad frozen original-CSR residual")
        rhs_norm = float(row["rhs_norm"])
        if not (math.isfinite(rhs_norm) and rhs_norm > 0.0):
            raise ValueError("bad RHS norm")
        fields = (
            "abs_ax_norm", "cancellation_ratio", "roundoff_bound_relative",
            "spmv_fsum_disagreement", "x_rayleigh", "diag_scaled_rayleigh",
        )
        if mode == "point_load":
            if rhs_norm != 1.0 or any(row[field] != "n/a" for field in fields):
                raise ValueError("point-load manufactured diagnostics must be n/a")
        else:
            vals = {field: float(row[field]) for field in fields}
            if any(not math.isfinite(x) for x in vals.values()):
                raise ValueError("nonfinite independent precision diagnostic")
            if any(vals[k] <= 0 for k in ("abs_ax_norm", "cancellation_ratio", "roundoff_bound_relative", "x_rayleigh", "diag_scaled_rayleigh")):
                raise ValueError("nonpositive precision or Rayleigh result")
            if vals["spmv_fsum_disagreement"] < 0 or vals["spmv_fsum_disagreement"] > 4 * vals["roundoff_bound_relative"]:
                raise ValueError("roundoff discrepancy violates stated bound")
            if abs(vals["abs_ax_norm"] / rhs_norm - vals["cancellation_ratio"]) > 2e-10 * vals["cancellation_ratio"]:
                raise ValueError("inconsistent cancellation ratio")
    if seen != {(case, mode) for case in CASES for mode in RHS}:
        raise ValueError("missing A10C1 case/RHS record")
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: C1H12_A10C1_precision_gate.py OUTPUT_CSV FROZEN_A10B")
    try:
        records = validate(Path(sys.argv[1]), Path(sys.argv[2]))
        print(f"C1-H12-A10C1 {len(records)}/12 CSR provenance, roundoff, and frozen-status gate PASS")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"C1-H12-A10C1 gate FAIL: {exc}") from exc
