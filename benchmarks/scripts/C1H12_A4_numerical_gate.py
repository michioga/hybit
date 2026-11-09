#!/usr/bin/env python3
"""Audit native two-level HyBIT PCG pilot logs without discarding failures.

Convergence is accepted only against an independently recomputed true residual.
Neither MaxIterations nor a slower solve is silently reclassified as success.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

CASES = {
    "island_shell": 103104,
    "multi_patch": 103104,
    "rotated_anisotropy": 144002,
}
ARMS = {
    "jacobi": ("JacobiOnly", None, None),
    "contiguous_piecewise": ("DirectCoarse", "Contiguous", "PiecewiseConstant"),
    "contiguous_smoothed": ("DirectCoarse", "Contiguous", "JacobiSmoothed"),
    "graph_piecewise": ("DirectCoarse", "Graph", "PiecewiseConstant"),
    "graph_smoothed": ("DirectCoarse", "Graph", "JacobiSmoothed"),
    "strong_graph_piecewise": ("DirectCoarse", "StrongGraph", "PiecewiseConstant"),
    "strong_graph_smoothed": ("DirectCoarse", "StrongGraph", "JacobiSmoothed"),
}
FLOATS = (
    "tol", "reported_relative", "verified_true_relative", "relative_x_error",
    "load_ms", "setup_ms", "solve_ms", "total_wall_ms",
)
COUNTS = ("rows", "nnz", "coarse_dimension", "coarse_factor_bytes", "iterations")
FIELDS = (
    "case", "arm", "repeat", "mode", "aggregation", "basis", "transfer_values",
    *COUNTS, "status", *FLOATS, "elapsed_s", "peak_rss_kib", "exit_status",
)
TIME_RE = re.compile(
    r"^elapsed_s=(\S+) peak_rss_kib=(\S+) exit_status=(\S+)$", re.MULTILINE
)


def parse_record(log: str) -> dict[str, str]:
    rows = [line for line in log.splitlines() if line.startswith("C1H12A4|")]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one C1H12A4 record, got {len(rows)}")
    result: dict[str, str] = {}
    for token in rows[0].split("|")[1:]:
        if "=" not in token:
            raise ValueError(f"malformed diagnostic token: {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise ValueError(f"duplicate or empty diagnostic field: {key!r}")
        result[key] = value
    expected = set(FIELDS) - {"case", "arm", "repeat", "elapsed_s", "peak_rss_kib", "exit_status"}
    if result.keys() != expected:
        raise ValueError(f"diagnostic fields mismatch: missing={expected - result.keys()}, extra={result.keys() - expected}")
    return result


def parse_time(raw: str) -> dict[str, int | float]:
    matches = list(TIME_RE.finditer(raw))
    if len(matches) != 1:
        raise ValueError(f"expected one GNU time row, got {len(matches)}")
    elapsed = float(matches[0].group(1))
    rss = int(matches[0].group(2))
    exit_status = int(matches[0].group(3))
    if not math.isfinite(elapsed) or elapsed < 0 or rss < 0 or exit_status < 0:
        raise ValueError("invalid GNU time measurements")
    return {"elapsed_s": elapsed, "peak_rss_kib": rss, "exit_status": exit_status}


def validate_record(raw: dict[str, str], case: str, arm: str) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = dict(raw)
    for key in COUNTS:
        row[key] = int(raw[key])
        if row[key] < 0:
            raise ValueError(f"negative {key}")
    for key in FLOATS:
        row[key] = float(raw[key])
        if not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError(f"nonfinite/negative {key}")

    if row["rows"] != 20736 or row["nnz"] != CASES[case]:
        raise ValueError("unexpected matrix dimensions or sparsity")
    if row["tol"] != 1e-10 or row["iterations"] > 600:
        raise ValueError("tolerance or iteration budget mismatch")
    expected_mode, expected_aggregation, expected_basis = ARMS[arm]
    if row["mode"] != expected_mode or row["transfer_values"] != "F64":
        raise ValueError("unexpected mode or transfer precision")
    if arm == "jacobi":
        if row["coarse_dimension"] != 0 or row["coarse_factor_bytes"] != 0:
            raise ValueError("Jacobi control allocated a coarse factor")
    elif (
        row["aggregation"] != expected_aggregation
        or row["basis"] != expected_basis
        or row["coarse_dimension"] <= 0
        or row["coarse_factor_bytes"] <= 0
    ):
        raise ValueError("unexpected or empty coarse configuration")

    status = row["status"]
    true_rel = row["verified_true_relative"]
    if status == "Converged":
        if true_rel > row["tol"] * (1.0 + 1e-3):
            raise ValueError(f"false convergence: verified={true_rel:.12e}")
        if row["reported_relative"] > row["tol"] * (1.0 + 1e-3):
            raise ValueError("reported residual contradicts Converged status")
    elif status == "MaxIterations":
        if row["iterations"] != 600:
            raise ValueError("MaxIterations without exhausting budget")
        if true_rel <= row["tol"] * (1.0 + 1e-3):
            raise ValueError("solver reported MaxIterations but true residual passes")
    elif status == "Breakdown":
        if true_rel <= row["tol"] * (1.0 + 1e-3):
            raise ValueError("solver reported Breakdown but true residual passes")
    else:
        raise ValueError(f"unexpected solver status: {status}")
    if row["total_wall_ms"] + 1e-6 < row["load_ms"] + row["setup_ms"] + row["solve_ms"]:
        raise ValueError("total wall understates measured phases")
    return row


def audit(run_dir: Path, repeats: int) -> int:
    errors: list[str] = []
    rows: list[dict[str, str | int | float]] = []
    expected_stems = {
        f"{case}_{arm}_{repeat}"
        for case in CASES for arm in ARMS for repeat in range(1, repeats + 1)
    }
    for extension in (".log", ".time"):
        actual = {path.name.removesuffix(extension) for path in run_dir.glob(f"*{extension}")}
        for missing in sorted(expected_stems - actual):
            errors.append(f"missing {missing}{extension}")
        for extra in sorted(actual - expected_stems):
            errors.append(f"unexpected {extra}{extension}")

    for case in CASES:
        for arm in ARMS:
            for repeat in range(1, repeats + 1):
                stem = f"{case}_{arm}_{repeat}"
                log_path = run_dir / f"{stem}.log"
                time_path = run_dir / f"{stem}.time"
                if not log_path.is_file() or not time_path.is_file():
                    continue
                try:
                    time = parse_time(time_path.read_text(encoding="utf-8"))
                    if time["exit_status"] != 0:
                        raise ValueError(f"process exit={time['exit_status']}")
                    data = validate_record(
                        parse_record(log_path.read_text(encoding="utf-8")), case, arm
                    )
                    rows.append({"case": case, "arm": arm, "repeat": repeat, **data, **time})
                except (ValueError, OSError, UnicodeDecodeError) as exc:
                    errors.append(f"{stem}: {exc}")

    csv_path = run_dir / "native_coarse_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    converged = sum(row["status"] == "Converged" for row in rows)
    max_iters = sum(row["status"] == "MaxIterations" for row in rows)
    breakdowns = sum(row["status"] == "Breakdown" for row in rows)
    print(f"C1-H12-A4 native PCG gate: {len(rows)}/{len(expected_stems)} complete; Converged={converged}, MaxIterations={max_iters}, Breakdown={breakdowns}")
    for case in CASES:
        case_rows = [row for row in rows if row["case"] == case]
        baseline = [row for row in case_rows if row["arm"] == "jacobi"]
        base_text = (
            f"{baseline[0]['status']} ({baseline[0]['iterations']} iter, true={baseline[0]['verified_true_relative']:.3e})"
            if baseline else "MISSING"
        )
        rescues = sorted({
            row["arm"] for row in case_rows
            if row["arm"] != "jacobi" and row["status"] == "Converged"
        })
        print(f"  {case}: Jacobi={base_text}; converged coarse arms={','.join(rescues) or 'none'}")
    print(f"CSV: {csv_path}")
    print("Scope: native two-level coarse+PCG only, no multilevel or PETSc equivalence; no single-process speedup claim.")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"C1-H12-A4 evidence gate FAIL ({len(errors)} errors); raw logs retained", file=sys.stderr)
        return 1
    print(f"C1-H12-A4 {len(expected_stems)}/{len(expected_stems)} log/numerical evidence PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not args.run_dir.is_dir() or not 1 <= args.repeats <= 10:
        parser.error("run_dir must exist and repeats must be 1..10")
    return audit(args.run_dir, args.repeats)


if __name__ == "__main__":
    sys.exit(main())
