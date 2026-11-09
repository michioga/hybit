#!/usr/bin/env python3
"""Crosscheck A5 certified PCG against all original A4 raw diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, CASES, parse_record
except ModuleNotFoundError:  # Direct execution from benchmarks/scripts/.
    from C1H12_A4_numerical_gate import ARMS, CASES, parse_record

A5_FIELDS = {
    "policy", "true_checks", "restarts", "rejected_recursive_convergences",
    "final_true_relative", "global_iterations", "status",
}


def parse_a5_record(log: str) -> dict[str, int | float | str]:
    lines = [line for line in log.splitlines() if line.startswith("C1H12A5|")]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one C1H12A5 row, got {len(lines)}")
    raw: dict[str, str] = {}
    for token in lines[0].split("|")[1:]:
        if "=" not in token:
            raise ValueError(f"malformed A5 token: {token!r}")
        name, value = token.split("=", 1)
        if not name or not value or name in raw:
            raise ValueError(f"duplicate/empty A5 field: {name!r}")
        raw[name] = value
    if raw.keys() != A5_FIELDS:
        raise ValueError(f"A5 field mismatch: {raw.keys() ^ A5_FIELDS}")
    row: dict[str, int | float | str] = dict(raw)
    for key in ("true_checks", "restarts", "rejected_recursive_convergences", "global_iterations"):
        row[key] = int(raw[key])
        if row[key] < 0:
            raise ValueError(f"negative {key}")
    row["final_true_relative"] = float(raw["final_true_relative"])
    if not math.isfinite(row["final_true_relative"]) or row["final_true_relative"] < 0:
        raise ValueError("invalid final true residual")
    if row["policy"] != "CertifiedRestart":
        raise ValueError("unknown A5 policy")
    if row["true_checks"] == 0 or row["global_iterations"] > 600:
        raise ValueError("bad A5 count or iteration budget")
    if not row["restarts"] <= row["rejected_recursive_convergences"] <= row["true_checks"]:
        raise ValueError("inconsistent restart/verification counts")
    if row["status"] not in ("Converged", "MaxIterations", "Breakdown"):
        raise ValueError("unknown A5 solver status")
    return row


def audit(base_dir: Path, certified_dir: Path) -> int:
    errors: list[str] = []
    rows: list[dict[str, str | float | int]] = []
    for stamp in ("HEAD.txt", "matrix_sha256.txt"):
        if not (base_dir / stamp).is_file() or not (certified_dir / stamp).is_file():
            errors.append(f"missing provenance: {stamp}")
        elif (base_dir / stamp).read_text() != (certified_dir / stamp).read_text():
            errors.append(f"baseline and A5 provenance mismatch: {stamp}")

    for case in CASES:
        for arm in ARMS:
            stem = f"{case}_{arm}_1"
            base = base_dir / f"{stem}.log"
            changed = certified_dir / f"{stem}.log"
            if not base.is_file() or not changed.is_file():
                errors.append(f"missing A4/A5 log: {stem}")
                continue
            try:
                old = parse_record(base.read_text(encoding="utf-8"))
                new_text = changed.read_text(encoding="utf-8")
                current = parse_record(new_text)
                verified = parse_a5_record(new_text)
                if int(current["iterations"]) != verified["global_iterations"]:
                    raise ValueError("A4/A5 iteration counts disagree")
                if current["status"] != verified["status"]:
                    raise ValueError("A4/A5 statuses disagree")
                if not math.isclose(
                    float(current["verified_true_relative"]),
                    float(verified["final_true_relative"]),
                    rel_tol=1e-9, abs_tol=1e-14,
                ):
                    raise ValueError("A4/A5 independently computed true residuals disagree")
                if current["status"] == "Converged" and float(current["verified_true_relative"]) > 1.001e-10:
                    raise ValueError("A5 falsely declares convergence")
                rows.append({
                    "case": case,
                    "arm": arm,
                    "before_status": old["status"],
                    "before_iterations": int(old["iterations"]),
                    "before_true_relative": float(old["verified_true_relative"]),
                    "after_status": current["status"],
                    "after_iterations": int(current["iterations"]),
                    "after_true_relative": float(current["verified_true_relative"]),
                    "true_checks": int(verified["true_checks"]),
                    "certified_restarts": int(verified["restarts"]),
                    "rejected_recursive_convergences": int(verified["rejected_recursive_convergences"]),
                    "after_setup_ms": float(current["setup_ms"]),
                    "after_solve_ms": float(current["solve_ms"]),
                    "after_coarse_factor_bytes": int(current["coarse_factor_bytes"]),
                })
            except (ValueError, OSError, UnicodeDecodeError) as exc:
                errors.append(f"{stem}: {exc}")

    csv_path = certified_dir / "certified_restart_comparison.csv"
    fieldnames = [
        "case", "arm", "before_status", "before_iterations", "before_true_relative",
        "after_status", "after_iterations", "after_true_relative", "true_checks",
        "certified_restarts", "rejected_recursive_convergences", "after_setup_ms",
        "after_solve_ms", "after_coarse_factor_bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"C1-H12-A5 A4 baseline vs certified comparison: {len(rows)}/21 complete")
    for case in CASES:
        for row in rows:
            if row["case"] != case or (case != "multi_patch" and row["arm"] != "strong_graph_smoothed"):
                continue
            print(
                f"  {case}/{row['arm']}: before={row['before_status']} "
                f"{row['before_iterations']} it true={row['before_true_relative']:.3e}; "
                f"after={row['after_status']} {row['after_iterations']} it "
                f"true={row['after_true_relative']:.3e}; "
                f"cert_restart={row['certified_restarts']}"
            )
    print(f"CSV: {csv_path}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    if len(rows) != 21:
        print("FAIL: expected 21 original/comparison pairs", file=sys.stderr)
        return 1
    print("C1-H12-A5 21/21 certificate/log/provenance comparisons PASS")
    print("Scope: same matrix RHS and PCG policy; no native multilevel AMG or proven whole-solve speedup.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_a4_dir", type=Path)
    parser.add_argument("certified_a5_dir", type=Path)
    args = parser.parse_args()
    if not args.baseline_a4_dir.is_dir() or not args.certified_a5_dir.is_dir():
        parser.error("both result directories must exist")
    return audit(args.baseline_a4_dir, args.certified_a5_dir)


if __name__ == "__main__":
    sys.exit(main())
