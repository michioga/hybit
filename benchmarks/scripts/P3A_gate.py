#!/usr/bin/env python3
"""P3-A audit data integrity gate; findings are *not* silently scored PASS."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

CASE_RE = re.compile(r"^P3A_CASE\|(.+)$", re.M)
SUMMARY_RE = re.compile(r"^P3A_SUMMARY\|cases=(\d+)\|contract_discrepancies=(\d+)$", re.M)
EXPECTED = {
    "spd_unit": "accept", "spd_small_1e12": "accept",
    "spd_small_1e16": "accept", "spd_large_1e16": "accept",
    "spd_congruence": "accept", "nonsym_small_1e12": "reject",
    "nonsym_unit": "reject",
}
EXPECTED_SCALE = {
    "spd_unit": 1.0, "spd_small_1e12": 1.0e-12,
    "spd_small_1e16": 1.0e-16, "spd_large_1e16": 1.0e16,
    "spd_congruence": 1.0,
    "nonsym_small_1e12": 1.0e-12, "nonsym_unit": 1.0,
}
FIELDS = {"name", "expected", "observed", "relative_residual",
          "relative_x_error", "factor_bytes", "mathematical_spd", "scale"}


def parse(text):
    cases = {}
    for raw in CASE_RE.findall(text):
        pairs = [s.split("=", 1) for s in raw.split("|")]
        if any(len(p) != 2 for p in pairs):
            raise ValueError("malformed case fields")
        row = dict(pairs)
        if len(row) != len(pairs) or set(row) != FIELDS:
            raise ValueError("duplicate/unknown case fields")
        name = row["name"]
        if name not in EXPECTED or name in cases:
            raise ValueError(f"unknown/duplicate case: {name}")
        if row["expected"] != EXPECTED[name]:
            raise ValueError(f"incorrect mathematical expectation: {name}")
        if row["observed"] not in ("accept", "reject"):
            raise ValueError(f"invalid observed status: {name}")
        if (row["mathematical_spd"] == "true") != (row["expected"] == "accept"):
            raise ValueError(f"mathematical SPD field mismatch: {name}")
        if row["mathematical_spd"] not in ("true", "false"):
            raise ValueError("invalid SPD field")
        for field in ("relative_residual", "relative_x_error", "scale"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}.{field}")
            row[field] = value
        if row["scale"] != EXPECTED_SCALE[name]:
            raise ValueError(f"case scale differs from declared configuration: {name}")
        row["factor_bytes"] = int(row["factor_bytes"])
        if row["factor_bytes"] < 0:
            raise ValueError("negative factor bytes")
        if row["observed"] == "accept" and row["factor_bytes"] == 0:
            raise ValueError("accepted case without factor storage")
        if row["observed"] == "reject" and row["factor_bytes"] != 0:
            raise ValueError("rejected case reporting a factor")
        row["contract_discrepancy"] = int(row["expected"] != row["observed"])
        row["accepted_spd_residual_bad"] = int(
            row["expected"] == "accept" and row["observed"] == "accept"
            and row["relative_residual"] > 1.0e-8
        )
        cases[name] = row
    if set(cases) != set(EXPECTED):
        raise ValueError("missing/extra cases: " + str(sorted(set(cases) ^ set(EXPECTED))))
    summaries = SUMMARY_RE.findall(text)
    if len(summaries) != 1:
        raise ValueError("missing/duplicate P3A_SUMMARY")
    count, discrepancies = map(int, summaries[0])
    actual = sum(r["contract_discrepancy"] for r in cases.values())
    if count != len(cases) or discrepancies != actual:
        raise ValueError("summary/observations disagree")
    return cases


def gate(input_file, output_dir):
    cases = parse(Path(input_file).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(cases.values())
    with (output_dir / "p3a_cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "cases": len(rows),
        "contract_discrepancies": sum(r["contract_discrepancy"] for r in rows),
        "unexpected_spd_rejections": [r["name"] for r in rows if r["expected"] == "accept" and r["observed"] == "reject"],
        "unexpected_nonsymmetric_acceptances": [r["name"] for r in rows if r["expected"] == "reject" and r["observed"] == "accept"],
        "accepted_spd_residual_bad": [r["name"] for r in rows if r["accepted_spd_residual_bad"]],
        "accepted_spd_forward_error_over_1e8": [r["name"] for r in rows if r["expected"] == "accept" and r["observed"] == "accept" and r["relative_x_error"] > 1.0e-8],
        "interpretation": "Measurement integrity PASS does not establish local Cholesky correctness; preserve discrepancies for P3-B."
    }
    (output_dir / "p3a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        gate(args.input_file, args.output_dir)
    except (OSError, ValueError, OverflowError) as e:
        print(f"P3-A DATA INTEGRITY FAIL: {e}", file=sys.stderr)
        return 1
    print("P3-A DATA INTEGRITY PASS (findings are not numerical PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
