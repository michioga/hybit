#!/usr/bin/env python3
"""P3-B independent OFF/ON classification and numerical-contract gate.

Reuses the P3-A parser without rewriting the baseline audit or altering raw logs.
The deliberately ill-conditioned congruence case's forward error is observed,
not confused with the independent true residual.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from benchmarks.scripts.P3A_gate import EXPECTED, parse

BASELINE_MISMATCHES = {"spd_small_1e16", "spd_congruence", "nonsym_small_1e12"}
WELL_SCALED_SPD = {"spd_unit", "spd_small_1e12", "spd_small_1e16", "spd_large_1e16"}


def evaluate(off_text: str, on_text: str):
    off = parse(off_text)
    on = parse(on_text)
    off_mismatches = {n for n, r in off.items() if r["contract_discrepancy"]}
    if off_mismatches != BASELINE_MISMATCHES:
        raise ValueError(f"legacy feature-OFF baseline changed: {sorted(off_mismatches)}")
    on_mismatches = {n for n, r in on.items() if r["contract_discrepancy"]}
    if on_mismatches:
        raise ValueError(f"P3-B unresolved classification discrepancies: {sorted(on_mismatches)}")
    output = []
    for name in EXPECTED:
        a, b = off[name], on[name]
        if a["expected"] != b["expected"] or a["scale"] != b["scale"]:
            raise ValueError(f"OFF/ON inputs changed: {name}")
        if b["observed"] == "accept":
            if b["relative_residual"] > 1.0e-8:
                raise ValueError(f"bad certified true residual: {name}")
            if not (b["factor_bytes"] > 0):
                raise ValueError(f"accepted case reports no local factor: {name}")
            if b["factor_bytes"] != 248:
                raise ValueError(f"wrong equilibrated factor bytes: {name}")
            if name in WELL_SCALED_SPD and b["relative_x_error"] > 1.0e-8:
                raise ValueError(f"well-scaled forward error regression: {name}")
        elif b["factor_bytes"] != 0:
            raise ValueError(f"rejected case owns local factor: {name}")
        output.append({
            "case": name, "expected": b["expected"],
            "off_observed": a["observed"], "on_observed": b["observed"],
            "off_relative_residual": a["relative_residual"],
            "on_relative_residual": b["relative_residual"],
            "off_relative_x_error": a["relative_x_error"],
            "on_relative_x_error": b["relative_x_error"],
            "off_factor_bytes": a["factor_bytes"],
            "on_factor_bytes": b["factor_bytes"],
        })
    summary = {
        "case_count": len(output),
        "feature_off_contract_discrepancies": len(off_mismatches),
        "feature_on_contract_discrepancies": len(on_mismatches),
        "all_accepted_spd_true_residuals_acceptable": True,
        "equilibrated_factor_bytes_per_four_dof_region": 248,
        "ill_conditioned_congruence_forward_error": on["spd_congruence"]["relative_x_error"],
        "note": "P3-B measures local-factor classification and true residual; poor forward accuracy for severely ill-conditioned A remains visible.",
        "not_proven": "No end-to-end FEM rescue or default production promotion is claimed.",
    }
    return output, summary


def gate(off_path: Path, on_path: Path, output_dir: Path):
    rows, summary = evaluate(off_path.read_text(), on_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "p3b_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "p3b_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("off_log", type=Path)
    parser.add_argument("on_log", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        gate(args.off_log, args.on_log, args.output_dir)
    except (ValueError, OSError, OverflowError) as e:
        print(f"P3-B NUMERICAL GATE FAIL: {e}", file=sys.stderr)
        return 1
    print("P3-B NUMERICAL GATE PASS (research opt-in only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
