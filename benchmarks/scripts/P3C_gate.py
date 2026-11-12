#!/usr/bin/env python3
"""P3-C synthetic mixed-DOF structural SPD research acceptance gate.

Preconditioner feature ON proves only a controlled local-factor contract;
reference PCG here is NOT production PCG and this is NOT assembled FEM.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

EXPECTED = {
    "balanced_6dof": "accept",
    "beam_rot_1e4": "accept",
    "beam_rot_1e8": "accept",
    "shell_rot_staggered": "accept",
    "beam_rot_large_1e4": "accept",
    "beam_uniform_1e16": "accept",
    "nonsym_rot_1e8": "reject",
    "indefinite_rot": "reject",
}
FIELDS = (
    "full_relres", "full_xerror", "factor_bytes", "estimated_bytes",
    "full_positive", "overlap_positive", "jacobi_status", "jacobi_iters",
    "jacobi_true_relres", "overlap_status", "overlap_iters",
    "overlap_true_relres", "full_status", "full_iters", "full_true_relres",
)
MAX_ITERS = 96
TOL = 1e-10


class GateError(ValueError):
    pass


def parse(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("P3C_CASE|"):
            row = {}
            for token in line.split("|")[1:]:
                if "=" not in token:
                    raise GateError(f"Malformed case token {token!r}")
                key, value = token.split("=", 1)
                if key in row:
                    raise GateError(f"Duplicate field {key}")
                row[key] = value
            if "name" not in row or row["name"] not in EXPECTED:
                raise GateError("Unknown case name")
            name = row["name"]
            if name in result:
                raise GateError(f"Duplicate case: {name}")
            if not set(FIELDS).issubset(row) or not {"expected", "observed"}.issubset(row):
                raise GateError(f"Missing fields for {name}")
            if row["expected"] != EXPECTED[name] or row["observed"] not in {"accept", "reject"}:
                raise GateError(f"Wrong case classification: {name}")
            result[name] = row
        elif line.startswith("P3C_SUMMARY|"):
            if summary is not None:
                raise GateError("Duplicate summary")
            summary = dict(t.split("=", 1) for t in line.split("|")[1:])
    if set(result) != set(EXPECTED):
        raise GateError(f"Missing case(s): {sorted(set(EXPECTED) - set(result))}")
    for key, expected in {
        "cases": "8", "nodes": "4", "dofs_per_node": "6",
        "reference_pcg_max_iters": "96",
    }.items():
        if summary is None or summary.get(key) != expected:
            raise GateError(f"Unexpected summary {key}: {summary}")
    if abs(float(summary["reference_pcg_tol"]) - TOL) > 1e-25:
        raise GateError("Wrong reference PCG tolerance")
    return result


def finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (ValueError, KeyError) as exc:
        raise GateError(f"Invalid float {key}") from exc
    if not math.isfinite(value):
        raise GateError(f"Nonfinite {key}")
    return value


def check_row(name: str, row: dict[str, str], strict: bool) -> None:
    observed = row["observed"]
    for key in ("full_relres", "full_xerror", "jacobi_true_relres", "overlap_true_relres", "full_true_relres"):
        if finite(row, key) < 0.0:
            raise GateError(f"Negative metric {name}: {key}")
    for key in ("factor_bytes", "estimated_bytes"):
        if not row[key].isdigit():
            raise GateError(f"Invalid bytes {name}: {key}")
    if observed == "reject":
        if row["factor_bytes"] != "0" or row["estimated_bytes"] != "0":
            raise GateError(f"Rejected case has factor allocation {name}")
        if any(row[f"{label}_status"] != "NA" for label in ("jacobi", "overlap", "full")):
            raise GateError(f"Rejected case ran reference PCG {name}")
        return
    if row["expected"] != "accept":
        if strict:
            raise GateError(f"Unsound acceptance {name}")
        if row["jacobi_status"] != "NA":
            raise GateError(f"Baseline ran PCG on non-SPD {name}")
        return
    if row["full_positive"] != "true" or row["overlap_positive"] != "true":
        raise GateError(f"Non-positive preconditioner {name}")
    if int(row["factor_bytes"]) <= 0 or row["factor_bytes"] != row["estimated_bytes"]:
        raise GateError(f"Factor bytes mismatch {name}")
    if finite(row, "full_relres") > 1.0e-8:
        raise GateError(f"Large full-local true residual {name}")
    for label in ("jacobi", "overlap", "full"):
        status = row[f"{label}_status"]
        if status not in {"Converged", "MaxIterations", "Breakdown"}:
            raise GateError(f"Invalid PCG status {name}: {label}")
        iterations = row[f"{label}_iters"]
        if not iterations.isdigit() or int(iterations) > MAX_ITERS:
            raise GateError(f"Reference PCG iteration cap exceeded {name}: {label}")
        residual = finite(row, f"{label}_true_relres")
        if status == "Converged" and residual > TOL * (1.0 + 1.0e-10):
            raise GateError(f"False certified PCG convergence {name}: {label}")


def evaluate(off: dict[str, dict[str, str]], on: dict[str, dict[str, str]]) -> dict:
    for name, row in off.items():
        check_row(name, row, strict=False)
    for name, row in on.items():
        check_row(name, row, strict=True)
    on_mismatches = [name for name, row in on.items() if row["observed"] != EXPECTED[name]]
    off_mismatches = [name for name, row in off.items() if row["observed"] != EXPECTED[name]]
    if on_mismatches:
        raise GateError(f"Equilibrated classification failed: {on_mismatches}")
    if off["balanced_6dof"]["observed"] != "accept":
        raise GateError("Baseline unexpectedly rejects balanced case")
    for name in EXPECTED:
        if off[name]["observed"] == "accept" and on[name]["observed"] == "accept":
            memory_diff = int(on[name]["factor_bytes"]) - int(off[name]["factor_bytes"])
            if memory_diff != 24 * 8:
                raise GateError(f"Unexpected P3-B diagonal-scale storage delta {name}: {memory_diff}")
    transition_count = 0
    for name in EXPECTED:
        if on[name]["observed"] != "accept":
            continue
        row = on[name]
        if row["jacobi_status"] != "Converged" and row["overlap_status"] == "Converged":
            transition_count += 1
    return {
        "case_count": len(EXPECTED),
        "feature_off_contract_discrepancies": len(off_mismatches),
        "feature_on_contract_discrepancies": len(on_mismatches),
        "accepted_spd_true_residuals_within_1e_minus_8": True,
        "accepted_preconditioner_positive_quadratic_forms": True,
        "factor_memory_accounting_verified": True,
        "reference_pcg_off_jacobi_to_overlap_transitions": transition_count,
        "max_reference_pcg_iterations": MAX_ITERS,
        "note": "Constructed 4-node x 6-DOF SPD model; NOT an assembled beam/shell FEM or production PCG rescue.",
    }


def main(args: list[str]) -> int:
    if len(args) != 3:
        print("Usage: python3 -B -m benchmarks.scripts.P3C_gate OFF.log ON.log OUTPUT_DIR", file=sys.stderr)
        return 2
    off, on = parse(Path(args[0])), parse(Path(args[1]))
    result = evaluate(off, on)
    out = Path(args[2])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "p3c_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "name", "expected", "observed", *FIELDS])
        writer.writeheader()
        for feature, group in (("off", off), ("on", on)):
            for name in EXPECTED:
                writer.writerow({"feature": feature, **group[name]})
    (out / "p3c_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("P3-C SYNTHETIC MIXED-DOF NUMERICAL GATE PASS (research-only)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (GateError, OSError, ValueError) as exc:
        print(f"P3-C GATE FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
