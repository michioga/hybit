#!/usr/bin/env python3
"""C1-E: independently audit a C1-D paired crossover evidence directory.

Offline and fail-closed. Never selects a production solver policy. Requires the
C1-D native nine-pair (or other C1-D) data, not older C1-A sweep schema.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

SCHEMA = "hybit-c1d-paired-crossover-v1"
REQUIRED_FIELDS = {
    "phase", "pair", "position", "arm", "pass", "exit_code", "timed_out",
    "hybit_status", "hybit_escalations", "hybit_verified_residual",
    "hybit_iterations", "hybit_wall_ms", "hybit_solve_ms",
    "hybit_diagnostics_ms", "hybit_local_factor_ms", "hybit_factor_mib",
    "peak_rss_kib",
}
COST_FIELDS = (
    "hybit_wall_ms", "hybit_solve_ms", "hybit_diagnostics_ms",
    "hybit_local_factor_ms", "hybit_iterations", "peak_rss_kib",
    "hybit_factor_mib",
)


def finite_number(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        raise ValueError("missing numerical measurement")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numerical measurement {value!r}") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(f"nonfinite or out-of-range measurement {value!r}")
    return number


def signed_finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid signed number {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"nonfinite signed number {value!r}")
    return number


def integer(value: object, *, positive: bool = False) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer {value!r}") from exc
    if result < (1 if positive else 0):
        raise ValueError(f"out-of-range integer {value!r}")
    return result


def yes(value: object) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def quantile(values: list[float], p: float) -> float:
    seq = sorted(values)
    x = p * (len(seq) - 1)
    lo, hi = math.floor(x), math.ceil(x)
    return seq[lo] + (seq[hi] - seq[lo]) * (x - lo)


def get_option(command: list[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"missing or duplicated command flag: {flag}")
    idx = command.index(flag)
    if idx + 1 == len(command):
        raise ValueError(f"missing command value: {flag}")
    return command[idx + 1]


def data_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(metadata: dict, rows: list[dict], summary: dict) -> dict:
    if metadata.get("schema") != SCHEMA:
        raise ValueError("requires C1-D crossover schema; do not mix C1-A/v2 sweep evidence")
    for key in ("matrix_sha256", "binary_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))):
            raise ValueError(f"invalid or missing {key}")
    for key in ("git_head", "command", "pairs", "seed", "threads", "arm_difference"):
        if key not in metadata:
            raise ValueError(f"missing provenance field: {key}")
    if not isinstance(metadata["command"], list):
        raise ValueError("invalid command")
    expected = integer(metadata["pairs"], positive=True)
    tol = finite_number(get_option(metadata["command"], "--tol"), positive=True)
    budget = finite_number(get_option(metadata["command"], "--factor-budget-mib"), positive=True)
    groups: dict[int, dict[str, dict]] = {}
    defects: list[str] = []
    for row in rows:
        if row.get("phase") not in ("warmup", "measure"):
            defects.append(f"unknown phase: {row.get('phase')!r}")
            continue
        if row.get("phase") != "measure":
            continue
        if not REQUIRED_FIELDS.issubset(row):
            defects.append("measurement row lacks required fields")
            continue
        try:
            k = integer(row["pair"], positive=True)
        except ValueError as exc:
            defects.append(str(exc))
            continue
        arms = groups.setdefault(k, {})
        arm = row.get("arm")
        if arm not in ("observe", "hybrid") or arm in arms:
            defects.append(f"pair {k}: unknown or duplicate arm {arm!r}")
        else:
            arms[arm] = row
    if sorted(groups) != list(range(1, expected + 1)):
        defects.append("measured pair IDs do not match metadata pairs")

    details: list[dict] = []
    for k in sorted(groups):
        arms = groups[k]
        if set(arms) != {"observe", "hybrid"}:
            defects.append(f"pair {k}: missing arm")
            continue
        a, b = arms["observe"], arms["hybrid"]
        try:
            if {integer(a["position"], positive=True), integer(b["position"], positive=True)} != {1, 2}:
                raise ValueError("arm positions not complementary")
            for arm, row in (("observe", a), ("hybrid", b)):
                if not yes(row["pass"]) or yes(row["timed_out"]) or integer(row["exit_code"]) != 0:
                    raise ValueError(f"{arm} failed numeric or process gate")
                if row["hybit_status"] != "Converged":
                    raise ValueError(f"{arm} did not converge")
                if finite_number(row["hybit_verified_residual"]) > tol * (1 + 2e-6):
                    raise ValueError(f"{arm} exceeded true CSR residual tolerance")
                if finite_number(row["hybit_factor_mib"]) > budget + 0.0005:
                    raise ValueError(f"{arm} exceeded factor budget")
                integer(row["hybit_iterations"], positive=True)
                finite_number(row["hybit_wall_ms"], positive=True)
                finite_number(row["hybit_solve_ms"], positive=True)
                for field in COST_FIELDS:
                    finite_number(row[field], positive=field in ("hybit_wall_ms", "hybit_solve_ms", "hybit_iterations"))
            if integer(a["hybit_escalations"]) != 0:
                raise ValueError("control unexpectedly escalated")
            if integer(b["hybit_escalations"]) == 0:
                raise ValueError("Hybrid arm never escalated")
        except ValueError as exc:
            defects.append(f"pair {k}: {exc}")
            continue
        costs = {f"delta_{field}": finite_number(b[field]) - finite_number(a[field])
                 for field in COST_FIELDS}
        ratio = finite_number(b["hybit_wall_ms"], positive=True) / finite_number(a["hybit_wall_ms"], positive=True) - 1
        details.append({"pair": k, "hybrid_first": integer(b["position"]) == 1,
                        "relative_cost": ratio, **costs})

    result = {
        "schema": "hybit-c1e-offline-cost-audit-v1",
        "matrix_sha256": metadata["matrix_sha256"],
        "git_head": metadata["git_head"],
        "git_dirty_at_measurement": metadata.get("git_dirty"),
        "rhs_kind": metadata.get("rhs_kind"),
        "measured_pairs_expected": expected,
        "measured_pairs_found": len(groups),
        "valid_escalated_pairs": len(details),
        "fail_closed": bool(defects) or len(details) != expected,
        "defects": defects,
        "median_relative_cost": None,
        "bootstrap_95pct": None,
        "two_sided_sign_p_exploratory": None,
        "median_deltas": None,
        "hybrid_first_pairs": sum(item["hybrid_first"] for item in details),
        "paired_details": details,
        "classification": "insufficient_or_invalid_evidence",
        "c1d_summary_consistent": None,
        "limitations": [
            "matrix/RHS/environment-specific fresh-process comparison, not production policy",
            "reported component times may overlap: do not sum solver, diagnosis and factor as independent costs",
            "preconditioner apply time and PCG restart loss are not separately measurable from these rows",
            "no SPD certification or prepared multi-RHS reuse from this experiment",
        ],
    }
    if result["fail_closed"]:
        return result

    ratios = [p["relative_cost"] for p in details]
    result["median_relative_cost"] = statistics.median(ratios)
    result["median_deltas"] = {
        field: statistics.median(p[f"delta_{field}"] for p in details)
        for field in COST_FIELDS
    }
    if len(ratios) >= 5:
        rng = random.Random(integer(metadata["seed"]))
        medians = [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(4000)]
        lo, hi = quantile(medians, 0.025), quantile(medians, 0.975)
        result["bootstrap_95pct"] = [lo, hi]
        margin = finite_number(summary.get("margin_fraction", 0.05), positive=True)
        result["classification"] = (
            "measured_regression_in_this_matrix" if lo > margin else
            "potential_benefit_in_this_matrix" if hi < -margin else "inconclusive"
        )
        num_pos = sum(x > 0 for x in ratios)
        num_neg = sum(x < 0 for x in ratios)
        non_ties = num_pos + num_neg
        if non_ties:
            # Exact two-sided sign test; exploratory because pairs may share OS
            # state and independent/stationary sampling is not established.
            smaller = min(num_pos, num_neg)
            result["two_sided_sign_p_exploratory"] = min(1.0, 2 * sum(
                math.comb(non_ties, i) for i in range(smaller + 1)) / 2 ** non_ties)
    else:
        result["classification"] = "insufficient_pairs_for_classification"

    their_ci = summary.get("bootstrap_95pct")
    our_ci = result["bootstrap_95pct"]
    ci_match = ((their_ci is None and our_ci is None) or
                (isinstance(their_ci, list) and our_ci is not None and
                 len(their_ci) == len(our_ci) == 2 and
                 all(math.isclose(signed_finite_number(x), y, rel_tol=1e-10, abs_tol=1e-10)
                     for x, y in zip(their_ci, our_ci))))
    match = (summary.get("measured_pairs") == expected and
             summary.get("valid_pairs") == len(details) and
             summary.get("escalated_pairs") == len(details) and
             summary.get("median_relative_cost") is not None and
             math.isclose(signed_finite_number(summary["median_relative_cost"]),
                          result["median_relative_cost"], rel_tol=1e-10, abs_tol=1e-10) and
             ci_match and
             summary.get("classification") == result["classification"])
    result["c1d_summary_consistent"] = match
    if not match:
        result["fail_closed"] = True
        result["defects"].append("C1-D summary disagrees with raw measured rows")
        result["classification"] = "inconsistent_evidence"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write audit to a NEW path (never overwrite)")
    args = parser.parse_args()
    folder = args.evidence_dir.expanduser().resolve()
    try:
        meta = json.loads((folder / "metadata.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        with (folder / "results.csv").open(newline="") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                raise ValueError("not a valid C1-D result CSV")
            rows = list(reader)
        report = audit(meta, rows, summary)
        report["evidence_hashes"] = {
            file: data_digest(folder / file)
            for file in ("metadata.json", "results.csv", "summary.json")
        }
        if args.output:
            target = args.output.expanduser().resolve()
            if target.exists():
                raise ValueError("output already exists; refusing overwrite")
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if report["fail_closed"] else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"C1-E audit rejected input: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
