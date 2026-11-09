#!/usr/bin/env python3
"""Audit A6 periodic true-residual PCG against same-binary A5 control.

No timing advantage or successful rescue is assumed. A convergence claim is
accepted only against the independent original-CSR true residual.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    from benchmarks.scripts.C1H12_A4_numerical_gate import (
        ARMS, CASES, parse_record, parse_time, validate_record,
    )
    from benchmarks.scripts.C1H12_A5_certification_gate import parse_a5_record
except ModuleNotFoundError:
    from C1H12_A4_numerical_gate import (
        ARMS, CASES, parse_record, parse_time, validate_record,
    )
    from C1H12_A5_certification_gate import parse_a5_record


POLICIES = ("a5", "p16", "p32")
A6_FIELDS = {
    "policy", "check_every", "trigger_ratio", "true_checks", "periodic_checks",
    "restarts", "proactive_restarts", "rejected_recursive_convergences",
    "peak_gap_ratio", "check_ms", "final_true_relative", "global_iterations", "status",
}
NUMERIC_FIELDS = ("trigger_ratio", "peak_gap_ratio", "check_ms", "final_true_relative")
COUNT_FIELDS = (
    "check_every", "true_checks", "periodic_checks", "restarts",
    "proactive_restarts", "rejected_recursive_convergences", "global_iterations",
)
CSV_FIELDS = (
    "case", "arm", "policy", "status", "iterations", "true_relative", "reported_relative",
    "true_checks", "restarts", "proactive_restarts", "rejected_recursive_convergences",
    "peak_gap_ratio", "check_ms", "setup_ms", "solve_ms", "total_wall_ms",
    "elapsed_s", "peak_rss_kib", "coarse_factor_bytes",
    "a5_status", "a5_iterations", "a5_true_relative", "a5_solve_ms",
)


def parse_a6_record(log: str, expected_interval: int) -> dict[str, int | float | str]:
    lines = [line for line in log.splitlines() if line.startswith("C1H12A6|")]
    if len(lines) != 1:
        raise ValueError(f"expected one A6 row, found {len(lines)}")
    raw: dict[str, str] = {}
    for field in lines[0].split("|")[1:]:
        if "=" not in field:
            raise ValueError(f"malformed A6 token: {field!r}")
        name, value = field.split("=", 1)
        if not name or not value or name in raw:
            raise ValueError(f"empty or duplicate A6 field: {name!r}")
        raw[name] = value
    if raw.keys() != A6_FIELDS:
        raise ValueError(f"A6 schema mismatch: {raw.keys() ^ A6_FIELDS}")

    row: dict[str, int | float | str] = dict(raw)
    for name in COUNT_FIELDS:
        row[name] = int(raw[name])
        if row[name] < 0:
            raise ValueError(f"negative A6 count: {name}")
    for name in NUMERIC_FIELDS:
        row[name] = float(raw[name])
        if not math.isfinite(row[name]) or row[name] < 0:
            raise ValueError(f"invalid A6 measurement: {name}")

    if row["policy"] != "PeriodicCertifiedRestart":
        raise ValueError("wrong A6 policy")
    if row["check_every"] != expected_interval or row["trigger_ratio"] != 2.0:
        raise ValueError("unexpected A6 interval or gap threshold")
    if row["true_checks"] == 0 or row["periodic_checks"] != row["true_checks"]:
        raise ValueError("periodic check count mismatch")
    if row["global_iterations"] > 600:
        raise ValueError("A6 exceeded global iteration budget")
    if row["restarts"] != row["proactive_restarts"] + row["rejected_recursive_convergences"]:
        raise ValueError("A6 restart count disagrees with causes")
    if row["restarts"] > row["true_checks"]:
        raise ValueError("A6 restart count exceeds observations")
    if row["status"] not in ("Converged", "MaxIterations", "Breakdown"):
        raise ValueError("bad A6 solver status")
    return row


def verify_record(log: str, timing: str, case: str, arm: str, policy: str) -> dict:
    if "C1H12A5|" in log and policy != "a5":
        raise ValueError("unexpected A5 record in A6 log")
    if "C1H12A6|" in log and policy == "a5":
        raise ValueError("unexpected A6 record in A5 log")

    parsed_time = parse_time(timing)
    if parsed_time["exit_status"] != 0:
        raise ValueError(f"process exit={parsed_time['exit_status']}")
    common = validate_record(parse_record(log), case, arm)
    assert common["iterations"] <= 600
    if policy == "a5":
        detail = parse_a5_record(log)
        detail["proactive_restarts"] = 0
        detail["peak_gap_ratio"] = 0.0
        detail["check_ms"] = 0.0
    else:
        detail = parse_a6_record(log, int(policy[1:]))

    if detail["global_iterations"] != common["iterations"] or detail["status"] != common["status"]:
        raise ValueError("diagnostic status/iteration mismatch")
    if not math.isclose(
        detail["final_true_relative"], common["verified_true_relative"],
        rel_tol=1e-9, abs_tol=1e-14,
    ):
        raise ValueError("certificate and A4 true residual disagree")
    if detail["check_ms"] > common["solve_ms"] + 0.1:
        raise ValueError("A6 checking cost exceeds solve duration")
    return {"common": common, "detail": detail, "time": parsed_time}


def audit(previous_a5: Path, run_dir: Path) -> int:
    errors: list[str] = []
    collected: dict[tuple[str, str, str], dict] = {}

    for stamp in ("HEAD.txt", "matrix_sha256.txt"):
        if not (previous_a5 / stamp).is_file() or not (run_dir / stamp).is_file():
            errors.append(f"missing provenance: {stamp}")
    historical_hashes = previous_a5 / "matrix_sha256.txt"
    current_hashes = run_dir / "matrix_sha256.txt"
    if historical_hashes.is_file() and current_hashes.is_file():
        if historical_hashes.read_bytes() != current_hashes.read_bytes():
            errors.append("A5/A6 matrix SHA-256 provenance differs")
    # Source HEAD must differ: the A5 evidence was collected before its own
    # acceptance commit. Preserve both commits, do not require equality.

    expected = {
        f"{case}_{arm}_{policy}_1"
        for case in CASES for arm in ARMS for policy in POLICIES
    }
    for ext in ("log", "time"):
        seen = {p.stem for p in run_dir.glob(f"*.{ext}")}
        for missing in sorted(expected - seen):
            errors.append(f"missing {missing}.{ext}")
        for extra in sorted(seen - expected):
            errors.append(f"unexpected {extra}.{ext}")

    for case in CASES:
        for arm in ARMS:
            previous_log = previous_a5 / f"{case}_{arm}_1.log"
            if not previous_log.is_file():
                errors.append(f"missing historical A5 log: {case}/{arm}")
                continue
            try:
                previous = validate_record(
                    parse_record(previous_log.read_text(encoding="utf-8")), case, arm
                )
                if "C1H12A5|" not in previous_log.read_text(encoding="utf-8"):
                    raise ValueError("historical A5 certified record missing")
            except (OSError, ValueError, UnicodeError) as exc:
                errors.append(f"historical A5 {case}/{arm}: {exc}")
                continue
            for policy in POLICIES:
                stem = f"{case}_{arm}_{policy}_1"
                log_path = run_dir / f"{stem}.log"
                time_path = run_dir / f"{stem}.time"
                if not log_path.is_file() or not time_path.is_file():
                    continue
                try:
                    result = verify_record(
                        log_path.read_text(encoding="utf-8"),
                        time_path.read_text(encoding="utf-8"), case, arm, policy,
                    )
                    if policy == "a5" and result["common"]["status"] != previous["status"]:
                        raise ValueError("same-binary A5 control no longer matches historical status")
                    collected[(case, arm, policy)] = result
                except (OSError, ValueError, UnicodeError) as exc:
                    errors.append(f"{stem}: {exc}")

    rows: list[dict[str, str | int | float]] = []
    for case in CASES:
        for arm in ARMS:
            control = collected.get((case, arm, "a5"))
            for policy in POLICIES:
                current = collected.get((case, arm, policy))
                if current is None:
                    continue
                common = current["common"]
                detail = current["detail"]
                timing = current["time"]
                rows.append({
                    "case": case, "arm": arm, "policy": policy,
                    "status": common["status"], "iterations": common["iterations"],
                    "true_relative": common["verified_true_relative"],
                    "reported_relative": common["reported_relative"],
                    "true_checks": detail["true_checks"], "restarts": detail["restarts"],
                    "proactive_restarts": detail["proactive_restarts"],
                    "rejected_recursive_convergences": detail["rejected_recursive_convergences"],
                    "peak_gap_ratio": detail["peak_gap_ratio"], "check_ms": detail["check_ms"],
                    "setup_ms": common["setup_ms"], "solve_ms": common["solve_ms"],
                    "total_wall_ms": common["total_wall_ms"],
                    "elapsed_s": timing["elapsed_s"], "peak_rss_kib": timing["peak_rss_kib"],
                    "coarse_factor_bytes": common["coarse_factor_bytes"],
                    "a5_status": control["common"]["status"] if control else "MISSING",
                    "a5_iterations": control["common"]["iterations"] if control else -1,
                    "a5_true_relative": control["common"]["verified_true_relative"] if control else -1,
                    "a5_solve_ms": control["common"]["solve_ms"] if control else -1,
                })

    with (run_dir / "periodic_pcg_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"C1-H12-A6 {len(collected)}/63 numerical/log records complete")
    for policy in POLICIES:
        items = [c["common"] for (case, arm, p), c in collected.items() if p == policy]
        print(
            f"  {policy}: Converged={sum(c['status'] == 'Converged' for c in items)}, "
            f"MaxIterations={sum(c['status'] == 'MaxIterations' for c in items)}, "
            f"Breakdown={sum(c['status'] == 'Breakdown' for c in items)}"
        )
        for case in CASES:
            representative = collected.get((case, "strong_graph_smoothed", policy))
            if representative is not None:
                c = representative["common"]
                d = representative["detail"]
                print(f"    {case}/strong_graph_smoothed: {c['status']} {c['iterations']} it "
                      f"true={c['verified_true_relative']:.3e}, restarts={d['restarts']}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"CSV (partial diagnostic, NOT accepted): {run_dir / 'periodic_pcg_comparison.csv'}")
        return 1
    if len(collected) != 63:
        print("FAIL: incomplete A6 run", file=sys.stderr)
        return 1
    print("C1-H12-A6 63/63 original-CSR numerical, policy and provenance gates PASS")
    print("Scope: single-process trial, no native multilevel AMG or runtime speedup claim.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_a5_dir", type=Path)
    parser.add_argument("a6_run_dir", type=Path)
    opts = parser.parse_args()
    if not opts.previous_a5_dir.is_dir() or not opts.a6_run_dir.is_dir():
        parser.error("existing A5 and A6 evidence directories required")
    return audit(opts.previous_a5_dir, opts.a6_run_dir)


if __name__ == "__main__":
    sys.exit(main())
