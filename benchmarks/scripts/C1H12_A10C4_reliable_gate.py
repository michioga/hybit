#!/usr/bin/env python3
"""Fail-closed A10C4 original-CSR reliable-restart/tolerance numerical gate."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import ARMS, POLICIES, read_time, parse_pipe
from benchmarks.scripts.C1H12_A10C3_trace_gate import finite, integer

CASE = "multi_patch"
RHS = ("Ones", "PointLoad")
TOLS = (1e-10, 1e-11, 1e-12)
PERIODS = (0, 25, 50)
MAX_ITER = 1200
CHECK = "C1H12A10C4_CHECK"
FINAL = "C1H12A10C4_FINAL"
SESSION = "C1H12A10C4_SESSION"

IDENTITY = {"case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "replacement_period"}
CHECK_KEYS = IDENTITY | {
    "iteration", "recursive_relative", "true_relative", "compensated_relative",
    "gap_ratio", "replacement_applied", "precond_calls",
}
FINAL_KEYS = IDENTITY | {
    "status", "iterations", "recursive_relative", "true_relative",
    "compensated_relative", "precond_calls", "true_checks", "replacements",
    "false_events", "best_true_relative", "relative_x_error",
}
SESSION_KEYS = {
    "case", "arm", "policy", "rows", "nnz", "rhs_count", "tol_count",
    "period_count", "max_iterations", "level_dims", "level_nnz", "abtm_words",
}
CHECK_COLUMNS = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "replacement_period",
    "iteration", "recursive_relative", "true_relative", "compensated_relative",
    "gap_ratio", "replacement_applied", "precond_calls",
)
FINAL_COLUMNS = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "replacement_period",
    "status", "iterations", "recursive_relative", "true_relative", "compensated_relative",
    "precond_calls", "true_checks", "replacements", "false_events", "best_true_relative",
    "relative_x_error", "max_gap_ratio", "process_elapsed_s", "process_peak_rss_kib",
)


def parse_records(path: Path) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    sessions, checks, finals = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(SESSION + "|"):
            row = parse_pipe(line, SESSION)
            if set(row) != SESSION_KEYS:
                raise ValueError("A10C4 session schema mismatch")
            sessions.append(row)
        elif line.startswith(CHECK + "|"):
            row = parse_pipe(line, CHECK)
            if set(row) != CHECK_KEYS:
                raise ValueError("A10C4 checkpoint schema mismatch")
            checks.append(row)
        elif line.startswith(FINAL + "|"):
            row = parse_pipe(line, FINAL)
            if set(row) != FINAL_KEYS:
                raise ValueError("A10C4 final schema mismatch")
            finals.append(row)
    if len(sessions) != 1 or len(finals) != 18 or not checks:
        raise ValueError("A10C4 incomplete session, 18 finals or checkpoints")
    return sessions[0], checks, finals


def read_frozen_c3(frozen: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with (frozen / "trace_comparison.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "case", "arm", "policy", "rhs_mode", "rhs_hash", "trace_status",
            "trace_iterations", "trace_recursive_relative", "trace_true_relative",
            "trace_compensated_relative", "trace_precond_calls",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("A10C3 reference schema mismatch")
        records = list(reader)
    if len(records) != 24:
        raise ValueError("A10C3 must have 24 independent reference outcomes")
    result = {}
    for row in records:
        key = (row["arm"], row["policy"], row["rhs_mode"])
        if row["case"] != CASE or key in result:
            raise ValueError("A10C3 reference identity duplicated or mismatched")
        result[key] = row
    expected = {(a, p, r) for a in ARMS for p in POLICIES.values()
                for r in ("Ones", "PointLoad", "IndexSmooth", "SeededRandom")}
    if set(result) != expected:
        raise ValueError("A10C3 missing frozen arm/policy/RHS")
    return result


def check_frozen_hashes(evidence: Path, frozen: Path) -> None:
    lines = (evidence / "FROZEN_A10C3_SHA256.txt").read_text(encoding="utf-8").splitlines()
    expected = {}
    for name in ("trace_comparison.csv", "residual_trace.csv"):
        path = frozen / name
        expected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    found = {}
    for line in lines:
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if not match or match.group(2) in found:
            raise ValueError("A10C4 invalid frozen A10C3 fingerprint schema")
        found[match.group(2)] = match.group(1)
    if found != expected:
        raise ValueError("A10C4 frozen A10C3 evidence hashes changed")


def check_session(row: dict[str, str], arm: str, policy: str) -> None:
    if (row["case"] != CASE or row["arm"] != arm or row["policy"] != policy
            or integer(row["rows"], "rows") != 20736
            or integer(row["nnz"], "nnz") != 103104
            or integer(row["rhs_count"], "rhs_count") != 2
            or integer(row["tol_count"], "tol_count") != 3
            or integer(row["period_count"], "period_count") != 3
            or integer(row["max_iterations"], "max_iterations") != MAX_ITER):
        raise ValueError("A10C4 process parameters changed")
    dimensions = [integer(part, "level_dims", 1) for part in row["level_dims"].split(",")]
    nnz = [integer(part, "level_nnz") for part in row["level_nnz"].split(",")]
    if (len(dimensions) < 2 or len(dimensions) != len(nnz)
            or dimensions[0] != 20736 or nnz[0] != 103104
            or any(left <= right for left, right in zip(dimensions, dimensions[1:]))):
        raise ValueError("A10C4 invalid hierarchy dimensions or nnz")
    if policy == "Multilevel" and integer(row["abtm_words"], "abtm_words", 1) == 0:
        raise ValueError("multilevel ABTM hierarchy words missing")


def validate_group(
    checks: list[dict[str, str]], result: dict[str, str],
    key: tuple[str, str, str, float, int], frozen: dict[str, str],
) -> float:
    arm, policy, rhs_mode, tol, period = key
    identity = {"case": CASE, "arm": arm, "policy": policy, "rhs_mode": rhs_mode,
                "rhs_hash": frozen["rhs_hash"]}
    if not checks or any(any(row[name] != value for name, value in identity.items()) for row in checks):
        raise ValueError("A10C4 checkpoint case/RHS/arm identity mismatch")
    if any(result[name] != value for name, value in identity.items()):
        raise ValueError("A10C4 final case/RHS/arm identity mismatch")
    for row in [*checks, result]:
        if finite(row["tol"], "tol") != tol or integer(row["replacement_period"], "period") != period:
            raise ValueError("A10C4 tolerance or replacement policy mismatch")
    step_max = integer(result["iterations"], "iterations")
    if step_max > MAX_ITER:
        raise ValueError("A10C4 work exceeds iteration budget")
    steps = [integer(row["iteration"], "iteration") for row in checks]
    if steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != step_max:
        raise ValueError("A10C4 duplicate, out-of-order or missing final checkpoint")
    mandatory = {0, step_max}
    if step_max:
        mandatory.add(1)
    mandatory.update(range(25, step_max + 1, 25))
    if not mandatory.issubset(steps):
        raise ValueError("A10C4 mandatory 25-step CSR checkpoint missing")
    replacement_count = 0
    false_count = 0
    verified_values = []
    for row, step in zip(checks, steps):
        recursive = finite(row["recursive_relative"], "recursive_relative")
        verified = finite(row["true_relative"], "true_relative")
        compensated = finite(row["compensated_relative"], "compensated_relative")
        ratio = finite(row["gap_ratio"], "gap_ratio")
        if not math.isclose(ratio, verified / max(recursive, sys.float_info.min), rel_tol=3e-12, abs_tol=1e-12):
            raise ValueError("A10C4 residual-gap value does not match original CSR")
        if integer(row["precond_calls"], "precond_calls") != step:
            raise ValueError("A10C4 preconditioner work inconsistent with iteration")
        if step == 0 and any(abs(value - 1.0) > 2e-14 for value in (recursive, verified, compensated)):
            raise ValueError("A10C4 zero-start residual must be one")
        replaced = integer(row["replacement_applied"], "replacement_applied")
        if replaced not in (0, 1):
            raise ValueError("A10C4 invalid replacement flag")
        false = recursive <= tol and (verified > tol or compensated > tol)
        if false:
            false_count += 1
        certified = recursive <= tol and verified <= tol and compensated <= tol
        expected_replace = int(period > 0 and 0 < step < MAX_ITER
                               and not certified and (step % period == 0 or false))
        if replaced != expected_replace:
            raise ValueError("A10C4 replacement policy does not match independently certified work")
        replacement_count += replaced
        verified_values.append(verified)
    if integer(result["true_checks"], "true_checks") != len(checks):
        raise ValueError("A10C4 true residual check count inconsistent")
    if integer(result["replacements"], "replacements") != replacement_count:
        raise ValueError("A10C4 replacement work count inconsistent")
    if integer(result["false_events"], "false_events") != false_count:
        raise ValueError("A10C4 false-convergence event count inconsistent")
    calls = integer(result["precond_calls"], "precond_calls")
    if calls < step_max or calls > step_max + 1:
        raise ValueError("A10C4 final preconditioner call count inconsistent")
    last = checks[-1]
    for name in ("recursive_relative", "true_relative", "compensated_relative"):
        if not math.isclose(finite(result[name], name), finite(last[name], name), rel_tol=3e-12, abs_tol=1e-14):
            raise ValueError("A10C4 final residual differs from last checkpoint")
    if not math.isclose(finite(result["best_true_relative"], "best_true_relative"),
                        min(verified_values), rel_tol=3e-12, abs_tol=1e-14):
        raise ValueError("A10C4 best verified true residual inconsistent")
    if rhs_mode == "PointLoad":
        if result["relative_x_error"] != "n/a":
            raise ValueError("point load has no manufactured exact solution")
    else:
        finite(result["relative_x_error"], "relative_x_error")
    status = result["status"]
    rec = finite(result["recursive_relative"], "recursive_relative")
    verified = finite(result["true_relative"], "true_relative")
    compensated = finite(result["compensated_relative"], "compensated_relative")
    if status == "Converged" and not (rec <= tol and verified <= tol and compensated <= tol):
        raise ValueError("A10C4 false original-CSR convergence claim")
    if status == "FalseConvergence" and not (period == 0 and rec <= tol and (verified > tol or compensated > tol)):
        raise ValueError("A10C4 invalid false-convergence marker")
    if status == "MaxIterations" and step_max != MAX_ITER:
        raise ValueError("A10C4 premature maximum-iteration status")
    if status not in {"Converged", "FalseConvergence", "MaxIterations", "Breakdown"}:
        raise ValueError("A10C4 invalid solver status")
    if tol == 1e-12 and period == 0:
        if status != frozen["trace_status"] or step_max != integer(frozen["trace_iterations"], "frozen iterations"):
            raise ValueError("A10C4 no-replacement control changed frozen A10C3 outcome")
        for field, source in (
            ("recursive_relative", "trace_recursive_relative"),
            ("true_relative", "trace_true_relative"),
            ("compensated_relative", "trace_compensated_relative"),
        ):
            if not math.isclose(finite(result[field], field), finite(frozen[source], source), rel_tol=3e-11, abs_tol=1e-14):
                raise ValueError("A10C4 no-replacement control diverges from frozen A10C3")
    return max(finite(row["gap_ratio"], "gap_ratio") for row in checks)


def validate(
    a10c3: Path, a10b: Path, run_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    for left, right in (
        (run_dir / "HEAD.txt", a10c3 / "HEAD.txt"),
        (run_dir / "FROZEN_A10C3_HEAD.txt", a10c3 / "HEAD.txt"),
        (run_dir / "FROZEN_A10B_HEAD.txt", a10b / "HEAD.txt"),
        (run_dir / "matrix_sha256.txt", a10c3 / "matrix_sha256.txt"),
        (a10c3 / "matrix_sha256.txt", a10b / "matrix_sha256.txt"),
    ):
        if left.read_bytes() != right.read_bytes():
            raise ValueError("A10C4 frozen HEAD or original CSR provenance mismatch")
    check_frozen_hashes(run_dir, a10c3)
    reference = read_frozen_c3(a10c3)
    all_checks, all_finals = [], []
    for arm in ARMS:
        for short_policy, policy in POLICIES.items():
            stem = f"multi_patch_{arm}_{short_policy}_1"
            session, checks, finals = parse_records(run_dir / f"{stem}.log")
            check_session(session, arm, policy)
            elapsed, peak = read_time(run_dir / f"{stem}.time")
            groups: dict[tuple[str, float, int], list[dict[str, str]]] = defaultdict(list)
            for row in checks:
                groups[(row["rhs_mode"], finite(row["tol"], "tol"), integer(row["replacement_period"], "period"))].append(row)
            results = {}
            for row in finals:
                key = (row["rhs_mode"], finite(row["tol"], "tol"), integer(row["replacement_period"], "period"))
                if key in results:
                    raise ValueError("A10C4 duplicate final status for one policy/RHS")
                results[key] = row
            expected = {(rhs, tol, period) for rhs in RHS for tol in TOLS for period in PERIODS}
            if set(groups) != expected or set(results) != expected:
                raise ValueError("A10C4 incomplete 18-policy/RHS grid")
            for rhs, tol, period in sorted(expected):
                key = (arm, policy, rhs, tol, period)
                gap = validate_group(groups[(rhs, tol, period)], results[(rhs, tol, period)],
                                     key, reference[arm, policy, rhs])
                all_checks.extend(groups[(rhs, tol, period)])
                all_finals.append({**results[(rhs, tol, period)], "max_gap_ratio": f"{gap:.15e}",
                                   "process_elapsed_s": f"{elapsed:.6f}", "process_peak_rss_kib": str(peak)})
    if len(all_finals) != 108:
        raise ValueError("A10C4 expected exactly 108 independent outcomes")
    return all_checks, all_finals


def write_csv(path: Path, columns: tuple[str, ...], records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: C1H12_A10C4_reliable_gate.py FROZEN_A10C3 FROZEN_A10B OUTPUT_DIR")
    try:
        output = Path(sys.argv[3])
        checks, results = validate(Path(sys.argv[1]), Path(sys.argv[2]), output)
        write_csv(output / "reliable_residual_trace.csv", CHECK_COLUMNS, checks)
        write_csv(output / "reliable_update_comparison.csv", FINAL_COLUMNS, results)
        for period in PERIODS:
            sub = [row for row in results if int(row["replacement_period"]) == period]
            print(f"  replacement_period={period}: certified Converged={sum(x['status']=='Converged' for x in sub)}/{len(sub)}, "
                  f"FalseConvergence={sum(x['status']=='FalseConvergence' for x in sub)}, "
                  f"MaxIterations={sum(x['status']=='MaxIterations' for x in sub)}, "
                  f"Breakdown={sum(x['status']=='Breakdown' for x in sub)}")
        print(f"C1-H12-A10C4 {len(checks)} original-CSR checkpoints, 108/108 RHS/tolerance/replacement provenance gates PASS")
        print("Scope: experimental restarted PCG only; no production precision/speedup guarantee")
    except (OSError, KeyError, ValueError, OverflowError) as error:
        raise SystemExit(f"C1-H12-A10C4 evidence FAIL: {error}") from error
