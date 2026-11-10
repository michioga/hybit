#!/usr/bin/env python3
"""Fail-closed A10C3 original-CSR residual-trace and frozen A10B status gate."""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import (
    ARMS, POLICIES, RHS_NATIVE, read_time, parse_pipe,
)

TRACE = "C1H12A10C3_TRACE"
FINAL = "C1H12A10C3_FINAL"
SESSION = "C1H12A10C3_SESSION"
CASE = "multi_patch"
TOL = 1e-12
MAX_ITER = 1200
TRACE_EVERY = 25

IDENTITY = {"case", "arm", "policy", "rhs_mode", "rhs_hash"}
TRACE_NUMBERS = {"recursive_relative", "true_relative", "compensated_relative", "gap_ratio"}
TRACE_INTS = {"iteration", "precond_calls"}
TRACE_KEYS = IDENTITY | TRACE_NUMBERS | TRACE_INTS
FINAL_NUMBERS = {
    "trace_recursive_relative", "trace_true_relative", "trace_compensated_relative",
    "baseline_true_relative", "baseline_reported_relative", "baseline_compensated_relative",
}
FINAL_INTS = {
    "trace_iterations", "trace_precond_calls", "baseline_iterations",
    "baseline_true_checks", "baseline_restarts",
}
FINAL_KEYS = IDENTITY | FINAL_NUMBERS | FINAL_INTS | {
    "trace_status", "baseline_status", "trace_relative_x_error",
}
SESSION_KEYS = {
    "case", "arm", "policy", "rows", "nnz", "tol", "max_iterations",
    "trace_every", "rhs_count", "levels", "level_dims", "level_nnz", "abtm_words",
}
TRACE_CSV = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "iteration",
    "recursive_relative", "true_relative", "compensated_relative",
    "gap_ratio", "precond_calls",
)
FINAL_CSV = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash",
    "trace_status", "trace_iterations", "trace_recursive_relative",
    "trace_true_relative", "trace_compensated_relative", "trace_precond_calls",
    "trace_relative_x_error", "baseline_status", "baseline_iterations",
    "baseline_true_relative", "baseline_reported_relative",
    "baseline_compensated_relative", "baseline_true_checks", "baseline_restarts",
    "max_observed_gap_ratio", "time_elapsed_s", "peak_rss_kib",
)


def finite(value: str, name: str, minimum: float = 0.0) -> float:
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"nonfinite or invalid {name}: {value}")
    return number


def integer(value: str, name: str, minimum: int = 0) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value) or int(value) < minimum:
        raise ValueError(f"invalid {name}: {value}")
    return int(value)


def parse_rows(path: Path) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    sessions, traces, finals = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(SESSION + "|"):
            item = parse_pipe(line, SESSION)
            if set(item) != SESSION_KEYS:
                raise ValueError("A10C3 session schema mismatch")
            sessions.append(item)
        elif line.startswith(TRACE + "|"):
            item = parse_pipe(line, TRACE)
            if set(item) != TRACE_KEYS:
                raise ValueError("A10C3 checkpoint schema mismatch")
            traces.append(item)
        elif line.startswith(FINAL + "|"):
            item = parse_pipe(line, FINAL)
            if set(item) != FINAL_KEYS:
                raise ValueError("A10C3 final schema mismatch")
            finals.append(item)
    if len(sessions) != 1:
        raise ValueError("A10C3 process did not print exactly one session marker")
    if len(finals) != 4 or not traces:
        raise ValueError("A10C3 missing four RHS results or checkpoint records")
    return sessions[0], traces, finals


def frozen_baseline(frozen_a10b: Path) -> dict[tuple[str, str, str], tuple[str, str, int]]:
    with (frozen_a10b / "prepared_reuse_comparison.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"case", "arm", "policy", "rhs_mode", "status", "rhs_hash", "reuse_mode", "iterations"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("frozen A10B comparison schema missing")
        rows = list(reader)
    if len(rows) != 720:
        raise ValueError("frozen A10B record count mismatch")
    expected = {}
    for arm in ARMS:
        for policy in POLICIES.values():
            for rhs in RHS_NATIVE.values():
                subset = [row for row in rows if row["case"] == CASE and row["arm"] == arm
                          and row["policy"] == policy and row["rhs_mode"] == rhs]
                if len(subset) != 10 or {row["reuse_mode"] for row in subset} != {"fresh", "reuse"}:
                    raise ValueError("incomplete frozen multi_patch lifecycle controls")
                hashes = {row["rhs_hash"] for row in subset}
                statuses = {row["status"] for row in subset}
                iterations = {integer(row["iterations"], "frozen iterations") for row in subset}
                if len(hashes) != 1 or len(statuses) != 1 or len(iterations) != 1:
                    raise ValueError("frozen A10B status, iterations or RHS fingerprint is not stable")
                expected[arm, policy, rhs] = statuses.pop(), hashes.pop(), iterations.pop()
    return expected


def check_session(row: dict[str, str], arm: str, policy: str) -> None:
    if (row["case"] != CASE or row["arm"] != arm or row["policy"] != policy
            or integer(row["rows"], "rows") != 20736
            or integer(row["nnz"], "nnz") != 103104
            or finite(row["tol"], "tol") != TOL
            or integer(row["max_iterations"], "max_iterations") != MAX_ITER
            or integer(row["trace_every"], "trace_every") != TRACE_EVERY
            or integer(row["rhs_count"], "rhs_count") != 4):
        raise ValueError("unexpected A10C3 process configuration")
    levels = integer(row["levels"], "levels", 2)
    dimensions = [integer(value, "level dimension", 1) for value in row["level_dims"].split(",")]
    nonzeros = [integer(value, "level nnz") for value in row["level_nnz"].split(",")]
    if (len(dimensions) != levels or len(nonzeros) != levels
            or dimensions[0] != 20736 or nonzeros[0] != 103104
            or any(a <= b for a, b in zip(dimensions, dimensions[1:]))):
        raise ValueError("A10C3 invalid hierarchy")
    if policy == "Multilevel" and integer(row["abtm_words"], "abtm_words", 1) == 0:
        raise ValueError("multilevel must expose ABTM hierarchy words")


def check_group(
    rows: list[dict[str, str]], final: dict[str, str],
    expected: tuple[str, str, int], key: tuple[str, str, str],
) -> float:
    arm, policy, mode = key
    base_status, rhs_hash, base_iterations = expected
    identity = {"case": CASE, "arm": arm, "policy": policy,
                "rhs_mode": mode, "rhs_hash": rhs_hash}
    if not rows or any(any(row[field] != value for field, value in identity.items()) for row in rows):
        raise ValueError("checkpoint RHS or arm identity changed")
    if any(final[field] != value for field, value in identity.items()):
        raise ValueError("final RHS or arm identity changed")
    if final["baseline_status"] != base_status:
        raise ValueError("native baseline disagrees with frozen A10B status")
    trace_status = final["trace_status"]
    if trace_status not in {"Converged", "MaxIterations", "FalseConvergence", "Breakdown"}:
        raise ValueError("invalid independent trace status")
    maximum = integer(final["trace_iterations"], "trace_iterations")
    if maximum > MAX_ITER:
        raise ValueError("trace exceeded solver budget")
    iterations = [integer(row["iteration"], "iteration") for row in rows]
    if iterations != sorted(set(iterations)) or iterations[0] != 0 or iterations[-1] != maximum:
        raise ValueError("missing, duplicate or out-of-order checkpoint")
    expected_checkpoints = {0}
    if maximum:
        expected_checkpoints.add(1)
    expected_checkpoints.update(range(TRACE_EVERY, maximum + 1, TRACE_EVERY))
    expected_checkpoints.add(maximum)
    if set(iterations) != expected_checkpoints:
        raise ValueError("checkpoint cadence changed or incomplete")
    for row in rows:
        count = integer(row["precond_calls"], "precond_calls")
        step = integer(row["iteration"], "iteration")
        if count != step and not (step == 0 and count == 0):
            raise ValueError("preconditioner trace work is inconsistent")
        recursive = finite(row["recursive_relative"], "recursive_relative")
        verified = finite(row["true_relative"], "true_relative")
        finite(row["compensated_relative"], "compensated_relative")
        gap = finite(row["gap_ratio"], "gap_ratio")
        ratio = verified / max(recursive, sys.float_info.min)
        if not math.isclose(gap, ratio, rel_tol=3e-12, abs_tol=1e-12):
            raise ValueError("reported residual-gap ratio is inconsistent")
        if step == 0 and (abs(recursive - 1.0) > 2e-14 or abs(verified - 1.0) > 2e-14):
            raise ValueError("zero-start PCG initial residual is not one")
    final_last = rows[-1]
    for key_name, source in (
        ("trace_recursive_relative", "recursive_relative"),
        ("trace_true_relative", "true_relative"),
        ("trace_compensated_relative", "compensated_relative"),
    ):
        if not math.isclose(finite(final[key_name], key_name), finite(final_last[source], source), rel_tol=3e-12, abs_tol=1e-14):
            raise ValueError("final residual differs from last original-CSR checkpoint")
    calls = integer(final["trace_precond_calls"], "trace_precond_calls")
    if calls < maximum or calls > maximum + 1:
        raise ValueError("independent trace preconditioner calls inconsistent")
    baseline_iter = integer(final["baseline_iterations"], "baseline_iterations")
    if baseline_iter != base_iterations or baseline_iter > MAX_ITER:
        raise ValueError("native certified PCG iterations disagree with frozen A10B")
    base_true = finite(final["baseline_true_relative"], "baseline_true_relative")
    finite(final["baseline_reported_relative"], "baseline_reported_relative")
    finite(final["baseline_compensated_relative"], "baseline_compensated_relative")
    integer(final["baseline_true_checks"], "baseline_true_checks", 1)
    integer(final["baseline_restarts"], "baseline_restarts")
    if base_status == "Converged" and base_true > TOL * (1 + 1e-9):
        raise ValueError("false convergence of certified native baseline")
    if base_status == "MaxIterations" and baseline_iter != MAX_ITER:
        raise ValueError("frozen native baseline MaxIterations work inconsistent")
    relative_x = final["trace_relative_x_error"]
    if mode == "PointLoad":
        if relative_x != "n/a":
            raise ValueError("point-load unknown exact solution must be n/a")
    else:
        finite(relative_x, "trace_relative_x_error")
    rec = finite(final["trace_recursive_relative"], "trace_recursive_relative")
    true = finite(final["trace_true_relative"], "trace_true_relative")
    comp = finite(final["trace_compensated_relative"], "trace_compensated_relative")
    if trace_status == "Converged" and not (rec <= TOL and true <= TOL and comp <= TOL):
        raise ValueError("false certified trace convergence")
    if trace_status == "FalseConvergence" and not (rec <= TOL and (true > TOL or comp > TOL)):
        raise ValueError("false convergence marker inconsistent with residuals")
    if trace_status == "MaxIterations" and maximum != MAX_ITER:
        raise ValueError("premature MaxIterations outcome")
    return max(finite(row["gap_ratio"], "gap_ratio") for row in rows)


def validate(a10b: Path, a10c2: Path, evidence: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if ((evidence / "FROZEN_A10B_HEAD.txt").read_bytes() != (a10b / "HEAD.txt").read_bytes()
            or (evidence / "FROZEN_A10C2_HEAD.txt").read_bytes() != (a10c2 / "HEAD.txt").read_bytes()
            or (evidence / "HEAD.txt").read_bytes() != (a10c2 / "HEAD.txt").read_bytes()
            or (evidence / "matrix_sha256.txt").read_bytes() != (a10b / "matrix_sha256.txt").read_bytes()
            or (a10c2 / "matrix_sha256.txt").read_bytes() != (a10b / "matrix_sha256.txt").read_bytes()):
        raise ValueError("A10C3 frozen source/matrix provenance mismatch")
    if not (a10c2 / "spectral_ritz.csv").is_file() or not (a10c2 / "coarse_projection.csv").is_file():
        raise ValueError("missing frozen A10C2 spectral evidence")
    frozen = frozen_baseline(a10b)
    all_traces, all_finals = [], []
    for arm in ARMS:
        for local_policy, policy in POLICIES.items():
            stem = f"{CASE}_{arm}_{local_policy}_1"
            session, traces, finals = parse_rows(evidence / f"{stem}.log")
            check_session(session, arm, policy)
            seconds, rss = read_time(evidence / f"{stem}.time")
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in traces:
                groups[row["rhs_mode"]].append(row)
            finishes = {}
            for row in finals:
                mode = row["rhs_mode"]
                if mode in finishes:
                    raise ValueError("duplicate RHS final record")
                finishes[mode] = row
            if set(groups) != set(RHS_NATIVE.values()) or set(finishes) != set(RHS_NATIVE.values()):
                raise ValueError("incomplete RHS trace/final grid")
            for mode in RHS_NATIVE.values():
                key = (arm, policy, mode)
                group = groups[mode]
                max_gap = check_group(group, finishes[mode], frozen[key], key)
                all_traces.extend(group)
                all_finals.append({**finishes[mode], "max_observed_gap_ratio": f"{max_gap:.15e}",
                                   "time_elapsed_s": f"{seconds:.6f}", "peak_rss_kib": str(rss)})
    if len(all_finals) != 24:
        raise ValueError("incomplete six-arm/four-RHS native outcome grid")
    return all_traces, all_finals


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: C1H12_A10C3_trace_gate.py FROZEN_A10B FROZEN_A10C2 RUN_DIR")
    try:
        run_dir = Path(sys.argv[3])
        traces, finals = validate(Path(sys.argv[1]), Path(sys.argv[2]), run_dir)
        write_csv(run_dir / "residual_trace.csv", TRACE_CSV, traces)
        write_csv(run_dir / "trace_comparison.csv", FINAL_CSV, finals)
        native = sum(row["baseline_status"] == "Converged" for row in finals)
        traced = sum(row["trace_status"] == "Converged" for row in finals)
        print(f"C1-H12-A10C3 {len(traces)} verified checkpoints, 24/24 original-CSR/frozen-A10B gate PASS")
        print(f"  original certified baseline converged={native}/24; independent trace converged={traced}/24")
        print("Scope: diagnostic textbook PCG vs original certified PCG, not same recurrence/production speedup")
    except (OSError, ValueError, KeyError, StopIteration, OverflowError) as error:
        raise SystemExit(f"C1-H12-A10C3 evidence FAIL: {error}") from error
