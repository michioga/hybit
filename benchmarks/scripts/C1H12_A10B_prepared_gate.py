#!/usr/bin/env python3
"""Fail-closed A10B repeated-RHS preparation lifecycle numerical/provenance gate."""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

CASES = ("island_shell", "multi_patch", "rotated_anisotropy")
ARMS = {
    "graph_piecewise": ("Graph", "PiecewiseConstant"),
    "strong_graph_piecewise": ("StrongGraph", "PiecewiseConstant"),
    "strong_graph_smoothed": ("StrongGraph", "JacobiSmoothed"),
}
POLICIES = {"two_level": "TwoLevel", "multilevel": "Multilevel"}
REUSE = {"fresh": "Fresh", "reuse": "Reuse"}
RHS = ("ones", "index_smooth", "seeded_random", "point_load")
RHS_NATIVE = {
    "ones": "Ones",
    "index_smooth": "IndexSmooth",
    "seeded_random": "SeededRandom",
    "point_load": "PointLoad",
}
REPEATS = 5
TOL = 1e-12
NROWS = 20736
NNZ = {"island_shell": 103104, "multi_patch": 103104, "rotated_anisotropy": 144002}
FLOATS = (
    "tol", "reported_relative", "verified_true_relative", "compensated_true_relative",
    "setup_ms", "solve_ms", "sample_total_ms",
)
INTS = (
    "repeat", "rows", "nnz", "levels", "coarse_factor_bytes", "abtm_words",
    "symbolic_nnz", "exact_zero_candidates", "scratch_bytes",
    "iterations", "max_iterations", "true_checks", "restarts",
    "rejected_recursive_convergences", "precond_calls",
)
REQUIRED = set(FLOATS + INTS) | {
    "case", "policy", "aggregation", "basis", "reuse_mode", "rhs_mode",
    "rhs_hash", "level_dims", "level_nnz", "status", "relative_x_error",
    "error_response_relative", "error_rayleigh", "residual_amplification",
}
CSV_FIELDS = (
    "case", "arm", "policy", "reuse_mode", "rhs_mode", "repeat", "rhs_hash",
    "status", "iterations", "tol", "reported_relative", "verified_true_relative",
    "compensated_true_relative", "relative_x_error", "error_rayleigh",
    "residual_amplification", "error_response_relative", "levels",
    "level_dims", "level_nnz", "coarse_factor_bytes", "symbolic_nnz",
    "exact_zero_candidates", "abtm_words", "scratch_bytes",
    "precond_calls", "true_checks", "restarts",
    "setup_ms", "solve_ms", "sample_total_ms", "elapsed_s", "peak_rss_kib",
)


def parse_pipe(line: str, marker: str) -> dict[str, str]:
    if not line.startswith(marker + "|"):
        raise ValueError("wrong record marker")
    fields: dict[str, str] = {}
    for item in line.strip().split("|")[1:]:
        if item.count("=") != 1:
            raise ValueError(f"invalid marker field: {item}")
        key, value = item.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError(f"empty or duplicate marker field: {key}")
        fields[key] = value
    return fields


def float_value(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}")
    return value


def read_time(path: Path) -> tuple[float, int]:
    pairs: dict[str, str] = {}
    # GNU time's -f format writes all three fields on one whitespace-delimited
    # line; legacy synthetic fixtures may put each field on its own line.
    for token in path.read_text(encoding="utf-8").split():
        if token.count("=") != 1:
            raise ValueError(f"bad GNU time token: {token}")
        key, value = token.split("=", 1)
        if not key or not value or key in pairs:
            raise ValueError("empty or duplicate GNU time value")
        pairs[key] = value
    if set(pairs) != {"elapsed_s", "peak_rss_kib", "exit_status"}:
        raise ValueError(f"bad GNU time schema: {path}")
    elapsed = float(pairs["elapsed_s"])
    rss = int(pairs["peak_rss_kib"])
    if not math.isfinite(elapsed) or elapsed < 0 or rss <= 0 or pairs["exit_status"] != "0":
        raise ValueError(f"invalid GNU time: {path}")
    return elapsed, rss


def parse_sample(
    row: dict[str, str], case: str, arm: str, policy: str, reuse: str,
) -> dict[str, str]:
    if set(row) != REQUIRED:
        raise ValueError(f"A10B sample schema mismatch: missing={REQUIRED-set(row)} extra={set(row)-REQUIRED}")
    aggregation, basis = ARMS[arm]
    for key, expected in (
        ("case", case), ("aggregation", aggregation), ("basis", basis),
        ("policy", POLICIES[policy]), ("reuse_mode", REUSE[reuse]),
    ):
        if row[key] != expected:
            raise ValueError(f"A10B mismatched {key}: {row[key]} vs {expected}")
    if row["rhs_mode"] not in RHS_NATIVE.values():
        raise ValueError("unknown RHS")
    if not re.fullmatch(r"[a-f0-9]{16}", row["rhs_hash"]):
        raise ValueError("invalid RHS fingerprint")
    for key in FLOATS:
        if float_value(row, key) < 0:
            raise ValueError(f"negative {key}")
    for key in INTS:
        row[key] = str(int(row[key]))  # canonical integer representation
        if int(row[key]) < 0:
            raise ValueError(f"negative {key}")
    if int(row["repeat"]) >= REPEATS or int(row["rows"]) != NROWS or int(row["nnz"]) != NNZ[case]:
        raise ValueError("invalid repetition or matrix dimensions")
    if float_value(row, "tol") != TOL or int(row["max_iterations"]) != 1200:
        raise ValueError("trial tolerance policy changed")
    if int(row["iterations"]) > 1200 or int(row["true_checks"]) < 1:
        raise ValueError("invalid PCG work counters")
    if int(row["restarts"]) > int(row["true_checks"]) or int(row["rejected_recursive_convergences"]) < int(row["restarts"]):
        raise ValueError("inconsistent certification counters")
    status = row["status"]
    verified = float_value(row, "verified_true_relative")
    if status == "Converged" and verified > TOL:
        raise ValueError("unverified convergence")
    if status == "MaxIterations" and int(row["iterations"]) != 1200:
        raise ValueError("early MaxIterations")
    if status not in ("Converged", "MaxIterations", "Breakdown"):
        raise ValueError("invalid solver status")
    if abs(float_value(row, "setup_ms") + float_value(row, "solve_ms") - float_value(row, "sample_total_ms")) > 3e-6:
        raise ValueError("invalid measured timing accounting")
    dims = tuple(int(x) for x in row["level_dims"].split(","))
    nnzs = tuple(int(x) for x in row["level_nnz"].split(","))
    if len(dims) != int(row["levels"]) or dims[0] != NROWS:
        raise ValueError("invalid hierarchy")
    if len(nnzs) != len(dims) or nnzs[0] != NNZ[case] or any(value < 0 for value in nnzs):
        raise ValueError("invalid hierarchy nonzero accounting")
    if any(a <= b for a, b in zip(dims, dims[1:])):
        raise ValueError("AMG hierarchy dimension not strictly decreasing")
    if policy == "multilevel":
        if (int(row["levels"]) < 3 or int(row["abtm_words"]) <= 0
            or int(row["symbolic_nnz"]) <= 0 or int(row["scratch_bytes"]) <= 0
            or int(row["precond_calls"]) < int(row["iterations"])):
            raise ValueError("invalid multilevel ABTM/cycle work")
        if int(row["exact_zero_candidates"]) > int(row["symbolic_nnz"]):
            raise ValueError("invalid Galerkin symbolic zero accounting")
    elif (int(row["levels"]) != 2 or int(row["abtm_words"]) != 0
          or int(row["precond_calls"]) != 0 or int(row["symbolic_nnz"]) != 0
          or int(row["exact_zero_candidates"]) != 0 or int(row["scratch_bytes"]) != 0
          or nnzs[1] != 0):
        raise ValueError("invalid two-level control")
    if int(row["coarse_factor_bytes"]) <= 0:
        raise ValueError("missing factor accounting")
    if row["rhs_mode"] == "PointLoad":
        if any(row[field] != "n/a" for field in ("relative_x_error", "error_response_relative", "error_rayleigh", "residual_amplification")):
            raise ValueError("point-load manufactured accuracy must be n/a")
    else:
        for field in ("relative_x_error", "error_response_relative", "error_rayleigh", "residual_amplification"):
            if float_value(row, field) < 0:
                raise ValueError(f"invalid forward-accuracy {field}")
    return row


def validate(a9: Path, a10a: Path, trial: Path, *, write_csv: bool = True) -> list[dict[str, str]]:
    for name in ("HEAD.txt", "matrix_sha256.txt"):
        if not (a9 / name).is_file() or not (a10a / name).is_file() or not (trial / name).is_file():
            raise ValueError(f"missing required provenance {name}")
        content = [(directory / name).read_bytes() for directory in (a9, a10a, trial)]
        if content[0] != content[1] or content[0] != content[2]:
            raise ValueError(f"frozen provenance mismatch: {name}")
    frozen_csv = a10a / "accuracy_tolerance_comparison.csv"
    if not frozen_csv.is_file():
        raise ValueError("frozen A10A 54-case summary missing")
    with frozen_csv.open(newline="", encoding="utf-8") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != 54:
            raise ValueError("frozen A10A 54-case summary incomplete")
    expected_rhs_hashes: dict[tuple[str, str], str] = {}
    for case in CASES:
        for rhs in RHS:
            frozen_log = a9 / f"{case}_graph_piecewise_{rhs}_two_level_1.log"
            if not frozen_log.is_file():
                raise ValueError(f"missing frozen A9 RHS fingerprint log: {frozen_log}")
            markers = [
                parse_pipe(line, "C1H12A9")
                for line in frozen_log.read_text(encoding="utf-8").splitlines()
                if line.startswith("C1H12A9|")
            ]
            if len(markers) != 1 or markers[0].get("rhs_mode") != RHS_NATIVE[rhs]:
                raise ValueError("invalid frozen A9 RHS record")
            fingerprint = markers[0].get("rhs_hash", "")
            if not re.fullmatch(r"[a-f0-9]{16}", fingerprint):
                raise ValueError("invalid frozen A9 RHS hash")
            expected_rhs_hashes[case, rhs] = fingerprint
    observed: list[dict[str, str]] = []
    fingerprints: dict[tuple[str, str], str] = {}
    pairs: dict[tuple[str, str, str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for case in CASES:
        for arm in ARMS:
            for policy in POLICIES:
                for reuse in REUSE:
                    stem = f"{case}_{arm}_{policy}_{reuse}_1"
                    logfile = trial / f"{stem}.log"
                    if not logfile.is_file():
                        raise ValueError(f"missing A10B run {stem}")
                    elapsed, rss = read_time(trial / f"{stem}.time")
                    session: list[dict[str, str]] = []
                    summary: list[dict[str, str]] = []
                    samples: list[dict[str, str]] = []
                    for line in logfile.read_text(encoding="utf-8").splitlines():
                        if line.startswith("C1H12A10B_SESSION|"):
                            session.append(parse_pipe(line, "C1H12A10B_SESSION"))
                        elif line.startswith("C1H12A10B_SUMMARY|"):
                            summary.append(parse_pipe(line, "C1H12A10B_SUMMARY"))
                        elif line.startswith("C1H12A10B|"):
                            samples.append(parse_pipe(line, "C1H12A10B"))
                    if len(session) != 1 or len(summary) != 1 or len(samples) != 4 * REPEATS:
                        raise ValueError(f"bad A10B session/summary/sample count: {stem}")
                    control = session[0]
                    if (control.get("case"), control.get("policy"), control.get("reuse_mode")) != (case, POLICIES[policy], REUSE[reuse]):
                        raise ValueError("session policy mismatch")
                    if control.get("repeats") != str(REPEATS) or control.get("rhs_count") != "4":
                        raise ValueError("session measurement policy mismatch")
                    for key in ("load_ms", "tol"):
                        float_value(control, key)
                    if float(control["tol"]) != TOL or control.get("max_iterations") != "1200":
                        raise ValueError("session tolerance mismatch")
                    timing_setup = timing_solve = 0.0
                    seen = set()
                    for item in samples:
                        item = parse_sample(item, case, arm, policy, reuse)
                        mode = next(key for key, value in RHS_NATIVE.items() if value == item["rhs_mode"])
                        index = int(item["repeat"])
                        key = (index, mode)
                        if key in seen:
                            raise ValueError("duplicate sample")
                        seen.add(key)
                        fingerprint_key = (case, mode)
                        if fingerprint_key in fingerprints and fingerprints[fingerprint_key] != item["rhs_hash"]:
                            raise ValueError("RHS fingerprint changed between arms or reuse schedules")
                        if item["rhs_hash"] != expected_rhs_hashes[fingerprint_key]:
                            raise ValueError("RHS fingerprint differs from frozen A9")
                        fingerprints[fingerprint_key] = item["rhs_hash"]
                        pair_key = (case, arm, policy, mode, index)
                        pairs[pair_key][reuse] = item
                        timing_setup += float(item["setup_ms"])
                        timing_solve += float(item["solve_ms"])
                        observed.append({
                            **{key: item[key] for key in CSV_FIELDS if key not in ("arm", "reuse_mode", "elapsed_s", "peak_rss_kib")},
                            "arm": arm, "reuse_mode": reuse, "elapsed_s": str(elapsed),
                            "peak_rss_kib": str(rss),
                        })
                    if len(seen) != 4 * REPEATS:
                        raise ValueError("incomplete repeats or RHS modes")
                    if reuse == "reuse" and sum(float(x["setup_ms"]) > 0 for x in samples) != 1:
                        raise ValueError("prepared setup not counted exactly once")
                    item = summary[0]
                    if item.get("samples") != str(4 * REPEATS):
                        raise ValueError("summary sample count mismatch")
                    for name, calculated in (
                        ("setup_ms_total", timing_setup),
                        ("solve_ms_total", timing_solve),
                        ("measured_total_ms", timing_setup + timing_solve),
                    ):
                        if abs(float_value(item, name) - calculated) > 5e-5:
                            raise ValueError(f"timing summary disagreement: {name}")
    if len(observed) != len(CASES) * len(ARMS) * len(POLICIES) * len(REUSE) * len(RHS) * REPEATS:
        raise ValueError("A10B sample total mismatch")
    for key, by_reuse in pairs.items():
        if set(by_reuse) != set(REUSE):
            raise ValueError("missing fresh/reuse pair")
        fresh, reused = by_reuse["fresh"], by_reuse["reuse"]
        for name in ("rhs_hash", "status", "iterations", "level_dims", "level_nnz", "coarse_factor_bytes", "symbolic_nnz", "exact_zero_candidates", "abtm_words", "scratch_bytes"):
            if fresh[name] != reused[name]:
                raise ValueError(f"fresh/reuse numerical/hierarchy divergence at {key}: {name}")
        if abs(float(fresh["verified_true_relative"]) - float(reused["verified_true_relative"])) > 1e-13:
            raise ValueError("fresh/reuse independently verified residual disagreement")
        if fresh["relative_x_error"] != "n/a" and abs(float(fresh["relative_x_error"]) - float(reused["relative_x_error"])) > 1e-7:
            raise ValueError("fresh/reuse forward error mismatch")
    if write_csv:
        with (trial / "prepared_reuse_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(observed)
    print(f"C1-H12-A10B {len(observed)}/{len(observed)} original-CSR, RHS reuse, ABTM hierarchy and provenance gates PASS")
    for reuse in REUSE:
        items = [row for row in observed if row["reuse_mode"] == reuse]
        print(
            f"  {reuse}: converged={sum(row['status'] == 'Converged' for row in items)}/{len(items)}; "
            f"setup_ms={sum(float(row['setup_ms']) for row in items):.3f}; "
            f"solve_ms={sum(float(row['solve_ms']) for row in items):.3f}; "
            f"total_ms={sum(float(row['sample_total_ms']) for row in items):.3f}"
        )
    print(f"CSV: {trial / 'prepared_reuse_comparison.csv'}")
    return observed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: C1H12_A10B_prepared_gate.py A9_RUN A10A_RUN A10B_RUN")
    try:
        validate(*(Path(item) for item in sys.argv[1:]))
    except (OSError, ValueError, StopIteration) as exc:
        raise SystemExit(f"C1-H12-A10B evidence FAIL: {exc}") from exc
