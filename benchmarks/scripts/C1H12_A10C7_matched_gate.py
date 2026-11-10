#!/usr/bin/env python3
"""Fail-closed matched A10C4/A10C6 numerical and timing provenance gate.

Timing is descriptive evidence, NEVER a success/performance threshold.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ("graph_piecewise", "strong_graph_piecewise", "strong_graph_smoothed")
POLICIES = {"two_level": "TwoLevel", "multilevel": "Multilevel"}
RHS = ("Ones", "PointLoad")
TOLS = (1e-10, 1e-11, 1e-12)
COOLDOWNS = (25, 50)
SOLVERS = ("reliable", "shadow")
REPETITIONS = 6
SESSION_TAG = "C1H12A10C7_SESSION"
SAMPLE_TAG = "C1H12A10C7_SAMPLE"
SESSION_FIELDS = frozenset(
    "case arm policy rows nnz rhs_count tol_count cooldown_count warmups "
    "measured_repetitions max_iterations stagnation_window progress_factor "
    "threshold_spacing setup_ns level_dims level_nnz abtm_words".split()
)
SAMPLE_FIELDS = frozenset(
    "case arm policy rhs_mode rhs_hash tol cooldown solver rep order status "
    "iterations recursive_relative true_relative compensated_relative precond_calls "
    "true_checks replacements false_events best_true_relative first_stagnation_candidate "
    "solve_ns certificate_ns replacement_ns".split()
)
SAMPLE_COLUMNS = tuple(
    "case arm policy rhs_mode rhs_hash tol cooldown solver rep order status "
    "iterations recursive_relative true_relative compensated_relative precond_calls "
    "true_checks replacements false_events best_true_relative first_stagnation_candidate "
    "solve_ns certificate_ns replacement_ns process_elapsed_s process_peak_rss_kib".split()
)
SUMMARY_COLUMNS = tuple(
    "case arm policy rhs_mode rhs_hash tol cooldown status_c4 status_c6 "
    "iterations_c4 iterations_c6 replacements_c4 replacements_c6 "
    "median_c4_solve_ns median_c6_solve_ns median_c4_certificate_ns "
    "median_c6_certificate_ns median_c4_replacement_ns median_c6_replacement_ns "
    "paired_median_speed_ratio_c4_over_c6 setup_ns performance_status".split()
)
C4_COLUMNS = tuple(
    "case arm policy rhs_mode rhs_hash tol replacement_period status iterations "
    "recursive_relative true_relative compensated_relative precond_calls true_checks "
    "replacements false_events best_true_relative relative_x_error max_gap_ratio "
    "process_elapsed_s process_peak_rss_kib".split()
)
C6_COLUMNS = tuple(
    "case arm policy rhs_mode rhs_hash tol cooldown status iterations recursive_relative "
    "true_relative compensated_relative precond_calls true_checks replacements "
    "false_events cooldown_suppressed best_true_relative last_progress_iteration "
    "relative_x_error first_stagnation_candidate converged_after_candidate "
    "a10c4_status a10c4_iterations a10c4_replacements a10c4_best_true_relative "
    "a10c5_status a10c5_iterations a10c5_replacements a10c5_best_true_relative "
    "process_elapsed_s process_peak_rss_kib".split()
)


def number(value: str, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {name}")
    return result


def integer(value: str, name: str, lower: int = 0) -> int:
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        raise ValueError(f"noncanonical integer {name}")
    result = int(value)
    if result < lower:
        raise ValueError(f"invalid {name}")
    return result


def close_numeric(actual: str, expected: str, name: str) -> None:
    if not math.isclose(number(actual, name), number(expected, f"frozen {name}"),
                        rel_tol=4e-12, abs_tol=2e-15):
        raise ValueError(f"numerical regression against frozen evidence: {name}")


def read_csv(path: Path, columns: tuple[str, ...], count: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"frozen CSV schema changed: {path.name}")
        rows = list(reader)
    if len(rows) != count or any(None in row for row in rows):
        raise ValueError(f"incomplete or malformed frozen reference: {path.name}")
    return rows


def identity(row: dict[str, str], period_field: str) -> tuple[str, str, str, float, int]:
    if row["case"] != "multi_patch" or row["arm"] not in ARMS \
            or row["policy"] not in POLICIES.values() or row["rhs_mode"] not in RHS \
            or not re.fullmatch(r"[0-9a-f]{16}", row["rhs_hash"]):
        raise ValueError("invalid FEM/arm/policy/RHS identity")
    tol = number(row["tol"], "tol")
    period = integer(row[period_field], period_field)
    if tol not in TOLS or period not in (0, 25, 50):
        raise ValueError("changed tolerance/replacement controls")
    return row["arm"], row["policy"], row["rhs_mode"], tol, period


def reference_maps(
    c4rows: list[dict[str, str]], c6rows: list[dict[str, str]],
) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    c4: dict[tuple, dict] = {}
    c6: dict[tuple, dict] = {}
    for row in c4rows:
        key = identity(row, "replacement_period")
        if key in c4:
            raise ValueError("duplicate frozen A10C4 result")
        c4[key] = row
    for row in c6rows:
        key = identity(row, "cooldown")
        if key in c6 or key[4] == 0:
            raise ValueError("duplicate or zero-period frozen A10C6 result")
        c6[key] = row
    expected4 = {(a, p, rhs, tol, cd) for a in ARMS for p in POLICIES.values()
                 for rhs in RHS for tol in TOLS for cd in (0, *COOLDOWNS)}
    expected6 = {key for key in expected4 if key[4] in COOLDOWNS}
    if set(c4) != expected4 or set(c6) != expected6:
        raise ValueError("incomplete frozen A10C4/A10C6 control grid")
    for key, reference in c6.items():
        previous = c4[key]
        for newer, older in (("a10c4_status", "status"),
                             ("a10c4_iterations", "iterations"),
                             ("a10c4_replacements", "replacements")):
            if reference[newer] != previous[older]:
                raise ValueError("frozen A10C6-to-A10C4 control disagreement")
        if reference["rhs_hash"] != previous["rhs_hash"]:
            raise ValueError("frozen RHS hash disagreement")
        close_numeric(reference["a10c4_best_true_relative"],
                      previous["best_true_relative"], "frozen best residual")
    return c4, c6


def parse_pipe(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix + "|"):
        raise ValueError(f"wrong {prefix} record prefix")
    row: dict[str, str] = {}
    for field in line[len(prefix) + 1:].split("|"):
        if "=" not in field:
            raise ValueError("malformed log field")
        key, val = field.split("=", 1)
        if not key or val == "" or key in row:
            raise ValueError("missing, empty or duplicated log field")
        row[key] = val
    return row


def parse_process(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    sessions: list[dict[str, str]] = []
    samples: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(SESSION_TAG + "|"):
            row = parse_pipe(line, SESSION_TAG)
            if set(row) != SESSION_FIELDS:
                raise ValueError("matched session schema mismatch")
            sessions.append(row)
        elif line.startswith(SAMPLE_TAG + "|"):
            row = parse_pipe(line, SAMPLE_TAG)
            if set(row) != SAMPLE_FIELDS:
                raise ValueError("matched sample schema mismatch")
            samples.append(row)
        elif line.startswith("C1H12A10C4_CHECK|") or line.startswith("C1H12A10C6_CHECK|"):
            raise ValueError("console I/O was enabled during timed PCG")
    if len(sessions) != 1 or len(samples) != 144:
        raise ValueError("expected one session and 144 measured records per process")
    return sessions[0], samples


def validate_session(row: dict[str, str], arm: str, policy: str) -> int:
    for name, value in {
        "case": "multi_patch", "arm": arm, "policy": policy,
        "rows": "20736", "nnz": "103104", "rhs_count": "2", "tol_count": "3",
        "cooldown_count": "2", "warmups": "1", "measured_repetitions": "6",
        "max_iterations": "1200", "stagnation_window": "200",
        "threshold_spacing": "5",
    }.items():
        if row[name] != value:
            raise ValueError(f"altered matched experiment parameter: {name}")
    if number(row["progress_factor"], "progress_factor") != 0.9:
        raise ValueError("altered stagnation progress factor")
    dims = [integer(n, "level_dims", 1) for n in row["level_dims"].split(",")]
    nnz = [integer(n, "level_nnz") for n in row["level_nnz"].split(",")]
    if len(dims) < 2 or len(dims) != len(nnz) or dims[0] != 20736 \
            or nnz[0] != 103104 or any(a <= b for a, b in zip(dims, dims[1:])):
        raise ValueError("invalid original ABTM multilevel hierarchy provenance")
    if policy == "Multilevel" and integer(row["abtm_words"], "abtm_words", 1) == 0:
        raise ValueError("missing ABTM word hierarchy")
    return integer(row["setup_ns"], "setup_ns", 1)


def validate_grid(
    samples: list[dict[str, str]], c4: dict[tuple, dict], c6: dict[tuple, dict],
    setup: dict[tuple[str, str], int],
) -> list[dict[str, str]]:
    buckets: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        key = identity(row, "cooldown")
        if key not in c6:
            raise ValueError("unmatched sample outside frozen control grid")
        solver = row["solver"]
        repetition = integer(row["rep"], "rep", 1)
        if solver not in SOLVERS or repetition > REPETITIONS \
                or row["order"] != ("C6C4" if repetition % 2 else "C4C6"):
            raise ValueError("unbalanced or altered AB/BA measurement schedule")
        expected = c4[key] if solver == "reliable" else c6[key]
        if row["rhs_hash"] != expected["rhs_hash"]:
            raise ValueError("timed RHS differs from frozen original CSR")
        for field in ("status", "iterations", "precond_calls", "true_checks",
                      "replacements", "false_events"):
            if row[field] != expected[field]:
                raise ValueError(f"timed numerical trajectory differs from frozen {solver}: {field}")
        expected_candidate = "0" if solver == "reliable" else expected["first_stagnation_candidate"]
        if row["first_stagnation_candidate"] != expected_candidate:
            raise ValueError("stagnation candidate differs from frozen A10C6")
        for field in ("recursive_relative", "true_relative", "compensated_relative",
                      "best_true_relative"):
            close_numeric(row[field], expected[field], field)
        status = row["status"]
        iterations = integer(row["iterations"], "iterations")
        if status not in ("Converged", "MaxIterations", "Breakdown") or iterations > 1200:
            raise ValueError("unsupported PCG status/iteration count")
        if status == "MaxIterations" and iterations != 1200:
            raise ValueError("premature MaxIterations")
        if status == "Converged" and any(number(row[name], name) > key[3] for name in
                                         ("recursive_relative", "true_relative", "compensated_relative")):
            raise ValueError("false original-CSR/Kahan convergence")
        solve = integer(row["solve_ns"], "solve_ns", 1)
        certificate = integer(row["certificate_ns"], "certificate_ns", 1)
        replacement = integer(row["replacement_ns"], "replacement_ns")
        if certificate + replacement > solve or (integer(row["replacements"], "replacements") > 0 and replacement == 0) \
                or (integer(row["replacements"], "replacements") == 0 and replacement > 0):
            raise ValueError("invalid timing decomposition or replacement count")
        buckets[(*key, solver, repetition)].append(row)
    expected_keys = {(*key, solver, repetition) for key in c6
                     for solver in SOLVERS for repetition in range(1, REPETITIONS + 1)}
    if set(buckets) != expected_keys or any(len(group) != 1 for group in buckets.values()):
        raise ValueError("missing/duplicate 864 paired measurements")
    if len(setup) != len(ARMS) * len(POLICIES):
        raise ValueError("missing matched hierarchy setup timings")
    summaries: list[dict[str, str]] = []
    for key in sorted(c6):
        a, p, rhs, tol, cd = key
        a_rows = [buckets[(*key, "reliable", i)][0] for i in range(1, REPETITIONS + 1)]
        b_rows = [buckets[(*key, "shadow", i)][0] for i in range(1, REPETITIONS + 1)]
        def med(rows: list[dict[str, str]], name: str) -> float:
            return statistics.median(int(row[name]) for row in rows)
        ratios = [int(row_a["solve_ns"]) / int(row_b["solve_ns"])
                  for row_a, row_b in zip(a_rows, b_rows)]
        summaries.append({
            "case": "multi_patch", "arm": a, "policy": p, "rhs_mode": rhs,
            "rhs_hash": c6[key]["rhs_hash"], "tol": f"{tol:.0e}", "cooldown": str(cd),
            "status_c4": c4[key]["status"], "status_c6": c6[key]["status"],
            "iterations_c4": c4[key]["iterations"], "iterations_c6": c6[key]["iterations"],
            "replacements_c4": c4[key]["replacements"],
            "replacements_c6": c6[key]["replacements"],
            "median_c4_solve_ns": f"{med(a_rows, 'solve_ns'):.1f}",
            "median_c6_solve_ns": f"{med(b_rows, 'solve_ns'):.1f}",
            "median_c4_certificate_ns": f"{med(a_rows, 'certificate_ns'):.1f}",
            "median_c6_certificate_ns": f"{med(b_rows, 'certificate_ns'):.1f}",
            "median_c4_replacement_ns": f"{med(a_rows, 'replacement_ns'):.1f}",
            "median_c6_replacement_ns": f"{med(b_rows, 'replacement_ns'):.1f}",
            "paired_median_speed_ratio_c4_over_c6": f"{statistics.median(ratios):.9f}",
            "setup_ns": str(setup[a, p]), "performance_status": "OBSERVATIONAL_ONLY",
        })
    return summaries


def check_frozen_hashes(evidence: Path, source: Path, label: str, names: tuple[str, ...]) -> None:
    lines = (evidence / f"FROZEN_{label}_SHA256.txt").read_text(encoding="utf-8").splitlines()
    expected = {str(source / name): hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in names}
    found: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in found:
            raise ValueError(f"bad or duplicate frozen {label} fingerprint")
        found[match[2]] = match[1]
    if found != expected:
        raise ValueError(f"frozen {label} evidence changed")


def read_time(path: Path) -> tuple[str, str]:
    data = path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"elapsed_s=([0-9]+(?:\.[0-9]+)?) peak_rss_kib=([0-9]+) exit_status=0", data)
    if not match or number(match[1], "elapsed_s") <= 0:
        raise ValueError("nonzero or malformed GNU time")
    return match[1], match[2]


def validate(
    a10c6: Path, a10c4: Path, a10b: Path, evidence: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    for left, right in (
        (evidence / "HEAD.txt", a10c6 / "HEAD.txt"),
        (evidence / "FROZEN_A10C6_HEAD.txt", a10c6 / "HEAD.txt"),
        (evidence / "FROZEN_A10C4_HEAD.txt", a10c4 / "HEAD.txt"),
        (evidence / "FROZEN_A10B_HEAD.txt", a10b / "HEAD.txt"),
        (evidence / "matrix_sha256.txt", a10b / "matrix_sha256.txt"),
        (a10c6 / "matrix_sha256.txt", a10c4 / "matrix_sha256.txt"),
        (a10c4 / "matrix_sha256.txt", a10b / "matrix_sha256.txt"),
    ):
        if left.read_bytes() != right.read_bytes():
            raise ValueError("HEAD/matrix provenance mismatch")
    check_frozen_hashes(evidence, a10c4, "A10C4",
                        ("reliable_update_comparison.csv", "reliable_residual_trace.csv"))
    check_frozen_hashes(evidence, a10c6, "A10C6",
                        ("shadow_comparison.csv", "shadow_residual_trace.csv"))
    c4, c6 = reference_maps(
        read_csv(a10c4 / "reliable_update_comparison.csv", C4_COLUMNS, 108),
        read_csv(a10c6 / "shadow_comparison.csv", C6_COLUMNS, 72),
    )
    samples: list[dict[str, str]] = []
    setup: dict[tuple[str, str], int] = {}
    for arm in ARMS:
        for policy_short, policy in POLICIES.items():
            stem = f"multi_patch_{arm}_{policy_short}_1"
            session, records = parse_process(evidence / f"{stem}.log")
            setup[arm, policy] = validate_session(session, arm, policy)
            chronological = []
            for rhs in RHS:
                for tol in TOLS:
                    for cooldown in COOLDOWNS:
                        for repetition in range(1, REPETITIONS + 1):
                            order = ("shadow", "reliable") if repetition % 2 else ("reliable", "shadow")
                            chronological.extend((rhs, tol, cooldown, repetition, solver)
                                                 for solver in order)
            actual = [(r["rhs_mode"], number(r["tol"], "tol"),
                       integer(r["cooldown"], "cooldown"), integer(r["rep"], "rep"),
                       r["solver"]) for r in records]
            if actual != chronological:
                raise ValueError("nonalternating or reordered chronological AB/BA samples")
            elapsed, peak = read_time(evidence / f"{stem}.time")
            samples.extend({**row, "process_elapsed_s": elapsed,
                            "process_peak_rss_kib": peak} for row in records)
    summaries = validate_grid(samples, c4, c6, setup)
    if len(samples) != 864 or len(summaries) != 72:
        raise ValueError("incomplete matched measurement collection")
    return samples, summaries


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: C1H12_A10C7_matched_gate.py FROZEN_A10C6 FROZEN_A10C4 FROZEN_A10B OUTPUT_DIR")
    try:
        target = Path(sys.argv[4])
        records, summary = validate(Path(sys.argv[1]), Path(sys.argv[2]),
                                    Path(sys.argv[3]), target)
        write_csv(target / "matched_timing_samples.csv", SAMPLE_COLUMNS, records)
        write_csv(target / "matched_timing_summary.csv", SUMMARY_COLUMNS, summary)
        for cooldown in COOLDOWNS:
            group = [row for row in summary if int(row["cooldown"]) == cooldown]
            ratios = [float(row["paired_median_speed_ratio_c4_over_c6"]) for row in group]
            print(f"  cooldown={cooldown}: 36 matched conditions, median C4/C6 solve-time "
                  f"ratio={statistics.median(ratios):.4f} (exploratory; no speedup claim)")
        print("C1-H12-A10C7 864/864 matched samples, 72/72 frozen numerical controls PASS")
        print("Scope: diagnostic code, shared setup, interleaved AB/BA, medians only; no production speedup guarantee")
    except (OSError, KeyError, ValueError, OverflowError) as error:
        raise SystemExit(f"C1-H12-A10C7 evidence FAIL: {error}") from error
