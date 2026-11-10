#!/usr/bin/env python3
"""Fail-closed A10C6 shadow-stagnation, original-CSR, A10C5-prefix gate."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import ARMS, POLICIES, parse_pipe, read_time
from benchmarks.scripts.C1H12_A10C3_trace_gate import finite, integer
from benchmarks.scripts.C1H12_A10C5_adaptive_gate import (
    CHECK_COLUMNS as C5_CHECK_COLUMNS,
    FINAL_COLUMNS as C5_FINAL_COLUMNS,
    validate as validate_c5,
)

CASE = "multi_patch"
RHS = ("Ones", "PointLoad")
TOLS = (1e-10, 1e-11, 1e-12)
COOLDOWNS = (25, 50)
MAX_ITER = 1200
WINDOW = 200
FACTOR = 0.9
SPACING = 5
SESSION = "C1H12A10C6_SESSION"
CHECK = "C1H12A10C6_CHECK"
FINAL = "C1H12A10C6_FINAL"
IDENTITY = {"case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown"}
CHECK_KEYS = IDENTITY | {
    "iteration", "recursive_relative", "true_relative", "compensated_relative", "gap_ratio",
    "replacement_applied", "cooldown_suppressed", "last_progress_iteration", "precond_calls",
    "stagnation_candidate",
}
FINAL_KEYS = IDENTITY | {
    "status", "iterations", "recursive_relative", "true_relative", "compensated_relative",
    "precond_calls", "true_checks", "replacements", "false_events", "cooldown_suppressed",
    "best_true_relative", "last_progress_iteration", "relative_x_error",
    "first_stagnation_candidate", "converged_after_candidate",
}
SESSION_KEYS = {
    "case", "arm", "policy", "rows", "nnz", "rhs_count", "tol_count", "cooldown_count",
    "stagnation_window", "progress_factor", "threshold_spacing", "max_iterations",
    "stop_policy",
    "level_dims", "level_nnz", "abtm_words",
}
CHECK_COLUMNS = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown",
    "iteration", "recursive_relative", "true_relative", "compensated_relative", "gap_ratio",
    "replacement_applied", "cooldown_suppressed", "last_progress_iteration", "precond_calls",
    "stagnation_candidate",
)
FINAL_COLUMNS = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown",
    "status", "iterations", "recursive_relative", "true_relative", "compensated_relative",
    "precond_calls", "true_checks", "replacements", "false_events", "cooldown_suppressed",
    "best_true_relative", "last_progress_iteration", "relative_x_error",
    "first_stagnation_candidate", "converged_after_candidate",
    "a10c4_status", "a10c4_iterations", "a10c4_replacements", "a10c4_best_true_relative",
    "a10c5_status", "a10c5_iterations", "a10c5_replacements", "a10c5_best_true_relative",
    "process_elapsed_s", "process_peak_rss_kib",
)


def parse_records(path: Path) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    sessions, checks, finals = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(SESSION + "|"):
            row = parse_pipe(line, SESSION)
            if set(row) != SESSION_KEYS:
                raise ValueError("A10C6 session schema mismatch")
            sessions.append(row)
        elif line.startswith(CHECK + "|"):
            row = parse_pipe(line, CHECK)
            if set(row) != CHECK_KEYS:
                raise ValueError("A10C6 original-CSR checkpoint schema mismatch")
            checks.append(row)
        elif line.startswith(FINAL + "|"):
            row = parse_pipe(line, FINAL)
            if set(row) != FINAL_KEYS:
                raise ValueError("A10C6 final record schema mismatch")
            finals.append(row)
    if len(sessions) != 1 or len(finals) != 12 or not checks:
        raise ValueError("A10C6 missing session, twelve outcomes or checkpoints")
    return sessions[0], checks, finals


def check_frozen_hashes(evidence: Path, frozen: Path) -> None:
    lines = (evidence / "FROZEN_A10C5_SHA256.txt").read_text(encoding="utf-8").splitlines()
    expected = {}
    for name in ("adaptive_update_comparison.csv", "adaptive_residual_trace.csv"):
        path = frozen / name
        expected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    found = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) in found:
            raise ValueError("A10C6 malformed frozen A10C5 fingerprints")
        found[match.group(2)] = match.group(1)
    if found != expected:
        raise ValueError("A10C6 frozen A10C5 evidence hash changed")


def read_c5_checked_records(
    a10c5: Path, a10c4: Path, a10b: Path,
) -> tuple[dict[tuple[str, str, str, float, int], dict[str, str]],
           dict[tuple[str, str, str, float, int], list[dict[str, str]]]]:
    """Validate original logs without mutating any frozen A10C5 evidence."""
    prior_checks, prior_results = validate_c5(a10c4, a10b, a10c5)
    for name, columns, expected in (
        ("adaptive_update_comparison.csv", C5_FINAL_COLUMNS, prior_results),
        ("adaptive_residual_trace.csv", C5_CHECK_COLUMNS, prior_checks),
    ):
        with (a10c5 / name).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != columns or list(reader) != expected:
                raise ValueError(f"A10C6 frozen A10C5 summary disagrees with raw logs: {name}")
    result_map = {}
    check_map = defaultdict(list)
    for row in prior_results:
        key = (row["arm"], row["policy"], row["rhs_mode"],
               finite(row["tol"], "frozen A10C5 tol"),
               integer(row["cooldown"], "frozen A10C5 cooldown"))
        if key in result_map:
            raise ValueError("A10C6 duplicate frozen A10C5 identity")
        result_map[key] = row
    for row in prior_checks:
        key = (row["arm"], row["policy"], row["rhs_mode"],
               finite(row["tol"], "frozen A10C5 tol"),
               integer(row["cooldown"], "frozen A10C5 cooldown"))
        check_map[key].append(row)
    if len(result_map) != 72 or len(check_map) != 72:
        raise ValueError("A10C6 incomplete frozen A10C5 reference")
    return result_map, check_map


def check_session(row: dict[str, str], arm: str, policy: str) -> None:
    if (row["case"] != CASE or row["arm"] != arm or row["policy"] != policy
            or integer(row["rows"], "rows") != 20736
            or integer(row["nnz"], "nnz") != 103104
            or integer(row["rhs_count"], "rhs_count") != 2
            or integer(row["tol_count"], "tol_count") != 3
            or integer(row["cooldown_count"], "cooldown_count") != 2
            or integer(row["stagnation_window"], "window") != WINDOW
            or finite(row["progress_factor"], "progress_factor") != FACTOR
            or integer(row["threshold_spacing"], "spacing") != SPACING
            or integer(row["max_iterations"], "max_iterations") != MAX_ITER
            or row["stop_policy"] != "observe_only"):
        raise ValueError("A10C6 experiment controls changed")
    dims = [integer(part, "level_dims", 1) for part in row["level_dims"].split(",")]
    nnz = [integer(part, "level_nnz") for part in row["level_nnz"].split(",")]
    if (len(dims) < 2 or len(dims) != len(nnz) or dims[0] != 20736
            or nnz[0] != 103104 or any(x <= y for x, y in zip(dims, dims[1:]))):
        raise ValueError("A10C6 invalid AMG hierarchy provenance")
    if policy == "Multilevel" and integer(row["abtm_words"], "abtm_words", 1) == 0:
        raise ValueError("A10C6 multilevel ABTM hierarchy unavailable")


def validate_group(
    checks: list[dict[str, str]], final: dict[str, str],
    identity: tuple[str, str, str, float, int], frozen: dict[str, str],
    frozen_checks: list[dict[str, str]],
) -> None:
    arm, policy, rhs_mode, tol, cooldown = identity
    expected_identity = {"case": CASE, "arm": arm, "policy": policy,
                         "rhs_mode": rhs_mode, "rhs_hash": frozen["rhs_hash"]}
    if not checks:
        raise ValueError("A10C6 missing original-CSR checkpoints")
    for row in [*checks, final]:
        if any(row[name] != value for name, value in expected_identity.items()):
            raise ValueError("A10C6 RHS/arm identity disagrees with frozen A10C5")
        if finite(row["tol"], "tol") != tol or integer(row["cooldown"], "cooldown") != cooldown:
            raise ValueError("A10C6 tolerance/cooldown mismatch")
    iterations = integer(final["iterations"], "iterations")
    if iterations > MAX_ITER:
        raise ValueError("A10C6 exceeded iteration budget")
    steps = [integer(row["iteration"], "iteration") for row in checks]
    if steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != iterations:
        raise ValueError("A10C6 missing, duplicated, or out-of-order final checkpoint")
    mandatory = {0, iterations}
    if iterations:
        mandatory.add(1)
    mandatory.update(range(25, iterations + 1, 25))
    if not mandatory.issubset(steps):
        raise ValueError("A10C6 mandatory independent CSR checkpoint missing")
    last_logged, last_replaced, last_progress = 0, 0, 0
    best_certificate, best_true = 1.0, 1.0
    replacements, false_events, suppressed = 0, 0, 0
    first_candidate = 0
    for row, step in zip(checks, steps):
        recursive = finite(row["recursive_relative"], "recursive_relative")
        verified = finite(row["true_relative"], "true_relative")
        compensated = finite(row["compensated_relative"], "compensated_relative")
        ratio = finite(row["gap_ratio"], "gap_ratio")
        if not math.isclose(ratio, verified / max(recursive, sys.float_info.min), rel_tol=3e-12, abs_tol=1e-12):
            raise ValueError("A10C6 gap ratio inconsistent with original CSR")
        if integer(row["precond_calls"], "precond_calls") != step:
            raise ValueError("A10C6 preconditioner calls disagree with iteration")
        if step == 0 and any(abs(v - 1.0) > 2e-14 for v in (recursive, verified, compensated)):
            raise ValueError("A10C6 zero-start certificates must equal one")
        if step not in (0, 1, iterations) and step % 25 != 0:
            if recursive > tol or step - last_logged < SPACING:
                raise ValueError("A10C6 unscheduled extra CSR checkpoint")
        if step > 0:
            joint = max(verified, compensated)
            if joint < best_certificate * FACTOR:
                best_certificate = joint
                last_progress = step
        else:
            joint = 1.0
        if integer(row["last_progress_iteration"], "last_progress_iteration") != last_progress:
            raise ValueError("A10C6 reported last progress conflicts with original CSR")
        false = recursive <= tol and joint > tol
        if false:
            false_events += 1
        certified = recursive <= tol and joint <= tol
        stalled = (step >= WINDOW and step - last_progress >= WINDOW and replacements > 0
                   and joint > tol)
        candidate = int(first_candidate == 0 and stalled)
        if candidate:
            first_candidate = step
        if integer(row["stagnation_candidate"], "stagnation_candidate") != candidate:
            raise ValueError("A10C6 missing or forged first stagnation candidate")
        wants = step > 0 and (step % 25 == 0 or false)
        eligible = step - last_replaced >= cooldown
        # On the first candidate checkpoint match A10C5 exactly. Afterwards,
        # continue normal cooldown-controlled restarts without early stopping.
        can_change = step < MAX_ITER and not certified and not candidate
        expected_replace = int(wants and eligible and can_change)
        expected_suppress = int(wants and not eligible and can_change)
        actual_replace = integer(row["replacement_applied"], "replacement_applied")
        actual_suppress = integer(row["cooldown_suppressed"], "cooldown_suppressed")
        if (final["status"] == "Breakdown" and step == iterations
                and (actual_replace, actual_suppress) == (0, 0)):
            # A final audit after numerical breakdown cannot begin another restart.
            expected_replace, expected_suppress = 0, 0
        if actual_replace != expected_replace:
            raise ValueError("A10C6 violated cooldown or replacement policy")
        if actual_suppress != expected_suppress:
            raise ValueError("A10C6 cooldown suppression count disagrees")
        if expected_replace:
            replacements += 1
            last_replaced = step
        suppressed += expected_suppress
        best_true = min(best_true, verified)
        last_logged = step
    if (integer(final["replacements"], "replacements") != replacements
            or integer(final["cooldown_suppressed"], "cooldown_suppressed") != suppressed
            or integer(final["false_events"], "false_events") != false_events
            or integer(final["true_checks"], "true_checks") != len(checks)
            or integer(final["last_progress_iteration"], "last_progress_iteration") != last_progress):
        raise ValueError("A10C6 work, progress, or event counters disagree")
    if integer(final["first_stagnation_candidate"], "first_stagnation_candidate") != first_candidate:
        raise ValueError("A10C6 first candidate summary disagrees with original CSR")
    if not math.isclose(finite(final["best_true_relative"], "best_true"), best_true,
                        rel_tol=3e-12, abs_tol=1e-14):
        raise ValueError("A10C6 best true residual inconsistent")
    if not (iterations <= integer(final["precond_calls"], "precond_calls") <= iterations + 1):
        raise ValueError("A10C6 final preconditioner calls inconsistent")
    for name in ("recursive_relative", "true_relative", "compensated_relative"):
        if not math.isclose(finite(final[name], name), finite(checks[-1][name], name),
                            rel_tol=3e-12, abs_tol=1e-14):
            raise ValueError("A10C6 final original-CSR certificate changed")
    status = final["status"]
    rec = finite(final["recursive_relative"], "final recursive")
    true = finite(final["true_relative"], "final true")
    kahan = finite(final["compensated_relative"], "final compensated")
    if status == "Converged" and not (rec <= tol and true <= tol and kahan <= tol):
        raise ValueError("A10C6 false convergence incorrectly accepted")
    if status != "Converged" and rec <= tol and true <= tol and kahan <= tol:
        raise ValueError("A10C6 failed to report a verified convergence")
    if status == "MaxIterations" and iterations != MAX_ITER:
        raise ValueError("A10C6 premature maximum-iteration status")
    if status not in {"Converged", "MaxIterations", "Breakdown"}:
        raise ValueError("A10C6 unsupported terminal status, including early stop")
    if integer(final["converged_after_candidate"], "converged_after_candidate") != int(
        first_candidate > 0 and status == "Converged"
    ):
        raise ValueError("A10C6 recovery marker inconsistent")
    # The frozen A10C5 trace must be bytewise-equivalent in numerical fields
    # throughout its prefix. Its own terminal state is a certified reference.
    prefix = [row for row in checks if integer(row["iteration"], "step")
              <= integer(frozen["iterations"], "frozen iteration")]
    if len(prefix) != len(frozen_checks):
        raise ValueError("A10C6 changed frozen A10C5 checkpoint schedule")
    for current, old in zip(prefix, frozen_checks):
        if {name: current[name] for name in C5_CHECK_COLUMNS} != old:
            raise ValueError("A10C6 diverged from frozen A10C5 before shadow continuation")
    if frozen["status"] == "Converged":
        if status != "Converged" or iterations != integer(frozen["iterations"], "frozen convergence"):
            raise ValueError("A10C6 changed a frozen A10C5 certified convergence")
        if first_candidate:
            raise ValueError("A10C6 invented candidate before frozen A10C5 convergence")
    elif frozen["status"] == "Stagnated":
        if first_candidate != integer(frozen["iterations"], "frozen stagnation"):
            raise ValueError("A10C6 did not reproduce frozen A10C5 stagnation decision")
        if iterations <= first_candidate:
            raise ValueError("A10C6 shadow mode failed to continue beyond candidate")
    else:
        raise ValueError("A10C6 unexpected frozen A10C5 terminal status")
    if rhs_mode == "PointLoad":
        if final["relative_x_error"] != "n/a":
            raise ValueError("point load must not assert manufactured solution error")
    else:
        finite(final["relative_x_error"], "relative_x_error")


def validate(
    a10c5: Path, a10c4: Path, a10b: Path, evidence: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    for left, right in (
        (evidence / "HEAD.txt", a10c5 / "HEAD.txt"),
        (evidence / "FROZEN_A10C5_HEAD.txt", a10c5 / "HEAD.txt"),
        (evidence / "FROZEN_A10C4_HEAD.txt", a10c4 / "HEAD.txt"),
        (evidence / "FROZEN_A10B_HEAD.txt", a10b / "HEAD.txt"),
        (evidence / "matrix_sha256.txt", a10c5 / "matrix_sha256.txt"),
        (a10c5 / "matrix_sha256.txt", a10c4 / "matrix_sha256.txt"),
        (a10c4 / "matrix_sha256.txt", a10b / "matrix_sha256.txt"),
    ):
        if left.read_bytes() != right.read_bytes():
            raise ValueError("A10C6 frozen HEAD or matrix provenance changed")
    check_frozen_hashes(evidence, a10c5)
    reference, reference_checks = read_c5_checked_records(a10c5, a10c4, a10b)
    all_checks, all_finals = [], []
    for arm in ARMS:
        for policy_short, policy in POLICIES.items():
            stem = f"multi_patch_{arm}_{policy_short}_1"
            session, checks, finals = parse_records(evidence / f"{stem}.log")
            check_session(session, arm, policy)
            elapsed, peak = read_time(evidence / f"{stem}.time")
            groups = defaultdict(list)
            for row in checks:
                groups[(row["rhs_mode"], finite(row["tol"], "tol"), integer(row["cooldown"], "cooldown"))].append(row)
            results = {}
            for row in finals:
                key = (row["rhs_mode"], finite(row["tol"], "tol"), integer(row["cooldown"], "cooldown"))
                if key in results:
                    raise ValueError("A10C6 duplicate final record")
                results[key] = row
            expected = {(rhs, tol, cd) for rhs in RHS for tol in TOLS for cd in COOLDOWNS}
            if set(groups) != expected or set(results) != expected:
                raise ValueError("A10C6 incomplete 12-outcome process grid")
            for rhs, tol, cooldown in sorted(expected):
                control = reference[arm, policy, rhs, tol, cooldown]
                row = results[rhs, tol, cooldown]
                validate_group(groups[rhs, tol, cooldown], row,
                               (arm, policy, rhs, tol, cooldown), control,
                               reference_checks[arm, policy, rhs, tol, cooldown])
                all_checks.extend(groups[rhs, tol, cooldown])
                all_finals.append({**row, "a10c4_status": control["a10c4_status"],
                                   "a10c4_iterations": control["a10c4_iterations"],
                                   "a10c4_replacements": control["a10c4_replacements"],
                                   "a10c4_best_true_relative": control["a10c4_best_true_relative"],
                                   "a10c5_status": control["status"],
                                   "a10c5_iterations": control["iterations"],
                                   "a10c5_replacements": control["replacements"],
                                   "a10c5_best_true_relative": control["best_true_relative"],
                                   "process_elapsed_s": f"{elapsed:.6f}",
                                   "process_peak_rss_kib": str(peak)})
    if len(all_finals) != 72:
        raise ValueError("A10C6 requires exactly 72 original-CSR shadow outcomes")
    return all_checks, all_finals


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: C1H12_A10C6_shadow_gate.py FROZEN_A10C5 FROZEN_A10C4 FROZEN_A10B OUTPUT_DIR")
    try:
        out = Path(sys.argv[4])
        checks, results = validate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), out)
        write_csv(out / "shadow_residual_trace.csv", CHECK_COLUMNS, checks)
        write_csv(out / "shadow_comparison.csv", FINAL_COLUMNS, results)
        for cd in COOLDOWNS:
            group = [row for row in results if int(row["cooldown"]) == cd]
            regressions = sum(
                r["a10c4_status"] == "Converged" and r["status"] != "Converged"
                for r in group
            )
            print(f"  cooldown={cd}: Converged={sum(r['status']=='Converged' for r in group)}/{len(group)}; "
                  f"MaxIterations={sum(r['status']=='MaxIterations' for r in group)}; "
                  f"Breakdown={sum(r['status']=='Breakdown' for r in group)}; "
                  f"candidate_count={sum(int(r['first_stagnation_candidate'])>0 for r in group)}; "
                  f"recoveries_after_candidate={sum(r['converged_after_candidate']=='1' for r in group)}; "
                  f"A10C5_converged={sum(r['a10c5_status']=='Converged' for r in group)}; "
                  f"A10C4_converged={sum(r['a10c4_status']=='Converged' for r in group)}; "
                  f"A10C4_convergence_regressions={regressions}; "
                  f"accuracy_nonregression={'PASS' if regressions == 0 else 'NOT_MET'}")
        print(f"C1-H12-A10C6 {len(checks)} original-CSR checkpoints, 72/72 prefix/cooldown/shadow/provenance gates PASS")
        print("Scope: diagnostic continuation only; convergence regressions are reported, not concealed by evidence gate PASS")
    except (OSError, KeyError, ValueError, OverflowError) as error:
        raise SystemExit(f"C1-H12-A10C6 evidence FAIL: {error}") from error
