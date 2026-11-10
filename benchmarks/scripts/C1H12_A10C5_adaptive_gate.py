#!/usr/bin/env python3
"""Fail-closed original-CSR A10C5 cooldown/stagnation experiment gate."""

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

CASE = "multi_patch"
RHS = ("Ones", "PointLoad")
TOLS = (1e-10, 1e-11, 1e-12)
COOLDOWNS = (25, 50)
MAX_ITER = 1200
WINDOW = 200
FACTOR = 0.9
SPACING = 5
SESSION = "C1H12A10C5_SESSION"
CHECK = "C1H12A10C5_CHECK"
FINAL = "C1H12A10C5_FINAL"
IDENTITY = {"case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown"}
CHECK_KEYS = IDENTITY | {
    "iteration", "recursive_relative", "true_relative", "compensated_relative", "gap_ratio",
    "replacement_applied", "cooldown_suppressed", "last_progress_iteration", "precond_calls",
}
FINAL_KEYS = IDENTITY | {
    "status", "iterations", "recursive_relative", "true_relative", "compensated_relative",
    "precond_calls", "true_checks", "replacements", "false_events", "cooldown_suppressed",
    "best_true_relative", "last_progress_iteration", "relative_x_error",
}
SESSION_KEYS = {
    "case", "arm", "policy", "rows", "nnz", "rhs_count", "tol_count", "cooldown_count",
    "stagnation_window", "progress_factor", "threshold_spacing", "max_iterations",
    "level_dims", "level_nnz", "abtm_words",
}
CHECK_COLUMNS = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown",
    "iteration", "recursive_relative", "true_relative", "compensated_relative", "gap_ratio",
    "replacement_applied", "cooldown_suppressed", "last_progress_iteration", "precond_calls",
)
FINAL_COLUMNS = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown",
    "status", "iterations", "recursive_relative", "true_relative", "compensated_relative",
    "precond_calls", "true_checks", "replacements", "false_events", "cooldown_suppressed",
    "best_true_relative", "last_progress_iteration", "relative_x_error",
    "a10c4_status", "a10c4_iterations", "a10c4_replacements", "a10c4_best_true_relative",
    "process_elapsed_s", "process_peak_rss_kib",
)


def parse_records(path: Path) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    sessions, checks, finals = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(SESSION + "|"):
            row = parse_pipe(line, SESSION)
            if set(row) != SESSION_KEYS:
                raise ValueError("A10C5 session schema mismatch")
            sessions.append(row)
        elif line.startswith(CHECK + "|"):
            row = parse_pipe(line, CHECK)
            if set(row) != CHECK_KEYS:
                raise ValueError("A10C5 original-CSR checkpoint schema mismatch")
            checks.append(row)
        elif line.startswith(FINAL + "|"):
            row = parse_pipe(line, FINAL)
            if set(row) != FINAL_KEYS:
                raise ValueError("A10C5 final record schema mismatch")
            finals.append(row)
    if len(sessions) != 1 or len(finals) != 12 or not checks:
        raise ValueError("A10C5 missing session, twelve outcomes or checkpoints")
    return sessions[0], checks, finals


def read_frozen_a10c4(frozen: Path) -> dict[tuple[str, str, str, float, int], dict[str, str]]:
    with (frozen / "reliable_update_comparison.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"case", "arm", "policy", "rhs_mode", "rhs_hash", "tol",
                    "replacement_period", "status", "iterations", "replacements",
                    "best_true_relative"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("A10C4 frozen outcome schema mismatch")
        rows = list(reader)
    if len(rows) != 108:
        raise ValueError("A10C4 frozen evidence must have 108 outcomes")
    result = {}
    for row in rows:
        key = (row["arm"], row["policy"], row["rhs_mode"], finite(row["tol"], "frozen tol"),
               integer(row["replacement_period"], "frozen replacement_period"))
        if row["case"] != CASE or key in result:
            raise ValueError("A10C4 frozen outcome identity mismatch")
        result[key] = row
    expected = {(a, p, rhs, tol, period) for a in ARMS for p in POLICIES.values()
                for rhs in RHS for tol in TOLS for period in (0, 25, 50)}
    if set(result) != expected:
        raise ValueError("A10C4 frozen outcome grid incomplete")
    return result


def check_frozen_hashes(evidence: Path, frozen: Path) -> None:
    lines = (evidence / "FROZEN_A10C4_SHA256.txt").read_text(encoding="utf-8").splitlines()
    expected = {}
    for name in ("reliable_update_comparison.csv", "reliable_residual_trace.csv"):
        path = frozen / name
        expected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    found = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) in found:
            raise ValueError("A10C5 malformed frozen A10C4 fingerprints")
        found[match.group(2)] = match.group(1)
    if found != expected:
        raise ValueError("A10C5 frozen A10C4 evidence hash changed")


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
            or integer(row["max_iterations"], "max_iterations") != MAX_ITER):
        raise ValueError("A10C5 experiment controls changed")
    dims = [integer(part, "level_dims", 1) for part in row["level_dims"].split(",")]
    nnz = [integer(part, "level_nnz") for part in row["level_nnz"].split(",")]
    if (len(dims) < 2 or len(dims) != len(nnz) or dims[0] != 20736
            or nnz[0] != 103104 or any(x <= y for x, y in zip(dims, dims[1:]))):
        raise ValueError("A10C5 invalid AMG hierarchy provenance")
    if policy == "Multilevel" and integer(row["abtm_words"], "abtm_words", 1) == 0:
        raise ValueError("A10C5 multilevel ABTM hierarchy unavailable")


def validate_group(
    checks: list[dict[str, str]], final: dict[str, str],
    identity: tuple[str, str, str, float, int], frozen: dict[str, str],
) -> None:
    arm, policy, rhs_mode, tol, cooldown = identity
    expected_identity = {"case": CASE, "arm": arm, "policy": policy,
                         "rhs_mode": rhs_mode, "rhs_hash": frozen["rhs_hash"]}
    if not checks:
        raise ValueError("A10C5 missing original-CSR checkpoints")
    for row in [*checks, final]:
        if any(row[name] != value for name, value in expected_identity.items()):
            raise ValueError("A10C5 RHS/arm identity disagrees with frozen A10C4")
        if finite(row["tol"], "tol") != tol or integer(row["cooldown"], "cooldown") != cooldown:
            raise ValueError("A10C5 tolerance/cooldown mismatch")
    iterations = integer(final["iterations"], "iterations")
    if iterations > MAX_ITER:
        raise ValueError("A10C5 exceeded iteration budget")
    steps = [integer(row["iteration"], "iteration") for row in checks]
    if steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != iterations:
        raise ValueError("A10C5 missing, duplicated, or out-of-order final checkpoint")
    mandatory = {0, iterations}
    if iterations:
        mandatory.add(1)
    mandatory.update(range(25, iterations + 1, 25))
    if not mandatory.issubset(steps):
        raise ValueError("A10C5 mandatory independent CSR checkpoint missing")
    last_logged, last_replaced, last_progress = 0, 0, 0
    best_certificate, best_true = 1.0, 1.0
    replacements, false_events, suppressed = 0, 0, 0
    for row, step in zip(checks, steps):
        recursive = finite(row["recursive_relative"], "recursive_relative")
        verified = finite(row["true_relative"], "true_relative")
        compensated = finite(row["compensated_relative"], "compensated_relative")
        ratio = finite(row["gap_ratio"], "gap_ratio")
        if not math.isclose(ratio, verified / max(recursive, sys.float_info.min), rel_tol=3e-12, abs_tol=1e-12):
            raise ValueError("A10C5 gap ratio inconsistent with original CSR")
        if integer(row["precond_calls"], "precond_calls") != step:
            raise ValueError("A10C5 preconditioner calls disagree with iteration")
        if step == 0 and any(abs(v - 1.0) > 2e-14 for v in (recursive, verified, compensated)):
            raise ValueError("A10C5 zero-start certificates must equal one")
        if step not in (0, 1, iterations) and step % 25 != 0:
            if recursive > tol or step - last_logged < SPACING:
                raise ValueError("A10C5 unscheduled extra CSR checkpoint")
        if step > 0:
            joint = max(verified, compensated)
            if joint < best_certificate * FACTOR:
                best_certificate = joint
                last_progress = step
        else:
            joint = 1.0
        if integer(row["last_progress_iteration"], "last_progress_iteration") != last_progress:
            raise ValueError("A10C5 reported last progress conflicts with original CSR")
        false = recursive <= tol and joint > tol
        if false:
            false_events += 1
        certified = recursive <= tol and joint <= tol
        stalled = (step >= WINDOW and step - last_progress >= WINDOW and replacements > 0
                   and joint > tol)
        wants = step > 0 and (step % 25 == 0 or false)
        eligible = step - last_replaced >= cooldown
        can_change = step < MAX_ITER and not certified and not stalled
        expected_replace = int(wants and eligible and can_change)
        expected_suppress = int(wants and not eligible and can_change)
        actual_replace = integer(row["replacement_applied"], "replacement_applied")
        actual_suppress = integer(row["cooldown_suppressed"], "cooldown_suppressed")
        if (final["status"] == "Breakdown" and step == iterations
                and (actual_replace, actual_suppress) == (0, 0)):
            # A final audit after numerical breakdown cannot begin another restart.
            expected_replace, expected_suppress = 0, 0
        if actual_replace != expected_replace:
            raise ValueError("A10C5 violated cooldown or replacement policy")
        if actual_suppress != expected_suppress:
            raise ValueError("A10C5 cooldown suppression count disagrees")
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
        raise ValueError("A10C5 work, progress, or event counters disagree")
    if not math.isclose(finite(final["best_true_relative"], "best_true"), best_true,
                        rel_tol=3e-12, abs_tol=1e-14):
        raise ValueError("A10C5 best true residual inconsistent")
    if not (iterations <= integer(final["precond_calls"], "precond_calls") <= iterations + 1):
        raise ValueError("A10C5 final preconditioner calls inconsistent")
    for name in ("recursive_relative", "true_relative", "compensated_relative"):
        if not math.isclose(finite(final[name], name), finite(checks[-1][name], name),
                            rel_tol=3e-12, abs_tol=1e-14):
            raise ValueError("A10C5 final original-CSR certificate changed")
    status = final["status"]
    rec = finite(final["recursive_relative"], "final recursive")
    true = finite(final["true_relative"], "final true")
    kahan = finite(final["compensated_relative"], "final compensated")
    if status == "Converged" and not (rec <= tol and true <= tol and kahan <= tol):
        raise ValueError("A10C5 false convergence incorrectly accepted")
    if status == "Stagnated" and not (iterations >= WINDOW
            and iterations - last_progress >= WINDOW and replacements > 0
            and max(true, kahan) > tol):
        raise ValueError("A10C5 invalid stagnation marker")
    if status == "MaxIterations" and iterations != MAX_ITER:
        raise ValueError("A10C5 premature maximum-iteration status")
    if status not in {"Converged", "Stagnated", "MaxIterations", "Breakdown"}:
        raise ValueError("A10C5 unsupported terminal status")
    if rhs_mode == "PointLoad":
        if final["relative_x_error"] != "n/a":
            raise ValueError("point load must not assert manufactured solution error")
    else:
        finite(final["relative_x_error"], "relative_x_error")


def validate(
    a10c4: Path, a10b: Path, evidence: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    for left, right in (
        (evidence / "HEAD.txt", a10c4 / "HEAD.txt"),
        (evidence / "FROZEN_A10C4_HEAD.txt", a10c4 / "HEAD.txt"),
        (evidence / "FROZEN_A10B_HEAD.txt", a10b / "HEAD.txt"),
        (evidence / "matrix_sha256.txt", a10c4 / "matrix_sha256.txt"),
        (a10c4 / "matrix_sha256.txt", a10b / "matrix_sha256.txt"),
    ):
        if left.read_bytes() != right.read_bytes():
            raise ValueError("A10C5 frozen HEAD or matrix provenance changed")
    check_frozen_hashes(evidence, a10c4)
    reference = read_frozen_a10c4(a10c4)
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
                    raise ValueError("A10C5 duplicate final record")
                results[key] = row
            expected = {(rhs, tol, cd) for rhs in RHS for tol in TOLS for cd in COOLDOWNS}
            if set(groups) != expected or set(results) != expected:
                raise ValueError("A10C5 incomplete 12-outcome process grid")
            for rhs, tol, cooldown in sorted(expected):
                control = reference[arm, policy, rhs, tol, cooldown]
                row = results[rhs, tol, cooldown]
                validate_group(groups[rhs, tol, cooldown], row,
                               (arm, policy, rhs, tol, cooldown), control)
                all_checks.extend(groups[rhs, tol, cooldown])
                all_finals.append({**row, "a10c4_status": control["status"],
                                   "a10c4_iterations": control["iterations"],
                                   "a10c4_replacements": control["replacements"],
                                   "a10c4_best_true_relative": control["best_true_relative"],
                                   "process_elapsed_s": f"{elapsed:.6f}",
                                   "process_peak_rss_kib": str(peak)})
    if len(all_finals) != 72:
        raise ValueError("A10C5 requires exactly 72 original-CSR adaptive outcomes")
    return all_checks, all_finals


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: C1H12_A10C5_adaptive_gate.py FROZEN_A10C4 FROZEN_A10B OUTPUT_DIR")
    try:
        out = Path(sys.argv[3])
        checks, results = validate(Path(sys.argv[1]), Path(sys.argv[2]), out)
        write_csv(out / "adaptive_residual_trace.csv", CHECK_COLUMNS, checks)
        write_csv(out / "adaptive_update_comparison.csv", FINAL_COLUMNS, results)
        for cd in COOLDOWNS:
            group = [row for row in results if int(row["cooldown"]) == cd]
            print(f"  cooldown={cd}: Converged={sum(r['status']=='Converged' for r in group)}/{len(group)}; "
                  f"Stagnated={sum(r['status']=='Stagnated' for r in group)}; "
                  f"MaxIterations={sum(r['status']=='MaxIterations' for r in group)}; "
                  f"replacements={sum(int(r['replacements']) for r in group)}; "
                  f"frozen_A10C4_replacements={sum(int(r['a10c4_replacements']) for r in group)}")
        print(f"C1-H12-A10C5 {len(checks)} original-CSR checkpoints, 72/72 cooldown/stagnation/provenance gates PASS")
        print("Scope: early-stop heuristic and diagnostic PCG only; no production precision/speedup claim")
    except (OSError, KeyError, ValueError, OverflowError) as error:
        raise SystemExit(f"C1-H12-A10C5 evidence FAIL: {error}") from error
