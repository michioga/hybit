#!/usr/bin/env python3
"""Fail-closed phase attribution against frozen A10C7 numerical trajectories.

Timing is descriptive; this tool does not establish a production speedup.
"""

from __future__ import annotations

import csv
import hashlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks.scripts import C1H12_A10C7_matched_gate as previous

SESSION_TAG = "C1H12A10C8A_SESSION"
SAMPLE_TAG = "C1H12A10C8A_SAMPLE"
PHASES = (
    "csr_ns", "kahan_ns", "precond_ns", "spmv_ns", "dot_ns", "vector_ns",
    "replace_spmv_ns", "replace_vector_ns", "direction_ns", "other_ns",
)
COUNTERS = (
    "timed_precond_calls", "timed_spmv_calls", "timed_csr_calls",
    "timed_kahan_calls", "timed_replacements",
)
NUMERICAL_FIELDS = (
    "case", "arm", "policy", "rhs_mode", "rhs_hash", "tol", "cooldown",
    "solver", "rep", "order", "status", "iterations", "recursive_relative",
    "true_relative", "compensated_relative", "precond_calls", "true_checks",
    "replacements", "false_events", "best_true_relative",
    "first_stagnation_candidate",
)
SAMPLE_FIELDS = frozenset((*NUMERICAL_FIELDS, "solve_ns", *PHASES, *COUNTERS))
SAMPLE_COLUMNS = (*NUMERICAL_FIELDS, "solve_ns", *PHASES, *COUNTERS,
                  "process_elapsed_s", "process_peak_rss_kib")
SUMMARY_COLUMNS = (
    "case", "arm", "policy", "rhs_mode", "tol", "cooldown", "solver",
    "status", "iterations", "replacements", "samples", "setup_ns",
    "median_solve_ns", *(f"median_{phase}" for phase in PHASES),
    "phase_sum_is_exclusive", "performance_status",
)


def record_key(row: dict[str, str]) -> tuple:
    key = previous.identity(row, "cooldown")
    solver = row["solver"]
    repetition = previous.integer(row["rep"], "rep", 1)
    if solver not in previous.SOLVERS or repetition > previous.REPETITIONS:
        raise ValueError("invalid matched solver/repetition")
    return (*key, solver, repetition)


def parse_process(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    sessions, samples = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(SESSION_TAG + "|"):
            row = previous.parse_pipe(line, SESSION_TAG)
            if set(row) != previous.SESSION_FIELDS | {"profile_version"}:
                raise ValueError("wrong profiler session schema")
            sessions.append(row)
        elif line.startswith(SAMPLE_TAG + "|"):
            row = previous.parse_pipe(line, SAMPLE_TAG)
            if set(row) != SAMPLE_FIELDS:
                raise ValueError("wrong profiler sample schema")
            samples.append(row)
        elif line.startswith("C1H12A10C4_CHECK|") or line.startswith("C1H12A10C6_CHECK|"):
            raise ValueError("checkpoint console IO enabled during measurement")
    if len(sessions) != 1 or len(samples) != 144:
        raise ValueError("expected one session and 144 profiler samples per process")
    if sessions[0]["profile_version"] != "1":
        raise ValueError("unknown timing profile version")
    return sessions[0], samples


def compare_frozen_sample(actual: dict[str, str], frozen: dict[str, str]) -> None:
    for field in NUMERICAL_FIELDS:
        if field in ("tol", "recursive_relative", "true_relative",
                     "compensated_relative", "best_true_relative"):
            previous.close_numeric(actual[field], frozen[field], field)
        elif actual[field] != frozen[field]:
            raise ValueError(f"changed A10C7 frozen numerical trajectory: {field}")


def validate_grid(
    samples: list[dict[str, str]], frozen: list[dict[str, str]],
    setup: dict[tuple[str, str], int],
) -> list[dict[str, str]]:
    old = {}
    new = {}
    for row in frozen:
        key = record_key(row)
        if key in old:
            raise ValueError("duplicate frozen matched sample")
        old[key] = row
    if len(old) != 864:
        raise ValueError("incomplete frozen A10C7 samples")
    for row in samples:
        key = record_key(row)
        if key not in old or key in new:
            raise ValueError("missing/duplicate/out-of-grid A10C8-A sample")
        compare_frozen_sample(row, old[key])
        solve = previous.integer(row["solve_ns"], "solve_ns", 1)
        periods = {name: previous.integer(row[name], name) for name in PHASES}
        counts = {name: previous.integer(row[name], name) for name in COUNTERS}
        if sum(periods.values()) != solve:
            raise ValueError("overlapping or incomplete phase-accounting budget")
        if periods["csr_ns"] == 0 or periods["kahan_ns"] == 0 \
                or periods["precond_ns"] == 0 or periods["spmv_ns"] == 0:
            raise ValueError("missing essential phase timer")
        checks = previous.integer(row["true_checks"], "true_checks", 1)
        calls = previous.integer(row["precond_calls"], "precond_calls", 1)
        iterations = previous.integer(row["iterations"], "iterations", 1)
        replacements = previous.integer(row["replacements"], "replacements")
        if counts != {
            "timed_precond_calls": calls,
            "timed_spmv_calls": iterations,
            "timed_csr_calls": checks - 1,
            "timed_kahan_calls": checks - 1,
            "timed_replacements": replacements,
        }:
            raise ValueError("timing call counters do not account for numerical work")
        for phase in ("replace_spmv_ns", "replace_vector_ns"):
            if (periods[phase] > 0) != (replacements > 0):
                raise ValueError("missing or forged replacement phase")
        if row["status"] == "Converged":
            tol = previous.number(row["tol"], "tol")
            if any(previous.number(row[part], part) > tol for part in
                   ("recursive_relative", "true_relative", "compensated_relative")):
                raise ValueError("false original-CSR or Kahan convergence")
        new[key] = row
    if set(new) != set(old) or len(new) != 864:
        raise ValueError("incomplete 864 matched phase samples")
    if len(setup) != 6:
        raise ValueError("missing six hierarchy setup profiles")
    summaries = []
    for arm in previous.ARMS:
        for policy in previous.POLICIES.values():
            for rhs in previous.RHS:
                for tol in previous.TOLS:
                    for cooldown in previous.COOLDOWNS:
                        for solver in previous.SOLVERS:
                            group = [new[(arm, policy, rhs, tol, cooldown, solver, rep)]
                                     for rep in range(1, previous.REPETITIONS + 1)]
                            first = group[0]
                            med = lambda name: statistics.median(
                                int(row[name]) for row in group)
                            summaries.append({
                                "case": "multi_patch", "arm": arm, "policy": policy,
                                "rhs_mode": rhs, "tol": f"{tol:.0e}",
                                "cooldown": str(cooldown), "solver": solver,
                                "status": first["status"], "iterations": first["iterations"],
                                "replacements": first["replacements"], "samples": "6",
                                "setup_ns": str(setup[arm, policy]),
                                "median_solve_ns": f"{med('solve_ns'):.1f}",
                                **{f"median_{phase}": f"{med(phase):.1f}" for phase in PHASES},
                                "phase_sum_is_exclusive": "true",
                                "performance_status": "ATTRIBUTION_ONLY",
                            })
    return summaries


def verify_frozen_sha(evidence: Path, frozen_dir: Path) -> None:
    for name in ("matched_timing_samples.csv", "matched_timing_summary.csv"):
        expected = hashlib.sha256((frozen_dir / name).read_bytes()).hexdigest()
        lines = (evidence / "FROZEN_A10C7_SHA256.txt").read_text(encoding="utf-8").splitlines()
        matches = [line for line in lines if line.endswith("  " + str(frozen_dir / name))]
        if len(matches) != 1 or matches[0] != f"{expected}  {frozen_dir / name}":
            raise ValueError("frozen A10C7 CSV SHA/provenance disagreement")
    if len(lines) != 2:
        raise ValueError("frozen A10C7 fingerprint file has unexpected entries")


def load_frozen_samples(path: Path) -> list[dict[str, str]]:
    return previous.read_csv(path, previous.SAMPLE_COLUMNS, 864)


def validate(
    a10c7: Path, a10c6: Path, a10c4: Path, a10b: Path, evidence: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    for left, right in (
        (evidence / "FROZEN_A10C7_HEAD.txt", a10c7 / "HEAD.txt"),
        (evidence / "FROZEN_A10C6_HEAD.txt", a10c6 / "HEAD.txt"),
        (evidence / "FROZEN_A10C4_HEAD.txt", a10c4 / "HEAD.txt"),
        (evidence / "FROZEN_A10B_HEAD.txt", a10b / "HEAD.txt"),
        (evidence / "matrix_sha256.txt", a10b / "matrix_sha256.txt"),
        (a10c7 / "matrix_sha256.txt", a10b / "matrix_sha256.txt"),
    ):
        if left.read_bytes() != right.read_bytes():
            raise ValueError("frozen HEAD or matrix provenance mismatch")
    verify_frozen_sha(evidence, a10c7)
    c7_samples, c7_summary = previous.validate(a10c6, a10c4, a10b, a10c7)
    if c7_samples != load_frozen_samples(a10c7 / "matched_timing_samples.csv"):
        raise ValueError("frozen A10C7 CSV differs from its verified raw logs")
    if c7_summary != previous.read_csv(
        a10c7 / "matched_timing_summary.csv", previous.SUMMARY_COLUMNS, 72
    ):
        raise ValueError("frozen A10C7 summary differs from its verified raw logs")
    new_samples = []
    setup = {}
    for arm in previous.ARMS:
        for short, policy in previous.POLICIES.items():
            stem = f"multi_patch_{arm}_{short}_1"
            session, records = parse_process(evidence / f"{stem}.log")
            setup[arm, policy] = previous.validate_session(
                {k: v for k, v in session.items() if k != "profile_version"}, arm, policy
            )
            expected_sequence = []
            for rhs in previous.RHS:
                for tol in previous.TOLS:
                    for cooldown in previous.COOLDOWNS:
                        for repetition in range(1, previous.REPETITIONS + 1):
                            order = ("shadow", "reliable") if repetition % 2 else ("reliable", "shadow")
                            expected_sequence.extend((rhs, tol, cooldown, repetition, solver)
                                                     for solver in order)
            actual_sequence = [
                (row["rhs_mode"], previous.number(row["tol"], "tol"),
                 previous.integer(row["cooldown"], "cooldown"),
                 previous.integer(row["rep"], "rep"), row["solver"])
                for row in records
            ]
            if actual_sequence != expected_sequence:
                raise ValueError("changed AB/BA chronological matched order")
            elapsed, peak = previous.read_time(evidence / f"{stem}.time")
            new_samples.extend({**row, "process_elapsed_s": elapsed,
                                "process_peak_rss_kib": peak} for row in records)
    return new_samples, validate_grid(new_samples, c7_samples, setup)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: C1H12_A10C8A_profile_gate.py A10C7 A10C6 A10C4 A10B NEW_EVIDENCE")
    try:
        root = Path(sys.argv[5])
        records, summary = validate(*(Path(v) for v in sys.argv[1:5]), root)
        write_csv(root / "phase_profile_samples.csv", SAMPLE_COLUMNS, records)
        write_csv(root / "phase_profile_summary.csv", SUMMARY_COLUMNS, summary)
        print("C1-H12-A10C8-A 864/864 disjoint phase budgets, 144/144 six-repeat summaries PASS")
        print("Frozen A10C7 numerical trajectories and AB/BA order: 864/864 PASS")
        print("Scope: instrumented exploratory attribution only; no speedup or production guarantee")
    except (OSError, KeyError, ValueError, OverflowError) as exc:
        raise SystemExit(f"C1-H12-A10C8-A evidence FAIL: {exc}") from exc
