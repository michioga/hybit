#!/usr/bin/env python3
"""Repeat the existing fem_bench CLI under a fixed local-factor budget.

Linux/WSL2 only: wait4 records each child process's peak RSS. The runner does
not change solver defaults, assert SPD, or convert a manufactured RHS into a
physical FEM load. Results include every failed and warmup invocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
FIELDS = [
    "run_id", "phase", "round", "max_region", "arm", "pair_id",
    "exit_code", "timed_out",
    "process_wall_ms", "peak_rss_kib", "hybit_pass", "plain_pass",
    "paired_pass", "failure_reason", "dimensions", "nnz", "load_ms",
    "hybit_status", "hybit_backend", "hybit_preconditioner",
    "hybit_iterations", "hybit_verified_residual", "hybit_x_error",
    "hybit_wall_ms", "hybit_solve_ms", "hybit_analysis_ms",
    "hybit_prepare_ms", "hybit_diagnostics_ms", "hybit_local_factor_ms",
    "hybit_probe_iterations", "hybit_probe_ms", "hybit_escalations",
    "hybit_monitor_checks", "hybit_monitor_trigger_iteration",
    "hybit_monitor_last_rate", "hybit_monitor_worst_rate", "hybit_monitor_candidate_count",
    "hybit_local_regions", "hybit_largest_region", "hybit_factor_mib",
    "hybit_budget_limited", "hybit_budget_skipped", "plain_status",
    "plain_iterations", "plain_verified_residual", "plain_x_error",
    "plain_setup_ms", "plain_solve_ms", "stdout_log", "stderr_log",
]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fields(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$", line)
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def number(section: dict[str, str], key: str) -> float | None:
    value = section.get(key)
    if value is None:
        return None
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def integer(section: dict[str, str], key: str) -> int | None:
    value = number(section, key)
    return int(value) if value is not None and math.isfinite(value) else None


def parse_fem_output(output: str) -> dict[str, object]:
    """Parse the public example's named sections; never infer convergence."""
    result: dict[str, object] = {}
    parts = re.split(r"\[\d/\d\] HyBIT Auto\s*\n", output, maxsplit=1)
    header = parts[0]
    hybit = fields(parts[1].split("\nComparison", 1)[0]) if len(parts) == 2 else {}
    plain_part = re.split(r"\[1/2\] Plain Jacobi-PCG\s*\n", header, maxsplit=1)
    plain = fields(plain_part[1]) if len(plain_part) == 2 else {}
    metadata = fields(plain_part[0])
    result["dimensions"] = metadata.get("dimensions")
    result["nnz"] = integer(metadata, "nnz")
    result["load_ms"] = number(metadata, "load time")

    for prefix, section in (("hybit", hybit), ("plain", plain)):
        result[f"{prefix}_status"] = section.get("status")
        result[f"{prefix}_iterations"] = integer(section, "iterations")
        result[f"{prefix}_verified_residual"] = number(section, "verified residual")
        result[f"{prefix}_x_error"] = number(section, "relative x error")

    result.update(
        hybit_backend=hybit.get("backend"),
        hybit_preconditioner=hybit.get("preconditioner"),
        hybit_wall_ms=number(hybit, "total wall"),
        hybit_solve_ms=number(hybit, "solver time"),
        hybit_analysis_ms=number(hybit, "analysis"),
        hybit_prepare_ms=number(hybit, "prepare"),
        hybit_diagnostics_ms=number(hybit, "diagnostics"),
        hybit_local_factor_ms=number(hybit, "local factor"),
        hybit_escalations=integer(hybit, "escalations"),
        hybit_monitor_checks=integer(hybit, "monitor checks"),
        hybit_monitor_trigger_iteration=integer(hybit, "monitor trigger"),
        hybit_local_regions=integer(hybit, "local regions"),
        hybit_largest_region=integer(hybit, "largest region"),
        hybit_factor_mib=number(hybit, "factor memory"),
        hybit_budget_limited=hybit.get("budget limited"),
        hybit_budget_skipped=integer(hybit, "budget skipped"),
    )
    probe = re.match(r"(\d+) iters, ([\d.eE+-]+) ms", hybit.get("probe", ""))
    result["hybit_probe_iterations"] = int(probe.group(1)) if probe else None
    result["hybit_probe_ms"] = float(probe.group(2)) if probe else None
    monitor_samples = re.findall(
        r"^\s+monitor iter\s+(\d+)\s*:\s*true residual\s+\S+, "
        r"per-iter ratio\s+(\S+), candidate\s+(true|false)\s*$",
        output, flags=re.MULTILINE,
    )
    monitor_rates = [float(value) for _, value, _ in monitor_samples
                     if value != "baseline"]
    result["hybit_monitor_last_rate"] = monitor_rates[-1] if monitor_rates else None
    result["hybit_monitor_worst_rate"] = max(monitor_rates) if monitor_rates else None
    result["hybit_monitor_candidate_count"] = sum(
        candidate == "true" for _, _, candidate in monitor_samples
    )
    setup = re.match(r"([\d.eE+-]+) / ([\d.eE+-]+) ms", plain.get("setup / solve", ""))
    result["plain_setup_ms"] = float(setup.group(1)) if setup else None
    result["plain_solve_ms"] = float(setup.group(2)) if setup else None
    return result


def numeric_gate(row: dict[str, object], tol: float, include_plain: bool, budget: int) -> None:
    """Both methods are evaluated independently; failure rows are retained."""
    reasons: list[str] = []
    if row["exit_code"] != 0:
        reasons.append("process_exit")
    if row["timed_out"]:
        reasons.append("timeout")

    for prefix in ("hybit", "plain"):
        if prefix == "plain" and not include_plain:
            row["plain_pass"] = None
            continue
        status = row.get(f"{prefix}_status")
        residual = row.get(f"{prefix}_verified_residual")
        valid = status == "Converged" and isinstance(residual, float)
        valid = valid and math.isfinite(residual) and residual <= tol * (1 + 2e-6)
        if prefix == "hybit":
            factor = row.get("hybit_factor_mib")
            # fem_bench prints three decimals; tolerate that rounding only.
            valid = valid and isinstance(factor, float) and math.isfinite(factor)
            valid = valid and factor <= budget + 0.0005
        row[f"{prefix}_pass"] = bool(valid) and row["exit_code"] == 0
        if not row[f"{prefix}_pass"]:
            reasons.append(f"{prefix}_numerical_or_budget")

    row["paired_pass"] = row["hybit_pass"] and (
        row["plain_pass"] if include_plain else True
    )
    row["failure_reason"] = ";".join(reasons)


def strict_json_row(row: dict[str, object]) -> dict[str, object]:
    """Keep invalid numerical evidence in raw logs, emit strict JSON nulls."""
    return {key: (None if isinstance(value, float) and not math.isfinite(value)
                  else value) for key, value in row.items()}


def run_process(command: list[str], cwd: Path, stdout: Path, stderr: Path,
                timeout: float, env: dict[str, str]) -> tuple[int, bool, float, int]:
    """wait4 gives per-invocation peak RSS (KiB on Linux), not cumulative RSS."""
    start = time.perf_counter()
    expired = False
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdout=out, stderr=err, env=env)
        except OSError as exc:
            err.write(f"Failed to launch: {exc}\n".encode())
            return 127, False, round((time.perf_counter() - start) * 1000, 3), 0
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    break
                if time.perf_counter() - start > timeout:
                    expired = True
                    process.kill()
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                # Polling overhead affects sub-millisecond fixtures; real FEM
                # performance comparisons should use fem_bench's inner timing.
                time.sleep(0.005)
        except BaseException:
            process.kill()
            process.wait()
            raise
        process.returncode = os.waitstatus_to_exitcode(status)
    return (process.returncode, expired,
            round((time.perf_counter() - start) * 1000, 3), usage.ru_maxrss)


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_regions(value: str) -> list[int]:
    try:
        values = [parse_positive_int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("regions must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("regions must be distinct positive integers")
    return values


def parse_cpus(value: str) -> list[int]:
    result: list[int] = []
    try:
        for part in value.split(","):
            bounds = [int(item) for item in part.strip().split("-")]
            if len(bounds) == 1:
                result.append(bounds[0])
            elif len(bounds) == 2 and bounds[1] >= bounds[0]:
                result.extend(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError("invalid CPU range")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cpus must look like 0-3,6,8") from exc
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("CPU IDs must be unique nonnegative integers")
    return result


def command_version(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL,
                                       text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def metadata(args: argparse.Namespace, command: list[str], affinity: list[int]) -> dict[str, object]:
    return {
        "schema": "hybit-c1-fem-region-sweep-v2",
        "matrix": str(args.matrix), "matrix_sha256": hash_file(args.matrix),
        "rhs": str(args.rhs) if args.rhs else None,
        "rhs_sha256": hash_file(args.rhs) if args.rhs else None,
        "spd_certification": "not performed by this runner",
        "rhs_kind": "provided" if args.rhs else "manufactured_A_times_ones",
        "binary": str(args.binary), "binary_sha256": hash_file(args.binary),
        "git_head": command_version(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": bool(command_version(["git", "-C", str(REPO), "status", "--porcelain"])),
        "rustc": command_version(["rustc", "--version"]),
        "host": platform.node(), "kernel": platform.release(),
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), None),
        "cpu_affinity": affinity,
        "cpus_requested": args.cpus,
        "threads": args.threads, "seed": args.seed,
        "warmups": args.warmups, "repetitions": args.repetitions,
        "max_regions": args.max_regions, "region_sizes": args.regions,
        "factor_budget_mib": args.factor_budget_mib,
        "tolerance": args.tol, "max_iterations": args.max_iters,
        "monitor_interval": args.monitor_iters,
        "monitor_observe_only": args.monitor_only,
        "monitor_rate_limit": args.monitor_rate_limit,
        "paired_observe_control": args.paired_observe_control,
        "max_escalations": args.max_escalations,
        "stage_iterations": args.stage_iters,
        "plain_included": not args.skip_plain,
        "command_template": command + ["--max-region", "<per-run>"],
        "paired_arm_rule": ("observe: --monitor-only; escalate: same command without it; "
                            "order shuffled within each region/round"
                            if args.paired_observe_control else None),
        "timer_scope": "separate child launch; includes matrix load and both solves; OS file cache may be warm",
        "hybit_wall_scope": "reported by fem_bench; excludes matrix load/plain",
        "peak_rss_scope": "wait4 ru_maxrss for one child (Linux KiB)",
        "prepared_multi_rhs_measured": False,
    }


def summarize(rows: list[dict[str, object]], regions: list[int],
              paired_observe_control: bool = False) -> dict[str, object]:
    result: dict[str, object] = {}
    for region in regions:
        measured = [row for row in rows if row["max_region"] == region
                    and row["phase"] == "measure"
                    and (not paired_observe_control or row.get("arm") == "escalate")]
        valid = [row for row in measured if row["paired_pass"]]
        escalated = [row for row in measured if (row.get("hybit_escalations") or 0) > 0]
        result[str(region)] = {
            "trials": len(measured), "hybit_pass": sum(bool(row["hybit_pass"]) for row in measured),
            "paired_pass": len(valid), "escalated_trials": len(escalated),
            "region_size_informative": bool(escalated),
            "median_hybit_wall_ms": statistics.median(row["hybit_wall_ms"] for row in valid) if valid else None,
            "median_process_wall_ms": statistics.median(row["process_wall_ms"] for row in valid) if valid else None,
            "median_peak_rss_kib": statistics.median(row["peak_rss_kib"] for row in valid) if valid else None,
        }
        if paired_observe_control:
            # Pair only identical region/round/RHS settings. If either arm
            # fails, or escalation never happened, do not count its timing as
            # evidence about the economics of local direct correction.
            paired_rows: dict[str, dict[str, dict[str, object]]] = {}
            for row in rows:
                if row["max_region"] == region and row["phase"] == "measure":
                    paired_rows.setdefault(str(row["pair_id"]), {})[str(row["arm"])] = row
            complete = [pair for pair in paired_rows.values()
                        if "observe" in pair and "escalate" in pair]
            numerically_valid = [pair for pair in complete
                                 if pair["observe"]["paired_pass"]
                                 and pair["escalate"]["paired_pass"]]
            informative = [pair for pair in numerically_valid
                           if (pair["observe"].get("hybit_escalations") or 0) == 0
                           and (pair["escalate"].get("hybit_escalations") or 0) > 0
                           and all(isinstance(pair[arm].get("hybit_wall_ms"), (float, int))
                                   and math.isfinite(pair[arm]["hybit_wall_ms"])
                                   and pair[arm]["hybit_wall_ms"] > 0
                                   and isinstance(pair[arm].get("peak_rss_kib"), (float, int))
                                   and math.isfinite(pair[arm]["peak_rss_kib"])
                                   and isinstance(pair[arm].get("hybit_iterations"), int)
                                   for arm in ("observe", "escalate"))]
            savings = [pair["observe"]["hybit_wall_ms"] - pair["escalate"]["hybit_wall_ms"]
                       for pair in informative]
            gains = [100 * saved / pair["observe"]["hybit_wall_ms"]
                     for saved, pair in zip(savings, informative)]
            result[str(region)]["paired_control"] = {
                "complete_pairs": len(complete),
                "numerically_valid_pairs": len(numerically_valid),
                "informative_escalation_pairs": len(informative),
                "treatment_faster_pairs": sum(saved > 0 for saved in savings),
                "median_hybit_wall_saved_ms": statistics.median(savings) if savings else None,
                "median_relative_wall_gain_percent": statistics.median(gains) if gains else None,
                "median_peak_rss_change_kib": (
                    statistics.median(pair["escalate"]["peak_rss_kib"]
                                      - pair["observe"]["peak_rss_kib"]
                                      for pair in informative) if informative else None),
                "median_iteration_change": (
                    statistics.median(pair["escalate"]["hybit_iterations"]
                                      - pair["observe"]["hybit_iterations"]
                                      for pair in informative) if informative else None),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--rhs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="must not exist; results are never silently overwritten")
    parser.add_argument("--binary", type=Path, default=REPO / "target/release/examples/fem_bench")
    parser.add_argument("--regions", type=parse_regions, default=[64, 128, 256, 512])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=parse_positive_int, default=9)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iters", type=parse_positive_int, default=3000)
    parser.add_argument("--probe-iters", type=parse_positive_int, default=12)
    parser.add_argument("--monitor-iters", type=int, default=0,
                        help="opt-in true-residual checkpoint interval; 0 disables")
    parser.add_argument("--monitor-only", action="store_true",
                        help="observe residual history without any local escalation")
    parser.add_argument("--paired-observe-control", action="store_true",
                        help="randomized matched monitor-only vs experimental escalation arms")
    parser.add_argument("--monitor-rate-limit", type=float,
                        help="experimental per-iteration residual ratio threshold in (0,1)")
    parser.add_argument("--max-escalations", type=parse_positive_int, default=3)
    parser.add_argument("--stage-iters", type=parse_positive_int, default=24)
    parser.add_argument("--max-regions", type=parse_positive_int, default=8)
    parser.add_argument("--factor-budget-mib", type=parse_positive_int, default=64)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--selector", choices=["candidate-order", "benefit-byte", "jacobi-byte"],
                        default="jacobi-byte")
    parser.add_argument("--backend", choices=["auto", "csr", "abtm"], default="auto")
    parser.add_argument("--threads", type=parse_positive_int, default=1)
    parser.add_argument("--cpus", type=parse_cpus,
                        help="optional Linux affinity of runner and children, e.g. 0-3; default: inherited")
    parser.add_argument("--timeout-seconds", type=parse_positive_int, default=900)
    parser.add_argument("--skip-plain", action="store_true")
    args = parser.parse_args()

    if sys.platform != "linux" or not hasattr(os, "wait4") or not hasattr(os, "sched_getaffinity"):
        parser.error("this memory-accounting runner requires Linux/WSL2")
    if args.warmups < 0 or args.overlap < 0 or args.monitor_iters < 0:
        parser.error("warmups, overlap, and monitor-iters must be nonnegative")
    if not math.isfinite(args.tol) or args.tol <= 0:
        parser.error("tol must be finite and positive")
    if (args.monitor_only or args.monitor_rate_limit is not None) and args.monitor_iters == 0:
        parser.error("monitor-only and monitor-rate-limit require monitor-iters > 0")
    if args.monitor_rate_limit is not None and not (
        math.isfinite(args.monitor_rate_limit) and 0 < args.monitor_rate_limit < 1
    ):
        parser.error("monitor-rate-limit must be finite and in (0,1)")
    if args.paired_observe_control and (
        args.monitor_only or args.monitor_iters == 0 or args.monitor_rate_limit is None
        or not args.skip_plain
    ):
        parser.error("paired-observe-control requires monitor-iters > 0, "
                     "monitor-rate-limit, skip-plain, and no monitor-only")
    args.matrix = args.matrix.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    args.rhs = args.rhs.expanduser().resolve() if args.rhs else None
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.matrix.is_file() or not args.binary.is_file() or (args.rhs and not args.rhs.is_file()):
        parser.error("matrix, binary, and optional RHS must exist")
    if not os.access(args.binary, os.X_OK):
        parser.error("binary is not executable; build with cargo build --release -p hybit --example fem_bench")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")

    command = [str(args.binary), "--matrix", str(args.matrix), "--tol", str(args.tol),
               "--max-iters", str(args.max_iters), "--probe-iters", str(args.probe_iters),
               "--monitor-iters", str(args.monitor_iters),
               "--max-escalations", str(args.max_escalations), "--stage-iters", str(args.stage_iters),
               "--max-regions", str(args.max_regions), "--factor-budget-mib", str(args.factor_budget_mib),
               "--overlap", str(args.overlap), "--selector", args.selector, "--backend", args.backend]
    if args.monitor_only:
        command.append("--monitor-only")
    if args.monitor_rate_limit is not None:
        command.extend(["--monitor-rate-limit", str(args.monitor_rate_limit)])
    if args.rhs:
        command.extend(["--rhs", str(args.rhs)])
    if args.skip_plain:
        command.append("--skip-plain")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), RAYON_NUM_THREADS=str(args.threads))
    available_cpus = os.sched_getaffinity(0)
    if args.cpus:
        if not set(args.cpus).issubset(available_cpus):
            parser.error(f"requested cpus not in allowed affinity {sorted(available_cpus)}")
        try:
            os.sched_setaffinity(0, args.cpus)
        except OSError as exc:
            parser.error(f"cannot set CPU affinity: {exc}")
    affinity = sorted(os.sched_getaffinity(0))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    logdir = args.output_dir / "logs"
    logdir.mkdir()
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata(args, command, affinity), indent=2, ensure_ascii=False) + "\n"
    )

    rows: list[dict[str, object]] = []
    rng = random.Random(args.seed)
    with (args.output_dir / "results.csv").open("w", newline="") as csv_file, \
         (args.output_dir / "results.jsonl").open("w") as json_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        run_id = 0
        for phase, count in (("warmup", args.warmups), ("measure", args.repetitions)):
            for round_number in range(1, count + 1):
                sizes = args.regions[:]
                rng.shuffle(sizes)
                for region in sizes:
                    pair_id = f"{phase}_round{round_number}_region{region}"
                    arms = ["observe", "escalate"] if args.paired_observe_control else ["standard"]
                    rng.shuffle(arms)
                    for arm in arms:
                        run_id += 1
                        stem = f"{run_id:03d}_{pair_id}_{arm}"
                        outlog, errlog = logdir / f"{stem}.stdout.log", logdir / f"{stem}.stderr.log"
                        cmd = command + ["--max-region", str(region)]
                        if arm == "observe":
                            cmd.append("--monitor-only")
                        exit_code, timed_out, wall, rss = run_process(
                            cmd, REPO, outlog, errlog, args.timeout_seconds, env
                        )
                        row: dict[str, object] = dict.fromkeys(FIELDS)
                        row.update(run_id=run_id, phase=phase, round=round_number,
                                   max_region=region, arm=arm,
                                   pair_id=pair_id if args.paired_observe_control else None,
                                   exit_code=exit_code, timed_out=timed_out,
                                   process_wall_ms=wall, peak_rss_kib=rss,
                                   stdout_log=str(outlog.relative_to(args.output_dir)),
                                   stderr_log=str(errlog.relative_to(args.output_dir)))
                        row.update(parse_fem_output(outlog.read_text(errors="replace")))
                        numeric_gate(row, args.tol, not args.skip_plain, args.factor_budget_mib)
                        # Retain the failure and its raw log without emitting
                        # non-standard NaN/Infinity into strict JSONL.
                        row = strict_json_row(row)
                        writer.writerow(row)
                        csv_file.flush()
                        json_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        json_file.flush()
                        rows.append(row)
                        print(f"[{run_id}] {phase} r={region} arm={arm} "
                              f"hybit={row['hybit_pass']} plain={row['plain_pass']} "
                              f"escalations={row['hybit_escalations']} "
                              f"process_ms={wall:.1f} rss_kib={rss} "
                              f"{row['failure_reason']}", flush=True)

    summary = summarize(rows, args.regions, args.paired_observe_control)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output_dir} ; summary: {json.dumps(summary, ensure_ascii=False)}")
    return 0 if all(row["paired_pass"] for row in rows if row["phase"] == "measure") else 1


if __name__ == "__main__":
    raise SystemExit(main())
