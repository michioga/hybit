#!/usr/bin/env python3
"""C1-D experimental paired crossover for a monitored Jacobi vs Hybrid solve.

Identical one-RHS FEM runs are alternated in fresh Linux processes. This is an
OFFLINE recommendation experiment, never an autonomous production selector.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys

from fem_region_sweep import (hash_file, parse_cpus, parse_fem_output,
                              numeric_gate, run_process, command_version, strict_json_row)

REPO = Path(__file__).resolve().parents[2]
ARMS = ("observe", "hybrid")
FIELDS = [
    "run_id", "phase", "pair", "position", "arm", "exit_code", "timed_out",
    "pass", "failure_reason", "process_wall_ms", "peak_rss_kib",
    "hybit_wall_ms", "hybit_solve_ms", "hybit_status",
    "hybit_iterations", "hybit_verified_residual", "hybit_x_error",
    "hybit_escalations", "hybit_local_regions", "hybit_factor_mib",
    "hybit_diagnostics_ms", "hybit_local_factor_ms", "hybit_monitor_checks",
    "hybit_monitor_trigger_iteration", "stdout_log", "stderr_log",
]


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value > 0


def quantile(values: list[float], proportion: float) -> float:
    sorted_values = sorted(values)
    position = proportion * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def paired_assessment(rows: list[dict], min_pairs: int = 5,
                      margin: float = 0.05, seed: int = 20261030,
                      expected_pairs: int | None = None) -> dict:
    """Diagnose empirical paired whole-solve ratios, not causal preconditioner gains.

    Incomplete/failed pairs are retained and invalidate the decision; a Hybrid
    experiment with no escalation cannot support local-Cholesky economics.
    """
    pairs: dict[int, dict[str, dict]] = {}
    for row in rows:
        if row["phase"] != "measure":
            continue
        arms = pairs.setdefault(int(row["pair"]), {})
        if row["arm"] in arms:
            raise ValueError("duplicate measured pair arm")
        arms[row["arm"]] = row
    ratios = []
    escalated = 0
    valid = 0
    for pair in sorted(pairs):
        arms = pairs[pair]
        if set(arms) != set(ARMS):
            continue
        a, b = arms["observe"], arms["hybrid"]
        if not (a.get("pass") is True and b.get("pass") is True):
            continue
        ta, tb = a.get("hybit_wall_ms"), b.get("hybit_wall_ms")
        if (a.get("hybit_escalations") != 0 or
                not (finite_positive(ta) and finite_positive(tb))):
            continue
        valid += 1
        ratios.append(tb / ta - 1.0)
        escalated += int((b.get("hybit_escalations") or 0) > 0)
    out = {"measured_pairs": len(pairs), "valid_pairs": valid,
           "escalated_pairs": escalated, "margin_fraction": margin,
           "median_relative_cost": statistics.median(ratios) if ratios else None,
           "bootstrap_95pct": None, "classification": "insufficient_evidence",
           "meaning": "offline comparison only; never changes default solver policy"}
    if (valid != len(pairs) or valid < min_pairs or
            (expected_pairs is not None and len(pairs) != expected_pairs)):
        return out
    if escalated != valid:
        out["classification"] = "mixed_or_no_activation"
        return out
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(ratios, k=len(ratios)))
               for _ in range(4000)]
    low, high = quantile(medians, 0.025), quantile(medians, 0.975)
    out["bootstrap_95pct"] = [low, high]
    if high < -margin:
        out["classification"] = "potential_benefit_in_this_matrix"
    elif low > margin:
        out["classification"] = "measured_regression_in_this_matrix"
    else:
        out["classification"] = "inconclusive"
    return out


def positive_int(s: str) -> int:
    try:
        result = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("expected positive integer")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rhs", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=REPO / "target/release/examples/fem_bench")
    p.add_argument("--monitor-iters", type=positive_int, default=64)
    p.add_argument("--monitor-rate-limit", type=float, default=0.99)
    p.add_argument("--max-escalations", type=positive_int, default=1)
    p.add_argument("--max-region", type=positive_int, default=128)
    p.add_argument("--max-regions", type=positive_int, default=8)
    p.add_argument("--factor-budget-mib", type=positive_int, default=64)
    p.add_argument("--probe-iters", type=positive_int, default=12)
    p.add_argument("--stage-iters", type=positive_int, default=24)
    p.add_argument("--max-iters", type=positive_int, default=3000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--threads", type=positive_int, default=1)
    p.add_argument("--cpus", type=parse_cpus)
    p.add_argument("--warmup-pairs", type=int, default=1)
    p.add_argument("--pairs", type=positive_int, default=9)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--timeout-seconds", type=positive_int, default=900)
    args = p.parse_args()
    if sys.platform != "linux" or not hasattr(os, "wait4") or not hasattr(os, "sched_setaffinity"):
        p.error("requires Linux/WSL2 wait4 and CPU affinity")
    if args.warmup_pairs < 0:
        p.error("warmup-pairs must be >= 0")
    if not math.isfinite(args.tol) or args.tol <= 0:
        p.error("tol must be finite and > 0")
    if not math.isfinite(args.monitor_rate_limit) or not 0 < args.monitor_rate_limit < 1:
        p.error("monitor-rate-limit must be finite in (0,1)")
    args.matrix = args.matrix.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    args.rhs = args.rhs.expanduser().resolve() if args.rhs else None
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.matrix.is_file() or not args.binary.is_file() or (args.rhs and not args.rhs.is_file()):
        p.error("matrix, binary and optional rhs must exist")
    if not os.access(args.binary, os.X_OK):
        p.error("binary is not executable")
    if args.output_dir.exists():
        p.error("output-dir must not already exist")
    allowed = os.sched_getaffinity(0)
    if args.cpus:
        if not set(args.cpus).issubset(allowed):
            p.error(f"CPU IDs not in allowed affinity {sorted(allowed)}")
        os.sched_setaffinity(0, args.cpus)
    command = [str(args.binary), "--matrix", str(args.matrix),
               "--tol", str(args.tol), "--max-iters", str(args.max_iters),
               "--probe-iters", str(args.probe_iters),
               "--monitor-iters", str(args.monitor_iters),
               "--monitor-rate-limit", str(args.monitor_rate_limit),
               "--max-escalations", str(args.max_escalations),
               "--max-region", str(args.max_region),
               "--max-regions", str(args.max_regions),
               "--factor-budget-mib", str(args.factor_budget_mib),
               "--stage-iters", str(args.stage_iters), "--backend", "csr",
               "--selector", "jacobi-byte", "--skip-plain"]
    if args.rhs:
        command += ["--rhs", str(args.rhs)]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), RAYON_NUM_THREADS=str(args.threads))
    args.output_dir.mkdir()
    logdir = args.output_dir / "logs"
    logdir.mkdir()
    settings = {"schema": "hybit-c1d-paired-crossover-v1",
                "matrix_sha256": hash_file(args.matrix),
                "rhs_sha256": hash_file(args.rhs) if args.rhs else None,
                "rhs_kind": "external" if args.rhs else "manufactured_A_times_ones",
                "spd_certification": "not_performed",
                "binary_sha256": hash_file(args.binary),
                "git_head": command_version(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
                "git_dirty": bool(command_version(["git", "-C", str(REPO), "status", "--porcelain"])),
                "host": platform.node(), "kernel": platform.release(),
                "rustc": command_version(["rustc", "--version"]),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "threads": args.threads, "warmup_pairs": args.warmup_pairs,
                "pairs": args.pairs, "seed": args.seed,
                "command": command,
                "arm_difference": "observe arm adds --monitor-only; hybrid arm permits opt-in escalation",
                "measurement": "fresh processes, alternating AB/BA, internal HyBIT total wall ms",
                "limitations": "one RHS, no prepared amortization, no SPD certification, WSL cache/thermal effects"}
    (args.output_dir / "metadata.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows = []
    rng = random.Random(args.seed)
    run_id = 0
    with (args.output_dir / "results.csv").open("w", newline="") as csv_file, \
         (args.output_dir / "results.jsonl").open("w") as json_file:
        writer = csv.DictWriter(csv_file, FIELDS)
        writer.writeheader()
        for phase, count in (("warmup", args.warmup_pairs), ("measure", args.pairs)):
            first = tuple(reversed(ARMS)) if rng.randrange(2) else ARMS
            for pair in range(1, count + 1):
                # Alternate AB/BA in adjacent pairs to counterbalance order.
                sequence = first if (pair - 1) % 2 == 0 else tuple(reversed(first))
                if pair % 2 == 0:
                    first = tuple(reversed(first)) if rng.randrange(2) else first
                for position, arm in enumerate(sequence, 1):
                    run_id += 1
                    label = f"{run_id:03d}_{phase}_pair{pair}_{arm}"
                    outlog = logdir / f"{label}.out"
                    errlog = logdir / f"{label}.err"
                    cmd = command + (["--monitor-only"] if arm == "observe" else [])
                    code, timed_out, wall, rss = run_process(
                        cmd, REPO, outlog, errlog, args.timeout_seconds, env)
                    parsed = parse_fem_output(outlog.read_text(errors="replace"))
                    parsed.update(exit_code=code, timed_out=timed_out)
                    numeric_gate(parsed, args.tol, False, args.factor_budget_mib)
                    result = dict.fromkeys(FIELDS)
                    result.update({key: parsed.get(key) for key in FIELDS})
                    result.update(run_id=run_id, phase=phase, pair=pair, position=position,
                                  arm=arm, exit_code=code, timed_out=timed_out,
                                  process_wall_ms=wall, peak_rss_kib=rss,
                                  **{"pass": parsed["hybit_pass"]},
                                  failure_reason=parsed["failure_reason"],
                                  stdout_log=str(outlog.relative_to(args.output_dir)),
                                  stderr_log=str(errlog.relative_to(args.output_dir)))
                    result = strict_json_row(result)
                    rows.append(result)
                    writer.writerow(result)
                    csv_file.flush()
                    json_file.write(json.dumps(result, allow_nan=False) + "\n")
                    json_file.flush()
                    print(f"{phase} pair={pair} arm={arm} pass={result['pass']} "
                          f"escalations={result['hybit_escalations']} wall_ms={result['hybit_wall_ms']}", flush=True)
    summary = paired_assessment(rows, seed=args.seed, expected_pairs=args.pairs)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    # No claims about superiority; fail only when any measured numerical gate fails.
    return 0 if all(row["pass"] for row in rows if row["phase"] == "measure") else 1


if __name__ == "__main__":
    raise SystemExit(main())
