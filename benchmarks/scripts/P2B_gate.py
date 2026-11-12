#!/usr/bin/env python3
"""P2-B: enforce paired-input and independent-residual evidence, no fabricated rescue.

Only a naturally observed OFF=not-converged -> ON=Converged transition is a
P2-A rescue. Ordinary MaxIterations are *not* interpreted as P2-A opportunities.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

STATUS_RE = re.compile(r"^status\s*:\s*(Converged|MaxIterations|Breakdown)\s*$", re.M)
ITER_RE = re.compile(r"^iterations\s*:\s*(\d+)\s*$", re.M)
VERIFIED_RE = re.compile(r"^verified residual\s*:\s*([+\-0-9.eE]+)\s*$", re.M)
PRECOND_RE = re.compile(r"^preconditioner\s*:\s*(\w+)\s*$", re.M)
ERROR_RE = re.compile(r"^relative x error\s*:\s*([+\-0-9.eE]+)\s*$", re.M)
TIME_RE = re.compile(r"elapsed_s=([+\-0-9.]+) peak_rss_kib=(\d+) exit_status=(\d+)")
CASE_RE = re.compile(r"^P2B_CASE\|(.+)$", re.M)


def unique(pattern, source, desc):
    matches = pattern.findall(source)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {desc}, found {len(matches)}")
    return matches[0]


def finite_float(value, desc):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{desc} must be finite and nonnegative: {value}")
    return number


def parse_fem(text, timing, max_iters, tol):
    status = unique(STATUS_RE, text, "HyBIT status")
    iterations = int(unique(ITER_RE, text, "HyBIT iteration count"))
    verified = finite_float(unique(VERIFIED_RE, text, "original CSR verification"), "verified residual")
    preconditioner = unique(PRECOND_RE, text, "preconditioner")
    forward = ERROR_RE.findall(text)
    if len(forward) > 1:
        raise ValueError("duplicate forward error")
    forward_error = finite_float(forward[0], "forward error") if forward else None
    if iterations > max_iters:
        raise ValueError("iteration cap exceeded")
    # fem_bench formats its verified relative residual with only six decimal digits.
    # Allow an explicit 0.01% presentation margin; never accept multi-order mismatches.
    if status == "Converged" and verified > tol * (1.0 + 1e-4):
        raise ValueError(f"FALSE CONVERGENCE: {verified} exceeds {tol}")
    elapsed, rss, exit_code = unique(TIME_RE, timing, "time record")
    if int(exit_code) != 0:
        raise ValueError(f"timed process exit_status={exit_code}")
    return dict(status=status, iterations=iterations, verified_residual=verified,
                preconditioner=preconditioner, forward_error=forward_error,
                elapsed_s=finite_float(elapsed, "elapsed_s"), peak_rss_kib=int(rss))


def parse_zero_probe(text, mode):
    records = {}
    for entry in CASE_RE.findall(text):
        fields = dict(kv.split("=", 1) for kv in entry.split("|") if "=" in kv)
        if set(fields) != {"name", "feature", "status", "iterations", "max_iters",
                           "true_norm", "target", "rhs_norm", "initial_norm", "x_error"}:
            raise ValueError(f"invalid P2B_CASE schema: {entry}")
        name = fields["name"]
        if fields["feature"] != mode or name in records:
            raise ValueError(f"duplicate/wrong feature in {name}")
        if fields["status"] not in ("Converged", "MaxIterations", "Breakdown"):
            raise ValueError("unknown zero-probe status")
        if int(fields["iterations"]) > int(fields["max_iters"]):
            raise ValueError("zero-probe iteration cap exceeded")
        for field in ("true_norm", "target", "rhs_norm", "initial_norm", "x_error"):
            fields[field] = finite_float(fields[field], field)
        if fields["status"] == "Converged" and fields["true_norm"] > fields["target"]:
            raise ValueError(f"FALSE ZERO-RHS CONVERGENCE: {name}")
        records[name] = fields
    required = {"zero_rhs_zero_x", "zero_rhs_nonzero_x_rel_only",
                "zero_rhs_nonzero_x_abs", "manufactured_nonzero_rhs"}
    if set(records) != required:
        raise ValueError(f"missing/extra zero-probe records: {sorted(set(records) ^ required)}")
    if records["zero_rhs_zero_x"]["status"] != "Converged":
        raise ValueError("zero RHS / zero x initial state did not converge")
    return records


def gate(directory: Path, tol: float, max_iters: int):
    raw = (directory / "cases_effective.tsv").read_text()
    cases = list(csv.reader(raw.splitlines(), delimiter="\t"))
    if not cases or any(len(row) != 4 for row in cases):
        raise ValueError("missing or malformed case manifest")
    labels = [row[0] for row in cases]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate FEM case labels")
    rows = []
    rescued = 0
    regressed = 0
    for name in labels:
        results = {}
        for mode in ("off", "on"):
            results[mode] = parse_fem(
                (directory / f"{name}_{mode}.log").read_text(),
                (directory / f"{name}_{mode}.time").read_text(),
                max_iters, tol,
            )
        off, on = results["off"], results["on"]
        rescue = off["status"] != "Converged" and on["status"] == "Converged"
        regression = off["status"] == "Converged" and on["status"] != "Converged"
        rescued += int(rescue)
        regressed += int(regression)
        for mode, result in results.items():
            rows.append(dict(case=name, mode=mode, **result,
                             natural_rescue=int(rescue), regression=int(regression)))
    zero = {mode: parse_zero_probe((directory / f"zero_{mode}.log").read_text(), mode)
            for mode in ("off", "on")}
    for name in zero["off"]:
        for key in ("rhs_norm", "initial_norm", "target"):
            if zero["off"][name][key] != zero["on"][name][key]:
                raise ValueError(f"zero-probe input disagreement: {name}.{key}")
    if regressed:
        raise ValueError(f"feature ON caused {regressed} certified-convergence regressions")
    output = directory / "p2b_comparison.csv"
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "case_count": len(labels), "run_count": len(rows),
        "natural_rescues": rescued, "certified_convergence_regressions": regressed,
        "feature_off_converged": sum(r["status"] == "Converged" for r in rows if r["mode"] == "off"),
        "feature_on_converged": sum(r["status"] == "Converged" for r in rows if r["mode"] == "on"),
        "zero_rhs_observations": {
            mode: {name: {"status": f["status"], "iterations": int(f["iterations"]),
                          "true_norm": f["true_norm"], "target": f["target"]}
                   for name, f in data.items()}
            for mode, data in zero.items()},
        "note": "No natural false convergence means no evidence P2-A improved this corpus."
    }
    (directory / "p2b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("case_count", "natural_rescues", "certified_convergence_regressions")}, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--max-iters", type=int, default=600)
    args = p.parse_args()
    if not 0 < args.tol < 1 or args.max_iters < 1:
        p.error("invalid tolerance/iterations")
    try:
        gate(args.run_dir, args.tol, args.max_iters)
    except (OSError, ValueError) as exc:
        print(f"P2-B EVIDENCE GATE FAIL: {exc}", file=sys.stderr)
        return 1
    print("P2-B EVIDENCE GATE PASS (numerical acceptance, not a rescue-performance claim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
