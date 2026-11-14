#!/usr/bin/env python3
"""C1-G5B: fail-closed matched Jacobi/Hybrid measurements (no production routing).

This *never* promotes A9 AMG or Hybrid based on a one-run pilot. The frozen
A9 RHS fingerprint is the authority for exact paired-input identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

CASES = ("island_shell", "multi_patch", "rotated_anisotropy")
RHS = ("ones", "index_smooth", "seeded_random", "point_load")
ARMS = ("jacobi_pcg", "hybrid_local_cholesky")
HEAD = "f121d6327414dad8830c50c770fe33650a932c5d"
SOURCE = "benchmarks/evidence/C1G4_AMG_20261113_comparison.csv"
SOURCE_SUMMARY = "benchmarks/evidence/C1G4_AMG_20261113_summary.json"
LOG_FIELDS = ("case", "arm", "rhs_mode", "rhs_hash", "status", "iterations", "tol",
              "max_iterations", "reported_relative", "verified_true_relative",
              "compensated_true_relative", "relative_x_error", "escalations",
              "local_regions", "local_factor_bytes", "effective_preconditioner",
              "rows", "nnz", "load_ms", "rhs_ms", "setup_ms", "solve_ms",
              "arm_work_ms", "inprocess_total_wall_ms")
CSV_FIELDS = (*LOG_FIELDS, "elapsed_s", "peak_rss_kib", "exit_status", "matrix_sha256",
              "source_head", "research_scope")
TIME_FIELDS = ("elapsed_s", "peak_rss_kib", "exit_status")
CSV_NAME = "c1g5b_matched_results.csv"
JSON_NAME = "c1g5b_matched_summary.json"
HASH_NAME = "c1g5b_raw_sha256.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numbers(s: str, field: str, min_: float = 0.0, positive: bool = False) -> float:
    x = float(s)
    if not math.isfinite(x) or x < min_ or (positive and x <= 0):
        raise ValueError(f"invalid {field}: {s}")
    return x


def integer(s: str, field: str, minimum: int = 0) -> int:
    value = int(s)
    if str(value) != s or value < minimum:
        raise ValueError(f"invalid integer {field}: {s}")
    return value


def csv_bytes(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise ValueError("internal CSV schema mismatch")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def reference(repo: Path) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    frozen = json.loads((repo / SOURCE_SUMMARY).read_text(encoding="utf-8"))
    if frozen.get("source_head") != "37bc52dfb265123267081afc58bd2e6e7b0b6c97":
        raise ValueError("wrong C1-G4 source head")
    if frozen.get("numerical_gate") != "PASS_ORIGINAL_CSR_AND_GRID" or frozen.get("runs") != 72:
        raise ValueError("C1-G4 gate not accepted")
    p = repo / SOURCE
    with p.open(encoding="utf-8", newline="") as f:
        lines = list(csv.DictReader(f))
    if len(lines) != 72:
        raise ValueError("wrong C1-G4 grid")
    hashes = {}
    for row in lines:
        key = (row["case"], row["rhs_mode"])
        if row["case"] not in CASES or row["rhs_mode"] not in RHS:
            raise ValueError("unexpected C1-G4 matrix/RHS")
        if key in hashes and hashes[key] != row["rhs_hash"]:
            raise ValueError("A9 RHS mismatches across arms")
        hashes[key] = row["rhs_hash"]
    if len(hashes) != 12:
        raise ValueError("incomplete C1-G4 RHS hash grid")
    matrices = frozen["matrix_sha256"]
    if set(matrices) != set(CASES):
        raise ValueError("incomplete C1-G4 matrix hash grid")
    return hashes, matrices


def parse_log(log: str, case: str, rhs: str, arm: str, expected_hash: str) -> dict:
    matches = [line for line in log.splitlines() if line.startswith("C1G5B|")]
    if len(matches) != 1:
        raise ValueError("expected exactly one structured C1G5B result")
    parts = matches[0].split("|")[1:]
    if any("=" not in p for p in parts):
        raise ValueError("malformed structured field")
    pairs = [p.split("=", 1) for p in parts]
    d = dict(pairs)
    if len(d) != len(pairs) or set(d) != set(LOG_FIELDS):
        raise ValueError("unknown, duplicate or missing structured field")
    if (d["case"], d["rhs_mode"], d["arm"], d["rhs_hash"]) != (case, rhs, arm, expected_hash):
        raise ValueError("matrix/arm/RHS-fingerprint mismatch")
    if d["status"] not in ("Converged", "MaxIterations", "Breakdown"):
        raise ValueError("unrecognized solve status")
    it = integer(d["iterations"], "iterations")
    if it > 600 or d["tol"] != "1.000000000000e-10" or d["max_iterations"] != "600":
        raise ValueError("numerical budget changed")
    if d["status"] == "MaxIterations" and it != 600:
        raise ValueError("MaxIterations must exhaust common budget")
    for k in ("reported_relative", "verified_true_relative", "compensated_true_relative"):
        numbers(d[k], k)
    if d["status"] == "Converged" and (float(d["verified_true_relative"]) > 1e-10
                                              or float(d["compensated_true_relative"]) > 1e-10):
        raise ValueError("false convergence against original CSR")
    if rhs == "point_load":
        if d["relative_x_error"] != "n/a":
            raise ValueError("point-load forward-error is unavailable")
    else:
        numbers(d["relative_x_error"], "forward error")
    if integer(d["rows"], "rows") != 20736:
        raise ValueError("wrong matrix dimension")
    if integer(d["nnz"], "nnz") != (144002 if case == "rotated_anisotropy" else 103104):
        raise ValueError("wrong matrix NNZ")
    escalations = integer(d["escalations"], "escalations")
    regions = integer(d["local_regions"], "local regions")
    factor = integer(d["local_factor_bytes"], "factor bytes")
    if escalations > 1 or regions > 8 or factor > 64 * 1024 * 1024:
        raise ValueError("out-of-budget Hybrid")
    if arm == "jacobi_pcg":
        if (escalations, regions, factor, d["effective_preconditioner"]) != (0, 0, 0, "Jacobi"):
            raise ValueError("Jacobi arm unexpectedly changed")
    elif regions == 0:
        if escalations != 0 or factor != 0 or d["effective_preconditioner"] != "JacobiOnly":
            raise ValueError("nonactivated Hybrid mislabeled")
    elif escalations != 1 or factor <= 0 or d["effective_preconditioner"] != "Hybrid":
        raise ValueError("invalid local Cholesky activation")
    for k in ("load_ms", "rhs_ms", "setup_ms", "solve_ms", "arm_work_ms",
              "inprocess_total_wall_ms"):
        numbers(d[k], k, positive=True)
    if (float(d["inprocess_total_wall_ms"]) + 1e-3 < float(d["arm_work_ms"])
            + float(d["load_ms"]) + float(d["rhs_ms"])):
        raise ValueError("inconsistent phase wall time")
    return d


def parse_time(s: str) -> dict:
    values = {}
    for piece in s.split():
        if "=" not in piece:
            raise ValueError("unexpected GNU time token")
        k, v = piece.split("=", 1)
        if k in values:
            raise ValueError("duplicate GNU time field")
        values[k] = v
    if set(values) != set(TIME_FIELDS):
        raise ValueError("GNU time fields absent")
    numbers(values["elapsed_s"], "elapsed_s")
    integer(values["peak_rss_kib"], "peak RSS", 1)
    if values["exit_status"] != "0":
        raise ValueError("nonzero process status")
    return values


def audit(repo: Path, evidence: Path) -> tuple[list[dict], dict, bytes]:
    hashes, matrices = reference(repo)
    info = (evidence / "HEAD.txt").read_text(encoding="utf-8").strip()
    if info != HEAD:
        raise ValueError("evidence source HEAD mismatch")
    source_info = (evidence / "SOURCE.sha256").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", source_info):
        raise ValueError("missing Rust-example source hash")
    source = repo / "crates/hybit/examples/c1g5b_jacobi_hybrid_probe.rs"
    if sha256(source) != source_info:
        raise ValueError("Rust-example source hash changed")
    output = []
    raw = []
    for case in CASES:
        for rhs in RHS:
            for arm in ARMS:
                stem = f"{case}_{rhs}_{arm}"
                logfile, timefile = evidence / (stem + ".log"), evidence / (stem + ".time")
                parsed = parse_log(logfile.read_text(encoding="utf-8"), case, rhs, arm, hashes[case, rhs])
                timed = parse_time(timefile.read_text(encoding="utf-8"))
                output.append({**parsed, **timed, "matrix_sha256": matrices[case],
                               "source_head": HEAD, "research_scope": "G5B_PAIRED_CONTROL_ONLY"})
                for path in (logfile, timefile):
                    raw.append({"file": path.name, "sha256": sha256(path)})
    counts = Counter()
    examples = []
    for i in range(0, len(output), 2):
        j, h = output[i], output[i + 1]
        valid_j = j["status"] == "Converged"
        valid_h = h["status"] == "Converged"
        used_factor = int(h["local_regions"]) > 0 and int(h["local_factor_bytes"]) > 0
        if valid_h and not valid_j and used_factor:
            outcome = "LOCAL_RESCUE"
        elif valid_h and not valid_j:
            outcome = "NO_FACTOR_NONLOCAL_DIFFERENCE"
        elif valid_j and not valid_h:
            outcome = "HYBRID_REGRESSION"
        elif valid_j and valid_h:
            outcome = "BOTH_CONVERGED"
        else:
            outcome = "BOTH_NOT_CONVERGED"
        counts[outcome] += 1
        examples.append({"case": j["case"], "rhs_mode": j["rhs_mode"],
                         "outcome": outcome, "hybrid_local_factor_active": used_factor,
                         "jacobi_relative": j["verified_true_relative"],
                         "hybrid_relative": h["verified_true_relative"],
                         "jacobi_work_ms": j["arm_work_ms"],
                         "hybrid_work_ms": h["arm_work_ms"]})
    result = {
        "checkpoint": "C1-G5B matched Jacobi/monitored-Hybrid pilot",
        "source_head": HEAD,
        "runs": len(output),
        "paired_problems": len(examples),
        "matrix_family_count": 1,
        "numerical_contract": {"tol": 1e-10, "max_iterations": 600, "threads": 1,
                               "rhs_hash": "frozen A9 exact bitwise match"},
        "outcomes": dict(sorted(counts.items())),
        "pairs": examples,
        "limits": "Standalone paired pilot; all RHS algebraic/manufactured, no external held-out FEM."
                  " A9 time and G5B time were measured in separate experiment code paths;"
                  " do not merge timing into a production oracle or claim speedup."
                  " Some monitored-Hybrid runs do not activate local factors.",
    }
    return output, result, csv_bytes(raw, ("file", "sha256"))


def write_once(path: Path, payload: bytes) -> None:
    with path.open("xb") as f:
        f.write(payload)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--evidence", required=True, type=Path)
    p.add_argument("--write", action="store_true")
    p.add_argument("--gate", action="store_true")
    args = p.parse_args()
    repo, evidence = args.repo.resolve(), args.evidence.resolve()
    try:
        records, summary, raw_sha = audit(repo, evidence)
        blobs = {
            CSV_NAME: csv_bytes(records, CSV_FIELDS),
            JSON_NAME: (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(),
            HASH_NAME: raw_sha,
        }
        if args.write and args.gate:
            raise ValueError("--write and --gate are mutually exclusive")
        if args.write:
            if any((evidence / name).exists() for name in blobs):
                raise ValueError("refusing to overwrite existing summary artifact")
            for name, payload in blobs.items():
                write_once(evidence / name, payload)
        if args.gate:
            for name, payload in blobs.items():
                if (evidence / name).read_bytes() != payload:
                    raise ValueError(f"frozen output changed: {name}")
        print("C1-G5B EVIDENCE PASS: 24 records, 12 matched pairs, exact RHS fingerprints")
        print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, UnicodeError, TypeError) as exc:
        print(f"C1-G5B EVIDENCE FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
