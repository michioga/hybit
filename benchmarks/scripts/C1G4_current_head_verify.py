#!/usr/bin/env python3
"""C1-G4: current-HEAD A9 native multilevel PCG full-RHS evidence validation.

Only the existing A9 original-CSR-certified output parser is reused. No claims
about SPD proof or physical RHS are inferred from a manufactured vector.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

CASES = ("island_shell", "multi_patch", "rotated_anisotropy")
ARMS = {"graph_piecewise": ("graph", "piecewise"),
        "strong_graph_piecewise": ("strong-graph", "piecewise"),
        "strong_graph_smoothed": ("strong-graph", "smoothed")}
MODES = {"ones": "ones", "index_smooth": "index-smooth",
         "seeded_random": "seeded-random", "point_load": "point-load"}
POLICIES = ("two_level", "multilevel")
FIELDS = ("case", "arm", "rhs_mode", "policy", "rhs_hash", "status",
          "iterations", "verified_true_relative", "relative_x_error", "levels",
          "coarse_factor_bytes", "scratch_bytes", "setup_ms", "solve_ms",
          "total_wall_ms", "elapsed_s", "peak_rss_kib")
EXPECTED_N = 3 * 3 * 4 * 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def matrix_hashes(corpus: Path) -> dict[str, str]:
    results = {}
    for case in CASES:
        path = corpus / f"c1g_p1_{case}_n144.mtx"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing or empty n144 matrix: {path}")
        results[case] = sha256(path)
    if len(set(results.values())) != len(results):
        raise ValueError("unexpected identical matrix hashes")
    return results


def expected_specs():
    for case in CASES:
        for arm in ARMS:
            for mode in MODES:
                for policy in POLICIES:
                    yield case, arm, mode, policy


def stem_for(spec: tuple[str, str, str, str]) -> str:
    return "_".join(spec)


def validate_pair(control: dict, candidate: dict) -> str:
    if control["rhs_hash"] != candidate["rhs_hash"]:
        raise ValueError("pair's RHS hash mismatch")
    for key in ("case", "arm", "rhs_mode"):
        if control[key] != candidate[key]:
            raise ValueError(f"pair identity mismatch {key}")
    if control["policy"] != "two_level" or candidate["policy"] != "multilevel":
        raise ValueError("invalid comparison policies")
    passed = lambda r: r["status"] == "Converged" and r["verified_true_relative"] <= 1.0e-10
    c, a = passed(control), passed(candidate)
    return ("BOTH_CONVERGED" if c and a else "RESCUE" if a else
            "REGRESSION" if c else "BOTH_NOT_CONVERGED")


def summarize(rows: list[dict]) -> dict:
    lookup = {(r["case"], r["arm"], r["rhs_mode"], r["policy"]): r for r in rows}
    if len(lookup) != EXPECTED_N:
        raise ValueError("missing/duplicate numerical grid")
    by_mode = {}
    overall = Counter()
    for mode in MODES:
        results = Counter()
        for case in CASES:
            for arm in ARMS:
                base = lookup[(case, arm, mode, "two_level")]
                multi = lookup[(case, arm, mode, "multilevel")]
                cls = validate_pair(base, multi)
                results[cls] += 1
                overall[cls] += 1
        by_mode[mode] = dict(sorted(results.items()))
    return {"pairs": EXPECTED_N // 2,
            "manufactured_ones_pairs": 9, "non_ones_pairs": 27,
            "outcomes_by_rhs": by_mode, "outcomes_total": dict(sorted(overall.items())),
            "interpretation": "A9 research-only, original-CSR certified, four RHS modes; "
            "point-load is an algebraic load, not verified physical provenance. "
            "One run per configuration; no unbiased runtime advantage claim."}


def normalized_record(spec: tuple[str, str, str, str], parsed: dict, timing: dict) -> dict:
    """Project the existing A9 parser schema onto C1-G4's comparison schema.

    A9's strict parser returns policy=TwoLevel/Multilevel and has no 'arm'
    field. The configuration identity is supplied by the validated log stem.
    """
    case, arm, mode, policy = spec
    native_policy = {"two_level": "TwoLevel", "multilevel": "Multilevel"}[policy]
    if (parsed["case"] != case or parsed["rhs_mode"] != mode
            or parsed["policy"] != native_policy):
        raise ValueError(f"A9 parsed identity mismatch: {stem_for(spec)}")
    source = dict(parsed)
    source["arm"] = arm
    source["policy"] = policy
    if not {"elapsed_s", "peak_rss_kib"} <= set(timing):
        raise ValueError("incomplete time record")
    return {key: (timing[key] if key in timing else source[key]) for key in FIELDS}


def audit(repo: Path, evidence: Path, *, check_only: bool = False) -> dict:
    sys.path.insert(0, str(repo))
    from benchmarks.scripts.C1H12_A9_rhs_gate import parse_a9  # noqa: PLC0415
    from benchmarks.scripts.C1H12_A4_numerical_gate import parse_time  # noqa: PLC0415
    invocation_file = evidence / "invocation.json"
    if not invocation_file.is_file():
        raise ValueError("missing invocation provenance")
    invocation = json.loads(invocation_file.read_text(encoding="utf-8"))
    if set(invocation) != {"head", "branch", "corpus", "matrix_sha256", "numeric_policy", "case_count"}:
        raise ValueError("unknown/missing invocation fields")
    if invocation["branch"] != "develop/0.10.0" or invocation["case_count"] != EXPECTED_N:
        raise ValueError("invalid branch or run count")
    head = invocation["head"]
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("bad git commit hash")
    import subprocess
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo).strip()
    if head != current_head:
        raise ValueError("head changed during evidence run")
    corpus = Path(invocation["corpus"])
    if not corpus.is_absolute() or invocation["matrix_sha256"] != matrix_hashes(corpus):
        raise ValueError("matrix hash mismatch")
    if invocation["numeric_policy"] != {"tol": 1e-10, "max_iterations": 600,
                                         "threads": 1, "rhs": list(MODES)}:
        raise ValueError("numerical policy changed")
    expected = {stem_for(spec) for spec in expected_specs()}
    for ext in ("log", "time"):
        observed = {p.stem for p in evidence.glob(f"*.{ext}")}
        if observed != expected:
            raise ValueError(f"{ext} grid mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    rows = []
    hashes_by_case_rhs = defaultdict(set)
    for spec in expected_specs():
        case, arm, mode, policy = spec
        stem = stem_for(spec)
        marker = (evidence / f"{stem}.log").read_text(encoding="utf-8")
        timing = parse_time((evidence / f"{stem}.time").read_text(encoding="utf-8"))
        if timing["exit_status"] != 0 or timing["peak_rss_kib"] <= 0:
            raise ValueError(f"failed process or missing RSS: {stem}")
        row = parse_a9(marker, case, arm, mode, policy)  # existing strict A9/A8 gates
        # Independent strict rejection at the declared tolerance (no rounding allowance).
        true_r = row["verified_true_relative"]
        if row["status"] == "Converged" and true_r > 1e-10:
            raise ValueError(f"false convergence: {stem}")
        if row["status"] != "Converged" and true_r <= 1e-10:
            raise ValueError(f"status/residual mismatch: {stem}")
        if mode != "point_load" and not math.isfinite(float(row["relative_x_error"])):
            raise ValueError(f"bad forward error: {stem}")
        hashes_by_case_rhs[(case, mode)].add(row["rhs_hash"])
        rows.append(normalized_record(spec, row, timing))
    if len(rows) != EXPECTED_N or any(len(v) != 1 for v in hashes_by_case_rhs.values()):
        raise ValueError("wrong row count or RHS mismatch across arms")
    for case in CASES:
        if len({next(iter(hashes_by_case_rhs[(case, m)])) for m in MODES}) != len(MODES):
            raise ValueError(f"RHS modes were not distinct for {case}")
    summary = summarize(rows)
    summary.update({"status": "PASS_ORIGINAL_CSR_AND_GRID", "head": head,
                    "matrix_sha256": invocation["matrix_sha256"],
                    "runs": len(rows), "numerical_policy": invocation["numeric_policy"],
                    "converged_by_policy": {
                        p: sum(r["status"] == "Converged" for r in rows if r["policy"] == p)
                        for p in POLICIES},
                    "mean_peak_rss_kib_by_policy": {
                        p: sum(r["peak_rss_kib"] for r in rows if r["policy"] == p) /
                           (len(rows) // 2) for p in POLICIES}})
    csv_path = evidence / "c1g4_current_head_comparison.csv"
    summary_path = evidence / "c1g4_current_head_summary.json"
    if not check_only and (csv_path.exists() or summary_path.exists()):
        raise ValueError("analysis files already exist; no overwrite")
    if not check_only:
        with csv_path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="validate all 72 frozen logs without writing analysis files")
    a = ap.parse_args()
    try:
        summary = audit(a.repo.resolve(), a.evidence.resolve(), check_only=a.check_only)
        print(json.dumps(summary, indent=2))
        label = "CHECK-ONLY" if a.check_only else "ANALYSIS"
        print(f"C1-G4 CURRENT-HEAD MULTILEVEL/A9 {label} NUMERICAL GATE PASS")
        return 0
    except (ValueError, OSError, KeyError, ImportError, UnicodeError, subprocess.CalledProcessError) as e:
        print(f"C1-G4 FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
