#!/usr/bin/env python3
"""Read-only preflight and explicit intake of C1-H12-A2/A3/A4/A5 WSL raw evidence.

The A4 six false convergences are *expected negative evidence* and must
remain visible. Never convert those A4 records into an accepted numerical gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
import subprocess
import sys
from pathlib import Path

FULL_HEAD = "4593c2010722876b2b66bda83b78b62827ace77a"
CASES = ("island_shell", "multi_patch", "rotated_anisotropy")
NNZ = {"island_shell": 103104, "multi_patch": 103104, "rotated_anisotropy": 144002}
FALSE_A4 = {f"multi_patch_{arm}" for arm in (
    "contiguous_piecewise", "contiguous_smoothed", "graph_piecewise",
    "graph_smoothed", "strong_graph_piecewise", "strong_graph_smoothed",
)}
NEW_PATHS = (
    "benchmarks/scripts/C1H12_A2_A5_evidence_accept.py",
    "benchmarks/scripts/test_C1H12_A2_A5_evidence_accept.py",
    "docs/C1H12_A2_A5_ACCEPTED_EVIDENCE.md",
    "benchmarks/evidence/C1H12_A2_A5_symbolic.csv",
    "benchmarks/evidence/C1H12_A2_A5_native_pcg.csv",
    "benchmarks/evidence/C1H12_A2_A5_raw_sha256.csv",
)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source_integrity(repo: Path) -> list[str]:
    manifest = (repo / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    expected = sorted(set(manifest))
    if expected != manifest or len(expected) != len(manifest):
        raise ValueError("MANIFEST is unsorted or contains duplicates")
    for name in manifest:
        if name.startswith("/") or ".." in Path(name).parts or not (repo / name).is_file():
            raise ValueError(f"invalid MANIFEST entry {name!r}")
    actual_hashes: dict[str, str] = {}
    for line in (repo / "SOURCE_SHA256.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in actual_hashes:
            raise ValueError("invalid SOURCE_SHA256 entry")
        actual_hashes[match[2]] = match[1]
    paths = set(manifest) - {"MANIFEST.txt", "SOURCE_SHA256.txt"}
    if set(actual_hashes) != paths:
        raise ValueError("SOURCE_SHA256 paths do not match MANIFEST")
    for name, digest in actual_hashes.items():
        if hash_file(repo / name) != digest:
            raise ValueError(f"source SHA-256 mismatch: {name}")
    return manifest


def validate_provenance(directories: dict[str, Path], current_head: str) -> tuple[str, list[str]]:
    baselines: list[tuple[str, str]] = []
    entries: list[str] = []
    for stage, directory in directories.items():
        if not directory.is_dir():
            raise ValueError(f"missing {stage} evidence directory: {directory}")
        head_path = directory / "HEAD.txt"
        matrix_path = directory / "matrix_sha256.txt"
        head = head_path.read_text(encoding="ascii").strip()
        hashes = matrix_path.read_text(encoding="ascii")
        if head != current_head or not re.fullmatch(r"[0-9a-f]{40}", head):
            raise ValueError(f"{stage} provenance HEAD mismatch")
        baselines.append((head, hashes))
    if len(set(baselines)) != 1:
        raise ValueError("A2-A5 matrix input hash manifests do not agree byte-for-byte")
    seen_cases: set[str] = set()
    for line in baselines[0][1].splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if not m:
            raise ValueError("matrix hash manifest requires absolute paths and SHA-256")
        case = next((case for case in CASES if m[2].endswith(f"/c1g_p1_{case}_n144.mtx")), None)
        if case is None or case in seen_cases:
            raise ValueError("unexpected/duplicated matrix in input hash manifest")
        seen_cases.add(case)
        if hash_file(Path(m[2])) != m[1]:
            raise ValueError(f"matrix checksum mismatch: {case}")
        entries.append(f"{case}: `{m[1]}`")
    if seen_cases != set(CASES):
        raise ValueError("not all three matrix input hashes are present")
    return sha256_bytes(baselines[0][1].encode("ascii")), entries


def parse_unique_line(path: Path, prefix: str) -> dict[str, str]:
    matches = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix + "|")]
    if len(matches) != 1:
        raise ValueError(f"{path.name}: expected exactly one {prefix} record")
    parsed: dict[str, str] = {}
    for token in matches[0].split("|")[1:]:
        if "=" not in token:
            raise ValueError(f"{path.name}: malformed record token")
        key, value = token.split("=", 1)
        if not key or not value or key in parsed:
            raise ValueError(f"{path.name}: empty/duplicated record field")
        parsed[key] = value
    return parsed


def finite_number(row: dict[str, str], key: str, *, integer: bool = False) -> float | int:
    value: float | int = int(row[key]) if integer else float(row[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"nonfinite/negative {key}")
    return value


def symbolic_record(stage: str, case: str, agg: int | None, raw: dict[str, str]) -> dict[str, object]:
    is_a2 = stage == "A2"
    required = (
        {"rows", "nnz", "threshold", "strong_edges", "symbolic_nnz", "csr_graph_bytes",
         "abtm_metadata_bytes", "abtm_nonempty_words", "strength_ms", "graph_csr_prepare_ms",
         "abtm_conversion_ms", "csr_symbolic_ms", "abtm_symbolic_ms", "repeats", "symbolic_hash", "exact_equal"}
        if is_a2 else
        {"rows", "nnz", "aggregate_nodes", "coarse_n", "fine_pattern_edges", "symbolic_nnz",
         "numeric_nnz", "exact_zero_candidates", "csr_adjacency_bytes", "temporary_csr_logical_bytes",
         "abtm_metadata_bytes", "abtm_nonempty_words", "graph_build_ms", "graph_csr_prepare_ms",
         "abtm_conversion_ms", "numeric_reference_ms", "csr_sort_ms", "csr_stamp_ms",
         "abtm_symbolic_ms", "repeats", "symbolic_hash", "exact_equal"}
    )
    if set(raw) != required:
        raise ValueError(f"{stage}/{case}: unexpected symbolic field schema: {set(raw) ^ required}")
    if raw["exact_equal"] != "true" or finite_number(raw, "repeats", integer=True) != 5:
        raise ValueError(f"{stage}/{case}: failed exact-equality or repeat gate")
    if finite_number(raw, "rows", integer=True) != 20736 or finite_number(raw, "nnz", integer=True) != NNZ[case]:
        raise ValueError(f"{stage}/{case}: wrong matrix shape")
    if is_a2 and float(raw["threshold"]) != 0.25:
        raise ValueError("A2 strength threshold changed")
    if not is_a2 and finite_number(raw, "aggregate_nodes", integer=True) != agg:
        raise ValueError("A3 aggregate configuration changed")
    if not re.fullmatch(r"[0-9a-f]{16}", raw["symbolic_hash"]):
        raise ValueError("invalid symbolic pattern hash")
    for key in required - {"symbolic_hash", "exact_equal", "threshold"}:
        finite_number(raw, key, integer=(key.endswith("_bytes") or key.endswith("_nnz") or key in {
            "rows", "nnz", "repeats", "strong_edges", "symbolic_nnz", "abtm_nonempty_words",
            "aggregate_nodes", "coarse_n", "fine_pattern_edges", "numeric_nnz", "exact_zero_candidates",
        }))
    if not is_a2 and int(raw["numeric_nnz"]) > int(raw["symbolic_nnz"]):
        raise ValueError("numeric Galerkin pattern exceeds Boolean support")
    return {
        "stage": stage, "case": case, "aggregate_nodes": "" if is_a2 else agg,
        "rows": int(raw["rows"]), "nnz": int(raw["nnz"]),
        "symbolic_nnz": int(raw["symbolic_nnz"]),
        "numeric_nnz": "" if is_a2 else int(raw["numeric_nnz"]),
        "csr_kernel_ms": float(raw["csr_symbolic_ms"] if is_a2 else raw["csr_stamp_ms"]),
        "abtm_kernel_ms": float(raw["abtm_symbolic_ms"]),
        "csr_prepare_ms": float(raw["graph_csr_prepare_ms"]),
        "abtm_conversion_ms": float(raw["abtm_conversion_ms"]),
        "abtm_metadata_bytes": int(raw["abtm_metadata_bytes"]),
        "symbolic_hash": raw["symbolic_hash"], "exact_equal": True,
    }


def to_csv(records: list[dict[str, object]], columns: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def evidence_rows(directories: dict[str, Path]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]], set[str]]:
    from benchmarks.scripts.C1H12_A4_numerical_gate import ARMS, parse_record, parse_time, validate_record
    from benchmarks.scripts.C1H12_A5_certification_gate import parse_a5_record

    symbolic: list[dict[str, object]] = []
    pcg: list[dict[str, object]] = []
    digests: list[dict[str, str]] = []
    observed_false: set[str] = set()

    for stage, directory in directories.items():
        stems = ([case for case in CASES] if stage == "A2" else
                 [f"{case}_a{agg}" for case in CASES for agg in (4, 16, 40)] if stage == "A3" else
                 [f"{case}_{arm}_1" for case in CASES for arm in ARMS])
        for ext in ("log", "time"):
            actual = {p.name for p in directory.glob(f"*.{ext}")}
            expected = {f"{stem}.{ext}" for stem in stems}
            if actual != expected:
                raise ValueError(f"{stage} {ext} file mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
        for stem in stems:
            log = directory / f"{stem}.log"
            timing = directory / f"{stem}.time"
            time = parse_time(timing.read_text(encoding="utf-8"))
            if time["exit_status"] != 0:
                raise ValueError(f"{stage}/{stem}: process did not exit successfully")
            for raw_file in (log, timing):
                digests.append({"stage": stage, "file": raw_file.name, "sha256": hash_file(raw_file)})
            if stage in ("A2", "A3"):
                case = stem if stage == "A2" else stem.rsplit("_a", 1)[0]
                agg = None if stage == "A2" else int(stem.rsplit("_a", 1)[1])
                symbolic.append(symbolic_record(stage, case, agg, parse_unique_line(log, f"C1H12{stage}")))
                continue
            case, arm, _repeat = next((case, arm, 1) for case in CASES for arm in ARMS if stem == f"{case}_{arm}_1")
            raw_text = log.read_text(encoding="utf-8")
            raw = parse_record(raw_text)
            false_id = f"{case}_{arm}"
            if stage == "A4" and false_id in FALSE_A4:
                if raw["status"] != "Converged" or float(raw["verified_true_relative"]) <= 1.001e-10:
                    raise ValueError(f"A4 missing recorded false convergence: {stem}")
                # Preserve the known false residual; delegate all other field
                # schema and configuration checks to the existing A4 validator.
                non_residual = dict(raw, verified_true_relative="9.5e-11")
                validate_record(non_residual, case, arm)
                observed_false.add(false_id)
            else:
                validate_record(raw, case, arm)
            if stage == "A5" and raw["status"] == "Converged" and float(raw["verified_true_relative"]) > 1e-10:
                raise ValueError(f"A5 true residual exceeds strict 1e-10 acceptance: {stem}")
            extra = {}
            if stage == "A5":
                cert = parse_a5_record(raw_text)
                if (cert["status"] != raw["status"] or
                    cert["global_iterations"] != int(raw["iterations"]) or
                    not math.isclose(float(cert["final_true_relative"]), float(raw["verified_true_relative"]), rel_tol=1e-9, abs_tol=1e-14)):
                    raise ValueError(f"A5 telemetry mismatch: {stem}")
                extra = {
                    "true_checks": int(cert["true_checks"]),
                    "restarts": int(cert["restarts"]),
                    "rejected_recursive_convergences": int(cert["rejected_recursive_convergences"]),
                }
            pcg.append({
                "stage": stage, "case": case, "arm": arm, "status": raw["status"],
                "iterations": int(raw["iterations"]),
                "reported_relative": float(raw["reported_relative"]),
                "verified_true_relative": float(raw["verified_true_relative"]),
                "setup_ms": float(raw["setup_ms"]), "solve_ms": float(raw["solve_ms"]),
                "coarse_factor_bytes": int(raw["coarse_factor_bytes"]),
                "true_checks": extra.get("true_checks", ""),
                "restarts": extra.get("restarts", ""),
                "rejected_recursive_convergences": extra.get("rejected_recursive_convergences", ""),
            })
    if observed_false != FALSE_A4:
        raise ValueError("A4 six-false-convergence evidence changed")
    old = {(r["case"], r["arm"]): r for r in pcg if r["stage"] == "A4"}
    changed = {(r["case"], r["arm"]): r for r in pcg if r["stage"] == "A5"}
    if old.keys() != changed.keys() or len(old) != 21:
        raise ValueError("A4/A5 arm pairing mismatch")
    for key, before in old.items():
        after = changed[key]
        if key[0] == "multi_patch" and key[1] != "jacobi":
            if (after["status"] != "Converged" or
                after["verified_true_relative"] > 1e-10 or
                after["restarts"] < 1 or
                after["iterations"] <= before["iterations"]):
                raise ValueError(f"A5 did not repair A4 false convergence: {key}")
        if before["status"] == "MaxIterations" and after["status"] != "MaxIterations":
            raise ValueError(f"A5 changed negative-control status: {key}")
    if sum(r["status"] == "Converged" for r in changed.values()) != 18:
        raise ValueError("A5 convergence count differs from owner evidence")
    return symbolic, pcg, digests, observed_false


def document(directories: dict[str, Path], matrix_manifest_hash: str,
             matrix_entries: list[str], symbolic: list[dict[str, object]],
             pcg: list[dict[str, object]], digests: list[dict[str, str]]) -> str:
    lines = [
        "# C1-H12-A2–A5 WSL evidence checkpoint (2026-11-09)", "",
        "Acceptance: **54/54 raw cases validated** (A2 3, A3 9, A4 21, A5 21).",
        "Original study source: `develop/0.10.0` full HEAD `" + FULL_HEAD + "`.",
        "Owner environment: Ubuntu 26.04 WSL2, Rust +1.99.1, one process/thread.",
        "Evidence accepted by `benchmarks/scripts/C1H12_A2_A5_evidence_accept.py`.", "",
        "## Input provenance", "",
        "Identical `HEAD.txt` and `matrix_sha256.txt` across all four raw runs;",
        "matrix files were rehashed at acceptance. Matrix manifest SHA-256: `" + matrix_manifest_hash + "`.", "",
        *["- " + entry for entry in matrix_entries], "",
        "## Evidence locations (raw files remain owner-side)", "",
        *[f"- {stage}: `{path}`" for stage, path in directories.items()], "",
        "The repository retains per-record summaries and SHA-256 hashes of all " + str(len(digests)) + " `.log`/`.time` files.",
        "Those hashes are references to owner-side raw evidence, **not** the raw logs themselves.", "",
        "## Symbolic structure (A2/A3)", "",
        "- A2: three exact CSR/ABTM two-hop support comparisons (not a Galerkin operator).",
        "- A3: nine exact one-hot piecewise-constant `P^T A P` Boolean candidate comparisons;",
        "  independently assembled numeric reference. ABTM construction/conversion cost is separate.",
        "- Do not generalize isolated kernel wins into AMG or whole-solve speedups.", "",
        "## Native two-level PCG (A4/A5)", "",
        "- A4 numerical acceptance was **15/21**, not 21/21: six `multi_patch`",
        "  coarse arms falsely declared recursive `Converged` with true residual above 1e-10.",
        "- A5: **21/21 independent true-residual and provenance gate PASS**;",
        "  18 true `Converged`, 3 honest `MaxIterations`, no Breakdown.",
        "- All six `multi_patch` coarse arms were repaired using opt-in certified restarts",
        "  with the unchanged total 600-iteration budget; native PCG defaults are untouched.",
        "- No multilevel AMG, PETSc/hypre linkage, automatic routing, or whole-solve speedup claim.", "",
        "| multi_patch arm | A4 iterations | A4 true rel | A5 iterations | A5 true rel | certified restarts |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    a4 = {r["arm"]: r for r in pcg if r["stage"] == "A4" and r["case"] == "multi_patch"}
    a5 = {r["arm"]: r for r in pcg if r["stage"] == "A5" and r["case"] == "multi_patch"}
    for arm in sorted(a4):
        before, after = a4[arm], a5[arm]
        lines.append(
            f"| {arm} | {before['iterations']} | {before['verified_true_relative']:.6e} | "
            f"{after['iterations']} | {after['verified_true_relative']:.6e} | {after['restarts']} |"
        )
    lines.extend([
        "", "## Machine-readable accepted artifacts", "",
        "- `benchmarks/evidence/C1H12_A2_A5_symbolic.csv`: 12 isolated symbolic cases.",
        "- `benchmarks/evidence/C1H12_A2_A5_native_pcg.csv`: 42 A4/A5 arms.",
        "- `benchmarks/evidence/C1H12_A2_A5_raw_sha256.csv`: 108 raw `.log`/`.time` hashes.",
        "", "## Next prioritized experiment", "",
        "A6: instrument periodic true-residual checks and recurrence drift in an opt-in",
        "PCG experiment; evaluate early safe restarts versus A5 under matched conditions.",
        "Preserve SPD invariants, original `b-A*x` certification, 600 global iterations",
        "and independent A5 negative controls. Never mutate production PCG by default.", "",
    ])
    return "\n".join(lines)


def prepare_outputs(repo: Path, package_script: Path, directories: dict[str, Path]) -> dict[str, bytes]:
    manifest = check_source_integrity(repo)
    if any((repo / name).exists() for name in NEW_PATHS):
        raise ValueError("evidence artifact already exists; refusing to overwrite")
    source_script = package_script.read_bytes()
    source_test = (package_script.parent / "test_C1H12_A2_A5_evidence_accept.py").read_bytes()
    matrix_hash, matrix_entries = validate_provenance(directories, run_git(repo, "rev-parse", "HEAD"))
    symbolic, pcg, digests, false_cases = evidence_rows(directories)
    if (len(symbolic), len(pcg), len(digests), len(false_cases)) != (12, 42, 108, 6):
        raise ValueError("raw case-count gate failed")

    symbolic_cols = ["stage", "case", "aggregate_nodes", "rows", "nnz", "symbolic_nnz",
                     "numeric_nnz", "csr_kernel_ms", "abtm_kernel_ms", "csr_prepare_ms",
                     "abtm_conversion_ms", "abtm_metadata_bytes", "symbolic_hash", "exact_equal"]
    pcg_cols = ["stage", "case", "arm", "status", "iterations", "reported_relative",
                "verified_true_relative", "setup_ms", "solve_ms", "coarse_factor_bytes",
                "true_checks", "restarts", "rejected_recursive_convergences"]
    outputs: dict[str, bytes] = {
        NEW_PATHS[0]: source_script,
        NEW_PATHS[1]: source_test,
        NEW_PATHS[2]: document(directories, matrix_hash, matrix_entries, symbolic, pcg, digests).encode("utf-8"),
        NEW_PATHS[3]: to_csv(symbolic, symbolic_cols).encode("utf-8"),
        NEW_PATHS[4]: to_csv(pcg, pcg_cols).encode("utf-8"),
        NEW_PATHS[5]: to_csv(digests, ["stage", "file", "sha256"]).encode("utf-8"),
    }
    session = "\n## 2026-11-09 — C1-H12-A2–A5 owner WSL evidence accepted\n\n"
    session += "- A2 3/3, A3 9/9 exact symbolic; A4 15/21 numerical gate (six false Converged), A5 21/21\n"
    session += "  true-residual certified/log/provenance PASS: 18 Converged, 3 MaxIterations.\n"
    session += "- Owner raw logs: " + "; ".join(f"{stage} `{path}`" for stage, path in directories.items()) + ".\n"
    session += "- Raw log/time SHA-256 manifest and machine-readable 54 records are recorded under\n"
    session += "  `benchmarks/evidence/`; source HEAD `" + FULL_HEAD + "`.\n"
    session += "- No native multilevel AMG, whole-solve speedup or new production PCG routing claim.\n"
    session += "- Next one action: A6 experimental periodic true-residual/drift pilot before production changes.\n"
    outputs["docs/SESSION_HANDOFF.md"] = (repo / "docs/SESSION_HANDOFF.md").read_bytes() + session.encode("utf-8")
    updated_manifest = sorted(set(manifest) | set(NEW_PATHS))
    outputs["MANIFEST.txt"] = ("\n".join(updated_manifest) + "\n").encode("utf-8")
    hashes: list[str] = []
    for name in updated_manifest:
        if name in {"MANIFEST.txt", "SOURCE_SHA256.txt"}:
            continue
        content_digest = sha256_bytes(outputs[name]) if name in outputs else hash_file(repo / name)
        hashes.append(f"{content_digest}  {name}")
    outputs["SOURCE_SHA256.txt"] = ("\n".join(hashes) + "\n").encode("utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    for stage in ("a2", "a3", "a4", "a5"):
        parser.add_argument(f"--{stage}", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    directories = {stage: getattr(args, stage.lower()).resolve() for stage in ("A2", "A3", "A4", "A5")}
    try:
        if run_git(repo, "branch", "--show-current") != "develop/0.10.0":
            raise ValueError("wrong branch")
        if run_git(repo, "rev-parse", "HEAD") != FULL_HEAD:
            raise ValueError("original A2-A5 source HEAD changed")
        if run_git(repo, "diff", "--cached", "--name-only"):
            raise ValueError("staged changes present")
        script = Path(__file__).resolve()
        outputs = prepare_outputs(repo, script, directories)
        print("C1-H12 A2–A5 evidence preflight PASS: 3 A2 + 9 A3 + 21 A4 + 21 A5 raw cases")
        print("  A4: 6 known false convergences preserved; A5: 21/21 certified, 18 Converged, 3 MaxIterations")
        print("  checksums: 108 log/time files, 3 original matrix files, 4 matching HEAD manifests")
        for name in outputs:
            print("  " + name)
        if args.dry_run:
            print("DRY RUN: no repository or raw evidence files changed")
            return 0
        for name, data in outputs.items():
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        check_source_integrity(repo)
        subprocess.run(("git", "diff", "--check"), cwd=repo, check=True)
        print("C1-H12 A2–A5 evidence installed; no raw logs modified; no commit/push")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError, UnicodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
