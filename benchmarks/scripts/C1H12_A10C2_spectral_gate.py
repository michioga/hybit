#!/usr/bin/env python3
"""Fail-closed A10C2 Ritz/projection schema and frozen-provenance audit."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import CASES, NNZ
from benchmarks.scripts.C1H12_A10C1_precision import load_expected_matrix_hashes
from benchmarks.scripts.C1H12_A10C1_precision_gate import validate as validate_c1
from benchmarks.scripts.C1H12_A10C2_spectral import (
    ARMS, PROJECTION_FIELDS, SEEDS, SPECTRAL_FIELDS, VECTORS,
)


def read_csv(path: Path, fields: tuple[str, ...], expected: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f"incorrect field schema: {path.name}")
        rows = list(reader)
    if len(rows) != expected:
        raise ValueError(f"wrong record count: {path.name}")
    if any(set(row) != set(fields) or any(not value for value in row.values()) for row in rows):
        raise ValueError("missing or extra per-record data")
    return rows


def number(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}")
    return value


def integer(row: dict[str, str], field: str, at_least: int = 0) -> int:
    value = row[field]
    if not re.fullmatch(r"0|[1-9][0-9]*", value) or int(value) < at_least:
        raise ValueError(f"invalid {field}")
    return int(value)


def close(a: float, b: float, tolerance: float = 2e-10) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-14)


def validate(target: Path, frozen_a10b: Path, frozen_a10c1: Path
             ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    original = validate_c1(frozen_a10c1 / "precision_floor.csv", frozen_a10b)
    hashes = load_expected_matrix_hashes(frozen_a10b / "matrix_sha256.txt")
    checksum = hashlib.sha256((frozen_a10c1 / "precision_floor.csv").read_bytes()).hexdigest()
    if (target / "FROZEN_A10C1_SHA256.txt").read_text(encoding="ascii").strip() != checksum:
        raise ValueError("frozen A10C1 evidence changed after pilot")
    for name in ("HEAD.txt", "FROZEN_A10B_HEAD.txt", "FROZEN_A10C1_HEAD.txt"):
        value = (target / name).read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"invalid provenance {name}")
    if (target / "FROZEN_A10B_HEAD.txt").read_bytes() != (frozen_a10b / "HEAD.txt").read_bytes():
        raise ValueError("frozen A10B HEAD mismatch")
    if (target / "FROZEN_A10C1_HEAD.txt").read_bytes() != (frozen_a10c1 / "HEAD.txt").read_bytes():
        raise ValueError("frozen A10C1 HEAD mismatch")

    spectra = read_csv(target / "spectral_ritz.csv", SPECTRAL_FIELDS, len(CASES) * len(SEEDS))
    spectral_keys = set()
    spectrum = {}
    for row in spectra:
        case, seed = row["case"], row["seed"]
        if case not in CASES or seed not in SEEDS or (case, seed) in spectral_keys:
            raise ValueError("duplicate or invalid spectral case/seed")
        spectral_keys.add((case, seed))
        spectrum[case, seed] = row
        if (row["matrix_sha256"] != hashes[case]
                or integer(row, "rows") != 20736
                or integer(row, "nnz") != NNZ[case]
                or not 1 <= integer(row, "steps") <= 48):
            raise ValueError("invalid frozen spectral matrix or steps")
        low, high = number(row, "ritz_low"), number(row, "ritz_high")
        lr, hr = number(row, "ritz_low_residual"), number(row, "ritz_high_residual")
        if not (0 < low <= high and lr >= 0 and hr >= 0):
            raise ValueError("bad SPD Ritz values or residuals")
        if not (close(number(row, "low_interval_lower"), low - lr)
                and close(number(row, "low_interval_upper"), low + lr)):
            raise ValueError("inconsistent spectral enclosure")
        if (number(row, "max_orthogonality") < 0
                or number(row, "max_orthogonality") > 1e-9
                or number(row, "lanczos_recurrence_error") < 0
                or number(row, "lanczos_recurrence_error") > 2e-9):
            raise ValueError("uncertified Lanczos recurrence/orthogonality")
        start = number(row, "scaled_start_rayleigh")
        if start < low - 2e-9 or start > high + 2e-9:
            raise ValueError("starting Rayleigh is outside computed Ritz interval")
        if seed == "scaled_ones":
            expected = next(x for x in original if x["case"] == case and x["rhs_mode"] == "ones")
            if not close(start, float(expected["diag_scaled_rayleigh"])):
                raise ValueError("independent A10C1 Rayleigh crosscheck failed")
    if spectral_keys != {(case, seed) for case in CASES for seed in SEEDS}:
        raise ValueError("incomplete spectral grid")

    projections = read_csv(target / "coarse_projection.csv", PROJECTION_FIELDS,
                           len(CASES) * len(ARMS) * len(VECTORS))
    projection_keys = set()
    arm_metadata = {}
    for row in projections:
        case, arm, vec = row["case"], row["arm"], row["vector"]
        key = (case, arm, vec)
        if case not in CASES or arm not in ARMS or vec not in VECTORS or key in projection_keys:
            raise ValueError("duplicate or unknown projection record")
        projection_keys.add(key)
        if (row["matrix_sha256"] != hashes[case]
                or integer(row, "rows") != 20736
                or integer(row, "nnz") != NNZ[case]):
            raise ValueError("projection original CSR provenance mismatch")
        aggregates = integer(row, "aggregates", 1)
        words = integer(row, "abtm_words", 1)
        edges = integer(row, "abtm_edges", 1)
        pnnz = integer(row, "prolongation_nnz", 20736)
        omega = number(row, "omega")
        if (aggregates < 20736 // 4 or aggregates > 20736
                or words > 2 * NNZ[case] or edges > NNZ[case]
                or pnnz > NNZ[case] + 20736 or not 0 < omega <= 0.67):
            raise ValueError("impossible ABTM aggregation or prolongation work")
        metadata = (aggregates, words, edges, pnnz, omega)
        if (case, arm) in arm_metadata and arm_metadata[case, arm] != metadata:
            raise ValueError("aggregate metadata varies with input vector")
        arm_metadata[case, arm] = metadata
        err, captured = number(row, "projection_error_d"), number(row, "captured_energy_d")
        normal, residual = number(row, "normal_relative_residual"), number(row, "vector_ritz_residual")
        steps = integer(row, "projection_iterations")
        if (not 0 <= err <= 1.0 + 2e-8 or not 0 <= captured <= 1.0 + 2e-8
                or abs(captured - max(0.0, 1.0 - err * err)) > 2e-8
                or not 0 <= normal <= 1e-9):
            raise ValueError("invalid D-projection or normal-equation certification")
        if (arm.endswith("smoothed") and not 1 <= steps <= 400) or (not arm.endswith("smoothed") and steps != 0):
            raise ValueError("projection solver schema differs from basis")
        if vec == "manufactured_ones":
            if residual != 0.0 or (not arm.endswith("smoothed") and err > 1e-10):
                raise ValueError("piecewise aggregate must exactly capture constants")
        else:
            seed = "scaled_ones" if vec == "ritz_low_ones" else "seeded_random"
            if not close(residual, number(spectrum[case, seed], "ritz_low_residual")):
                raise ValueError("low Ritz vector residual fingerprint mismatch")
    expected_keys = {(case, arm, vec) for case in CASES for arm in ARMS for vec in VECTORS}
    if projection_keys != expected_keys:
        raise ValueError("incomplete ABTM coarse-projection grid")
    return spectra, projections


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: C1H12_A10C2_spectral_gate.py OUTPUT_DIR FROZEN_A10B FROZEN_A10C1")
    try:
        s, p = validate(*map(Path, sys.argv[1:]))
        print(f"C1-H12-A10C2 {len(s)}/6 Ritz and {len(p)}/27 ABTM D-projection/provenance gates PASS")
        print("Scope: deterministic subspace Ritz candidates, not certified global spectral extrema or PCG speedup")
    except (ValueError, OSError, StopIteration) as error:
        raise SystemExit(f"C1-H12-A10C2 evidence FAIL: {error}") from error
