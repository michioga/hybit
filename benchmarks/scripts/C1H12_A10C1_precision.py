#!/usr/bin/env python3
"""A10C-1: independent Matrix Market cancellation and attainable-residual diagnostics.

This estimates floating-point SpMV effects at manufactured exact vectors; it
does not modify the A10B PCG algorithm or estimate eigenvalue extremals.
"""

from __future__ import annotations

import csv
import hashlib
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmarks.scripts.C1H12_A10B_prepared_gate import CASES, NNZ, RHS

FIELDS = (
    "case", "rhs_mode", "matrix_sha256", "rows", "nnz", "max_row_nnz",
    "rhs_norm", "abs_ax_norm", "cancellation_ratio", "roundoff_bound_relative",
    "spmv_fsum_disagreement", "x_rayleigh", "diag_scaled_rayleigh",
    "frozen_rhs_hash", "fresh_converged", "fresh_max_iterations",
    "reuse_converged", "reuse_max_iterations", "frozen_min_true_relative",
    "frozen_max_true_relative",
)


@dataclass
class Matrix:
    rows: list[list[tuple[int, float]]]
    diag: list[float]
    nnz: int

    @property
    def n(self) -> int:
        return len(self.rows)


def read_matrix_market(path: Path) -> Matrix:
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip().lower().split()
        if len(header) != 5 or header[:4] != ["%%matrixmarket", "matrix", "coordinate", "real"]:
            raise ValueError("expected real coordinate Matrix Market matrix")
        if header[4] not in ("general", "symmetric"):
            raise ValueError("unsupported Matrix Market symmetry")
        dimensions = None
        for line in stream:
            if line.lstrip().startswith("%") or not line.strip():
                continue
            dimensions = line.split()
            break
        if dimensions is None or len(dimensions) != 3:
            raise ValueError("missing Matrix Market dimensions")
        nrows, ncols, entries = map(int, dimensions)
        if nrows != ncols or nrows <= 0 or entries <= 0:
            raise ValueError("expected positive square Matrix Market matrix")
        sparse: list[dict[int, list[float]]] = [{} for _ in range(nrows)]
        seen = 0
        for line in stream:
            if not line.strip() or line.lstrip().startswith("%"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError("invalid Matrix Market entry")
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            value = float(parts[2])
            if not (0 <= i < nrows and 0 <= j < ncols and math.isfinite(value)):
                raise ValueError("nonfinite or out-of-range matrix entry")
            sparse[i].setdefault(j, []).append(value)
            if header[4] == "symmetric" and i != j:
                sparse[j].setdefault(i, []).append(value)
            seen += 1
        if seen != entries:
            raise ValueError("Matrix Market entry count mismatch")
    rows = [sorted((j, math.fsum(values)) for j, values in row.items()) for row in sparse]
    diag = [next((v for j, v in row if j == i), 0.0) for i, row in enumerate(rows)]
    if any(d <= 0.0 or not math.isfinite(d) for d in diag):
        raise ValueError("nonpositive or missing diagonal")
    # Reject non-symmetric input, rather than manufacturing a spectral result.
    for i, row in enumerate(rows):
        for j, value in row:
            transpose = dict(rows[j]).get(i)
            if transpose is None or abs(value - transpose) > 1e-12 * max(1.0, abs(value), abs(transpose)):
                raise ValueError("asymmetric matrix: not a symmetric-SPD spectral diagnostic")
    return Matrix(rows, diag, sum(len(row) for row in rows))


def manufactured_x(mode: str, n: int) -> list[float] | None:
    if mode == "point_load":
        return None
    if mode == "ones":
        return [1.0] * n
    if mode == "index_smooth":
        return [
            0.5 + 0.3 * math.sin(2.0 * math.pi * i / n)
            + 0.2 * math.cos(6.0 * math.pi * i / n)
            for i in range(n)
        ]
    if mode == "seeded_random":
        state = 0x9E3779B97F4A7C15
        values = []
        mask = (1 << 64) - 1
        for _ in range(n):
            state ^= (state << 13) & mask
            state ^= state >> 7
            state ^= (state << 17) & mask
            values.append((state >> 11) * (2.0 / (1 << 53)) - 1.0)
        return values
    raise ValueError("unknown manufactured RHS")


def norm(values: list[float]) -> float:
    return math.sqrt(math.fsum(v * v for v in values))


def rhs_hash(values: list[float]) -> str:
    value = 0xCBF29CE484222325
    for item in values:
        for byte in struct.pack("<d", item):
            value = ((value ^ byte) * 0x00000100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def diagnostics(matrix: Matrix, mode: str) -> dict[str, str]:
    n = matrix.n
    x = manufactured_x(mode, n)
    max_row_nnz = max(map(len, matrix.rows))
    if x is None:
        b = [0.0] * n
        b[n // 2] = 1.0
        return {
            "rhs_norm": "1.000000000000e+00", "abs_ax_norm": "n/a",
            "cancellation_ratio": "n/a", "roundoff_bound_relative": "n/a",
            "spmv_fsum_disagreement": "n/a", "x_rayleigh": "n/a",
            "diag_scaled_rayleigh": "n/a", "reference_rhs_hash": rhs_hash(b),
            "max_row_nnz": str(max_row_nnz),
        }
    # Same row order as canonical CSR, plus an independently compensated sum.
    b = []
    b_fsum = []
    abs_ax = []
    for row in matrix.rows:
        total = 0.0
        terms = []
        for j, v in row:
            term = v * x[j]
            total += term
            terms.append(term)
        b.append(total)
        b_fsum.append(math.fsum(terms))
        abs_ax.append(math.fsum(abs(t) for t in terms))
    b_norm = norm(b)
    if b_norm == 0.0:
        raise ValueError("zero manufactured RHS: relative-residual floor undefined")
    abs_norm = norm(abs_ax)
    u = 2.0 ** -53
    # Conservative elementwise dot-product model: multiply and add rounding.
    gamma = 2 * max_row_nnz * u / (1.0 - 2 * max_row_nnz * u)
    rayleigh_num = math.fsum(xi * yi for xi, yi in zip(x, b_fsum))
    rayleigh_den = math.fsum(xi * xi for xi in x)
    scaled_den = math.fsum(d * xi * xi for d, xi in zip(matrix.diag, x))
    if scaled_den <= 0.0 or rayleigh_den <= 0.0:
        raise ValueError("invalid Rayleigh denominator")
    return {
        "rhs_norm": f"{b_norm:.12e}",
        "abs_ax_norm": f"{abs_norm:.12e}",
        "cancellation_ratio": f"{abs_norm / b_norm:.12e}",
        "roundoff_bound_relative": f"{gamma * abs_norm / b_norm:.12e}",
        "spmv_fsum_disagreement": f"{norm([a - c for a, c in zip(b, b_fsum)]) / b_norm:.12e}",
        "x_rayleigh": f"{rayleigh_num / rayleigh_den:.12e}",
        "diag_scaled_rayleigh": f"{rayleigh_num / scaled_den:.12e}",
        "reference_rhs_hash": rhs_hash(b),
        "max_row_nnz": str(max_row_nnz),
    }


def summarize_frozen(rows: list[dict[str, str]], case: str, mode: str) -> dict[str, str]:
    subset = [r for r in rows if r["case"] == case and r["rhs_mode"].lower().replace("smooth", "_smooth").replace("random", "_random").replace("load", "_load") == mode]
    if len(subset) != 60:
        raise ValueError(f"A10B case/RHS sample count mismatch: {case}/{mode}")
    hashes = {r["rhs_hash"] for r in subset}
    if len(hashes) != 1:
        raise ValueError("inconsistent frozen RHS fingerprint")
    outcome = {"frozen_rhs_hash": hashes.pop()}
    for reuse in ("fresh", "reuse"):
        part = [r for r in subset if r["reuse_mode"] == reuse]
        if len(part) != 30:
            raise ValueError("frozen lifecycle sample count mismatch")
        outcome[f"{reuse}_converged"] = str(sum(r["status"] == "Converged" for r in part))
        outcome[f"{reuse}_max_iterations"] = str(sum(r["status"] == "MaxIterations" for r in part))
    residuals = [float(r["verified_true_relative"]) for r in subset]
    outcome["frozen_min_true_relative"] = f"{min(residuals):.12e}"
    outcome["frozen_max_true_relative"] = f"{max(residuals):.12e}"
    return outcome


def analyze(corpus: Path, frozen_rows: list[dict[str, str]], matrix_hashes: dict[str, str]) -> list[dict[str, str]]:
    result = []
    for case in CASES:
        path = corpus / f"c1g_p1_{case}_n144.mtx"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != matrix_hashes[case]:
            raise ValueError(f"original FEM SHA mismatch: {case}")
        matrix = read_matrix_market(path)
        if matrix.n != 20736 or matrix.nnz != NNZ[case]:
            raise ValueError(f"unexpected original CSR dimensions: {case}")
        for mode in RHS:
            diag = diagnostics(matrix, mode)
            frozen = summarize_frozen(frozen_rows, case, mode)
            # Point load has an exact integer-valued RHS and can be hashed
            # identically to the frozen Rust source.
            if mode == "point_load" and diag["reference_rhs_hash"] != frozen["frozen_rhs_hash"]:
                raise ValueError("point-load RHS differs from frozen A10B")
            result.append({
                "case": case, "rhs_mode": mode, "matrix_sha256": digest,
                "rows": str(matrix.n), "nnz": str(matrix.nnz),
                **{k: v for k, v in diag.items() if k != "reference_rhs_hash"},
                **frozen,
            })
    return result


def load_expected_matrix_hashes(path: Path) -> dict[str, str]:
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) != 2 or len(tokens[0]) != 64:
            raise ValueError("invalid frozen matrix SHA manifest")
        for case in CASES:
            if Path(tokens[1]).name == f"c1g_p1_{case}_n144.mtx":
                if case in expected:
                    raise ValueError("duplicate frozen matrix SHA")
                expected[case] = tokens[0]
    if set(expected) != set(CASES):
        raise ValueError("incomplete frozen matrix SHA manifest")
    return expected


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: C1H12_A10C1_precision.py CORPUS FROZEN_A9 FROZEN_A10A FROZEN_A10B OUTPUT_CSV")
    from benchmarks.scripts.C1H12_A10B_prepared_gate import validate

    corpus, a9, a10a, a10b, output = map(Path, sys.argv[1:])
    try:
        frozen_rows = validate(a9, a10a, a10b, write_csv=False)
        expected_hashes = load_expected_matrix_hashes(a10b / "matrix_sha256.txt")
        outcome = analyze(corpus, frozen_rows, expected_hashes)
        write_csv(outcome, output)
        print("C1-H12-A10C1 independent CSR precision probe: 12/12 records PASS")
        for row in outcome:
            if row["rhs_mode"] in ("ones", "point_load"):
                print(f"  {row['case']}/{row['rhs_mode']}: cancellation={row['cancellation_ratio']}; "
                      f"bound={row['roundoff_bound_relative']}; "
                      f"fresh_converged={row['fresh_converged']}/30")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"C1-H12-A10C1 FAIL: {exc}") from exc
