#!/usr/bin/env python3
"""A10C2: independent symmetric Lanczos and ABTM coarse-space diagnostics.

Research-only, standard-library implementation. Ritz values are subspace
estimates, NEVER certified extremal eigenvalues or PCG convergence guarantees.
"""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from benchmarks.scripts.C1H12_A10C1_precision import (
    Matrix, load_expected_matrix_hashes, read_matrix_market,
)
from benchmarks.scripts.C1H12_A10B_prepared_gate import CASES, NNZ

SEEDS = ("scaled_ones", "seeded_random")
ARMS = ("graph_piecewise", "strong_graph_piecewise", "strong_graph_smoothed")
VECTORS = ("manufactured_ones", "ritz_low_ones", "ritz_low_random")
AGGREGATE_SIZE = 4
STRENGTH_THRESHOLD = 0.25
LANCZOS_STEPS = 48

SPECTRAL_FIELDS = (
    "case", "matrix_sha256", "seed", "rows", "nnz", "steps",
    "ritz_low", "ritz_high", "ritz_low_residual", "ritz_high_residual",
    "low_interval_lower", "low_interval_upper", "max_orthogonality",
    "lanczos_recurrence_error", "scaled_start_rayleigh",
)
PROJECTION_FIELDS = (
    "case", "matrix_sha256", "arm", "vector", "rows", "nnz",
    "aggregates", "abtm_words", "abtm_edges", "prolongation_nnz",
    "omega", "projection_error_d", "captured_energy_d",
    "normal_relative_residual", "projection_iterations", "vector_ritz_residual",
)


def dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def scaled_operator(a: Matrix) -> list[list[tuple[int, float]]]:
    d = [math.sqrt(x) for x in a.diag]
    return [[(j, v / (d[i] * d[j])) for j, v in row]
            for i, row in enumerate(a.rows)]


def spmv(rows: list[list[tuple[int, float]]], x: list[float]) -> list[float]:
    return [math.fsum(v * x[j] for j, v in row) for row in rows]


def normalize(v: list[float]) -> list[float]:
    length = norm(v)
    if length <= 0.0 or not math.isfinite(length):
        raise ValueError("zero or nonfinite Lanczos starting vector")
    return [x / length for x in v]


def random_vector(n: int) -> list[float]:
    state, mask = 0x9E3779B97F4A7C15, (1 << 64) - 1
    result = []
    for _ in range(n):
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        result.append(((state >> 11) / (1 << 53)) * 2.0 - 1.0)
    return result


def jacobi_symmetric(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Deterministic, dependency-free symmetric Jacobi eigensystem."""
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("invalid symmetric eigensystem")
    a = [row.copy() for row in matrix]
    v = [[float(i == j) for j in range(n)] for i in range(n)]
    scale = max(1.0, max(abs(a[i][j]) for i in range(n) for j in range(n)))
    for _ in range(max(20, 15 * n * n)):
        p, q, maximum = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > maximum:
                    p, q, maximum = i, j, abs(a[i][j])
        if maximum <= 2e-14 * scale:
            break
        tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
        t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k == p or k == q:
                continue
            akp, akq = a[k][p], a[k][q]
            a[k][p] = a[p][k] = c * akp - s * akq
            a[k][q] = a[q][k] = s * akp + c * akq
        for k in range(n):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    else:
        raise ValueError("small symmetric Jacobi eigensolver did not converge")
    order = sorted(range(n), key=lambda j: a[j][j])
    return [a[j][j] for j in order], [[v[i][j] for i in range(n)] for j in order]


@dataclass
class Ritz:
    low: float
    high: float
    low_residual: float
    high_residual: float
    low_vector: list[float]
    steps: int
    orthogonality: float
    recurrence_error: float
    start_rayleigh: float


def lanczos(rows: list[list[tuple[int, float]]], start: list[float],
            steps: int = LANCZOS_STEPS) -> Ritz:
    n = len(rows)
    if n != len(start) or steps <= 0:
        raise ValueError("invalid Lanczos dimensions")
    basis: list[list[float]] = []
    alpha: list[float] = []
    beta: list[float] = []
    q = normalize(start)
    prev = [0.0] * n
    previous_beta = 0.0
    last_beta = 0.0
    for k in range(min(steps, n)):
        z = spmv(rows, q)
        ak = dot(q, z)
        z = [zi - ak * qi - previous_beta * pi for zi, qi, pi in zip(z, q, prev)]
        # Full double reorthogonalization prevents ghost Ritz values.
        candidate = basis + [q]
        for _ in range(2):
            for old in candidate:
                coeff = dot(old, z)
                if coeff:
                    z = [zi - coeff * vi for zi, vi in zip(z, old)]
        last_beta = norm(z)
        basis.append(q)
        alpha.append(ak)
        if last_beta < 1e-14:
            last_beta = 0.0
            break
        if k + 1 >= min(steps, n):
            break
        beta.append(last_beta)
        prev, q = q, [zi / last_beta for zi in z]
        previous_beta = last_beta
    m = len(alpha)
    t = [[0.0] * m for _ in range(m)]
    for i in range(m):
        t[i][i] = alpha[i]
        if i + 1 < m:
            t[i][i + 1] = t[i + 1][i] = beta[i]
    values, eigenvectors = jacobi_symmetric(t)
    if values[0] <= 0.0 or not all(math.isfinite(val) for val in values):
        raise ValueError("nonpositive/nonfinite Ritz value: SPD is not established")

    orthogonality = 0.0
    for i in range(m):
        for j in range(i):
            orthogonality = max(orthogonality, abs(dot(basis[i], basis[j])))
    if orthogonality > 1e-9:
        raise ValueError("Lanczos basis lost orthogonality")

    def reconstruct(theta: float, weights: list[float]) -> tuple[list[float], float, float]:
        vector = [math.fsum(weights[j] * basis[j][i] for j in range(m)) for i in range(n)]
        vector = normalize(vector)
        observed = norm([vi - theta * ui for vi, ui in zip(spmv(rows, vector), vector)])
        expected = last_beta * abs(weights[-1])
        if abs(observed - expected) > 2e-9 * max(1.0, observed, expected):
            raise ValueError("independent Ritz residual contradicts Lanczos recurrence")
        return vector, observed, abs(observed - expected)

    low_vec, low_r, low_err = reconstruct(values[0], eigenvectors[0])
    _, high_r, high_err = reconstruct(values[-1], eigenvectors[-1])
    return Ritz(values[0], values[-1], low_r, high_r, low_vec,
                m, orthogonality, max(low_err, high_err), alpha[0])


def make_abtm_adjacency(matrix: Matrix, strong: bool) -> tuple[list[dict[int, int]], int, int]:
    """Bidirectional edge union with actual u64 packed bit-mask traversal."""
    words: list[dict[int, int]] = [{} for _ in range(matrix.n)]
    for i, row in enumerate(matrix.rows):
        maximum = max((abs(value) for j, value in row if j != i), default=0.0)
        for j, value in row:
            if i == j or (strong and abs(value) < STRENGTH_THRESHOLD * maximum):
                continue
            wi, wj = i >> 6, j >> 6
            words[i][wj] = words[i].get(wj, 0) | (1 << (j & 63))
            words[j][wi] = words[j].get(wi, 0) | (1 << (i & 63))
    word_count = sum(len(row) for row in words)
    edge_count = sum(mask.bit_count() for row in words for mask in row.values()) // 2
    return words, word_count, edge_count


def abtm_neighbors(words: dict[int, int]):
    for word in sorted(words):
        bits = words[word]
        while bits:
            low = bits & -bits
            yield (word << 6) + (low.bit_length() - 1)
            bits &= bits - 1


def aggregate(matrix: Matrix, strong: bool) -> tuple[list[int], int, int, int]:
    words, word_count, edge_count = make_abtm_adjacency(matrix, strong)
    owner = [-1] * matrix.n
    count = 0
    for seed in range(matrix.n):
        if owner[seed] != -1:
            continue
        owner[seed] = count
        queue = deque([seed])
        members = 1
        while members < AGGREGATE_SIZE and queue:
            row = queue.popleft()
            for neighbor in abtm_neighbors(words[row]):
                if owner[neighbor] < 0:
                    owner[neighbor] = count
                    members += 1
                    if members < AGGREGATE_SIZE:
                        queue.append(neighbor)
                    if members == AGGREGATE_SIZE:
                        break
        count += 1
    if min(owner) < 0 or count < 1 or max(owner) != count - 1:
        raise ValueError("invalid ABTM aggregation")
    return owner, count, word_count, edge_count


def prolongation(matrix: Matrix, owners: list[int], smoothed: bool) -> tuple[list[list[tuple[int, float]]], float]:
    ratio = max(math.fsum(abs(value) for _, value in row) / d
                for row, d in zip(matrix.rows, matrix.diag))
    omega = min(0.67, 1.6 / ratio)
    rows: list[list[tuple[int, float]]] = []
    for i, row in enumerate(matrix.rows):
        weights = {owners[i]: 1.0}
        if smoothed:
            for j, value in row:
                group = owners[j]
                weights[group] = weights.get(group, 0.0) - omega * value / matrix.diag[i]
        result = sorted((j, v) for j, v in weights.items() if v != 0.0)
        if not result:
            raise ValueError("empty AMG prolongation row")
        rows.append(result)
    return rows, omega


def d_projection(matrix: Matrix, p: list[list[tuple[int, float]]], coarse_n: int,
                 vector: list[float], owners: list[int] | None = None) -> tuple[float, float, float, int]:
    """D-orthogonal least-squares projection; smoothed P uses normal-PCG."""
    diagonal = matrix.diag
    v_norm_sq = math.fsum(d * v * v for d, v in zip(diagonal, vector))
    if v_norm_sq <= 0.0:
        raise ValueError("zero probe norm")
    right = [0.0] * coarse_n
    gram_diag = [0.0] * coarse_n
    for i, row in enumerate(p):
        for j, weight in row:
            right[j] += diagonal[i] * weight * vector[i]
            gram_diag[j] += diagonal[i] * weight * weight
    if any(d <= 0.0 or not math.isfinite(d) for d in gram_diag):
        raise ValueError("singular coarse least-squares normal operator")

    def gram_mv(x: list[float]) -> list[float]:
        result = [0.0] * coarse_n
        for i, row in enumerate(p):
            tmp = diagonal[i] * math.fsum(weight * x[j] for j, weight in row)
            for j, weight in row:
                result[j] += weight * tmp
        return result

    if owners is not None:
        coeff = [right[j] / gram_diag[j] for j in range(coarse_n)]
        iterations = 0
    else:
        coeff = [0.0] * coarse_n
        residual = right.copy()
        z = [residual[i] / gram_diag[i] for i in range(coarse_n)]
        direction = z.copy()
        rz = dot(residual, z)
        right_norm = norm(right)
        iterations = 0
        for k in range(1, 401):
            image = gram_mv(direction)
            den = dot(direction, image)
            if den <= 0.0 or not math.isfinite(den):
                raise ValueError("coarse normal-PCG lost positive definiteness")
            step = rz / den
            coeff = [x + step * dx for x, dx in zip(coeff, direction)]
            residual = [r - step * y for r, y in zip(residual, image)]
            iterations = k
            if norm(residual) <= 3e-12 * right_norm:
                break
            z = [residual[i] / gram_diag[i] for i in range(coarse_n)]
            rz_next = dot(residual, z)
            if rz <= 0.0 or rz_next < 0.0:
                raise ValueError("invalid normal-PCG recurrence")
            beta = rz_next / rz
            direction = [zi + beta * di for zi, di in zip(z, direction)]
            rz = rz_next
        else:
            raise ValueError("coarse normal-PCG did not converge")
    normal_res = norm([x - y for x, y in zip(gram_mv(coeff), right)]) / norm(right)
    if normal_res > 1e-9:
        raise ValueError("coarse projection normal residual is uncertified")
    diff = [v - math.fsum(weight * coeff[j] for j, weight in row)
            for v, row in zip(vector, p)]
    error_sq = math.fsum(d * r * r for d, r in zip(diagonal, diff)) / v_norm_sq
    error = math.sqrt(max(0.0, error_sq))
    if error > 1.0 + 2e-8:
        raise ValueError("D-orthogonal projection exceeds initial norm")
    return error, max(0.0, 1.0 - error_sq), normal_res, iterations


def analyze(corpus: Path, matrix_hashes: dict[str, str],
            a10c1_rows: list[dict[str, str]], steps: int = LANCZOS_STEPS
            ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spectra = []
    projections = []
    for case in CASES:
        source = corpus / f"c1g_p1_{case}_n144.mtx"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != matrix_hashes[case]:
            raise ValueError(f"frozen original FEM fingerprint differs: {case}")
        matrix = read_matrix_market(source)
        if matrix.n != 20736 or matrix.nnz != NNZ[case]:
            raise ValueError(f"unexpected original FEM dimensions: {case}")
        scaled = scaled_operator(matrix)
        starts = {
            "scaled_ones": [math.sqrt(d) for d in matrix.diag],
            "seeded_random": random_vector(matrix.n),
        }
        ritz = {seed: lanczos(scaled, start, steps) for seed, start in starts.items()}
        ones_record = next(row for row in a10c1_rows if row["case"] == case and row["rhs_mode"] == "ones")
        actual = ritz["scaled_ones"].start_rayleigh
        expected = float(ones_record["diag_scaled_rayleigh"])
        if abs(actual - expected) > 3e-10 * max(abs(expected), 1e-14):
            raise ValueError(f"A10C1 Rayleigh cross-check failed for {case}")
        for seed, item in ritz.items():
            spectra.append({
                "case": case, "matrix_sha256": digest, "seed": seed,
                "rows": str(matrix.n), "nnz": str(matrix.nnz), "steps": str(item.steps),
                "ritz_low": f"{item.low:.15e}", "ritz_high": f"{item.high:.15e}",
                "ritz_low_residual": f"{item.low_residual:.15e}",
                "ritz_high_residual": f"{item.high_residual:.15e}",
                "low_interval_lower": f"{item.low - item.low_residual:.15e}",
                "low_interval_upper": f"{item.low + item.low_residual:.15e}",
                "max_orthogonality": f"{item.orthogonality:.15e}",
                "lanczos_recurrence_error": f"{item.recurrence_error:.15e}",
                "scaled_start_rayleigh": f"{item.start_rayleigh:.15e}",
            })
        vectors = {
            "manufactured_ones": ([1.0] * matrix.n, 0.0),
            "ritz_low_ones": (
                [v / math.sqrt(d) for v, d in zip(ritz["scaled_ones"].low_vector, matrix.diag)],
                ritz["scaled_ones"].low_residual,
            ),
            "ritz_low_random": (
                [v / math.sqrt(d) for v, d in zip(ritz["seeded_random"].low_vector, matrix.diag)],
                ritz["seeded_random"].low_residual,
            ),
        }
        for arm in ARMS:
            owner, nc, words, edges = aggregate(matrix, arm.startswith("strong_"))
            p, omega = prolongation(matrix, owner, arm.endswith("smoothed"))
            for vector_name, (v, residual) in vectors.items():
                err, captured, normal, iters = d_projection(
                    matrix, p, nc, v, None if arm.endswith("smoothed") else owner)
                projections.append({
                    "case": case, "matrix_sha256": digest, "arm": arm,
                    "vector": vector_name, "rows": str(matrix.n), "nnz": str(matrix.nnz),
                    "aggregates": str(nc), "abtm_words": str(words), "abtm_edges": str(edges),
                    "prolongation_nnz": str(sum(len(row) for row in p)),
                    "omega": f"{omega:.15e}",
                    "projection_error_d": f"{err:.15e}",
                    "captured_energy_d": f"{captured:.15e}",
                    "normal_relative_residual": f"{normal:.15e}",
                    "projection_iterations": str(iters),
                    "vector_ritz_residual": f"{residual:.15e}",
                })
        print(f"A10C2 {case}: Ritz starts=2, ABTM arms=3, projection vectors=3", flush=True)
    return spectra, projections


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: C1H12_A10C2_spectral.py ORIGINAL_FEM FROZEN_A10B FROZEN_A10C1 OUTPUT_DIRECTORY")
    corpus, a10b, a10c1, target = map(Path, sys.argv[1:])
    from benchmarks.scripts.C1H12_A10C1_precision_gate import validate
    frozen = validate(a10c1 / "precision_floor.csv", a10b)
    expected = load_expected_matrix_hashes(a10b / "matrix_sha256.txt")
    spectra, projection = analyze(corpus, expected, frozen)
    write_csv(target / "spectral_ritz.csv", SPECTRAL_FIELDS, spectra)
    write_csv(target / "coarse_projection.csv", PROJECTION_FIELDS, projection)
    print(f"C1-H12-A10C2 wrote {len(spectra)} Lanczos and {len(projection)} ABTM coarse-projection records")


if __name__ == "__main__":
    main()
