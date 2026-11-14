#!/usr/bin/env python3
"""C1-G5F owner log intake: exact six-file audit of Rust shadow CSR features.

Only research/evidence processing. Never imported by hybit-auto's Rust solve path.
The six raw source logs must exactly match the owner-supplied checkpoint bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

BASE_HEAD = 'fc32f4aeb4ee3e76f85d7db7b05db1a8e2dac509'
G4_DIR = 'c1g_g2_hard_z32Jkd'
G5E_DIR = 'c1g5e_tri_p1_elasticity'
CASES = (
    # case, family, matrix filename, matrix hash, DOF, NNZ, off-diagonal NNZ, raw owner log hash
    ('scalar_island_shell', 'scalar_p1_diffusion', 'c1g_p1_island_shell_n144.mtx',
     '2d3f2834e3737610357525a3fdae9abaebb9ee21969d1800b0bd9c15414be45f',
     20736, 103104, 82368, 'ff5b90687966aaf09507e79de0dc9a4d19424853e7a72c44a0ebbe323571d84d'),
    ('scalar_multi_patch', 'scalar_p1_diffusion', 'c1g_p1_multi_patch_n144.mtx',
     '06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d',
     20736, 103104, 82368, '2afc0d4ed87a8f311115711af1e1419b5347510e6c8382530cd914b8bd5a76b2'),
    ('scalar_rotated_anisotropy', 'scalar_p1_diffusion', 'c1g_p1_rotated_anisotropy_n144.mtx',
     '03abf3e7d98f840dbb4a2ed09282d7abfbf28c928cff18837f0087af4fb51c42',
     20736, 144002, 123266, '40acfc3a8341e84feb411f3244fdfa318da2bc9078e72e59e49f33930c19d281'),
    ('elastic_uniform', 'tri_p1_plane_strain_elasticity', 'elastic_uniform.mtx',
     '7a594bd286172a8787db65bde0b9e74bd6bf887473371ce12f7523245f8a23ab',
     3280, 44546, 41266, '3fc1744d1846bfb2bb918890272bd58f96f1d84b441b551c7bac29a4d2e30a99'),
    ('elastic_stiff_lens', 'tri_p1_plane_strain_elasticity', 'elastic_stiff_lens.mtx',
     'f8f2da4fb488dded8d0882c2f0e4ede6f3db3e8ad321231ea368437c7cba9905',
     3280, 44546, 41266, '0d01068564f9ffa2132dd505b7a58d1246017255b67f6d792a72fd6e3a5822a5'),
    ('elastic_shear_layers', 'tri_p1_plane_strain_elasticity', 'elastic_shear_layers.mtx',
     'b6ba04300933286611802515e9dbfe52ddfd532d6927f85202fc12ed19a4319d',
     3280, 44528, 41248, '472e00b8bbf449300d02c6e14e38dd253e3f44703631988272b1756a70aa02c3'),
)
FIELDS = (
    'path', 'rows', 'nnz', 'offdiag', 'positive_fraction', 'nondominant_rows',
    'max_offdiag_ratio', 'strong_fraction', 'invalid_ratio', 'rhs_nnz_fraction',
    'diagonal_spread', 'shadow_reason', 'research_rank', 'production_recommended',
    'audit_ms',
)
CSV_FIELDS = (
    'case', 'family', 'matrix_sha256', 'log_sha256', 'rows', 'nnz',
    'offdiag', 'nondominant_rows', 'nondominant_fraction',
    'positive_fraction', 'diagonal_log10_spread', 'max_offdiag_ratio',
    'strong_fraction', 'rhs_nonzero_fraction', 'invalid_ratio',
    'audit_ms', 'shadow_reason', 'research_rank', 'production_recommended',
)
EXPECTED_RAW = frozenset({f'{case}.log' for case, *_ in CASES} | {'HEAD.txt', 'SHA256_LOGS.txt'})


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n', extrasaction='raise')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode('utf-8')


def json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def read_opt_float(text: str, key: str) -> float:
    match = re.fullmatch(r'Some\(([^()]+)\)', text)
    if match is None:
        raise ValueError(f'missing or nonfinite optional value: {key}')
    value = float(match.group(1))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'invalid optional value: {key}')
    return value


def read_opt_int(text: str, key: str) -> int:
    match = re.fullmatch(r'Some\(([0-9]+)\)', text)
    if match is None:
        raise ValueError(f'missing optional count: {key}')
    return int(match.group(1))


def read_fraction(text: str, key: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not (0 <= value <= 1):
        raise ValueError(f'fraction out of range: {key}')
    return value


def parse_log(data: bytes, case: tuple, scalar_dir: Path, elastic_dir: Path) -> dict:
    name, family, matrix_name, matrix_sha, expected_rows, expected_nnz, expected_offdiag, digest = case
    if sha(data) != digest:
        raise ValueError(f'owner raw log SHA-256 differs: {name}')
    try:
        content = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError('raw log is not UTF-8') from exc
    if not content.endswith('\n') or content.count('\n') != 1 or not content.startswith('C1G5F|'):
        raise ValueError(f'not exactly one C1G5F record: {name}')
    parts = content.strip().split('|')
    values = {}
    for part in parts[1:]:
        if part.count('=') != 1:
            raise ValueError(f'malformed field: {name}')
        k, v = part.split('=', 1)
        if k in values:
            raise ValueError(f'duplicate field: {k}')
        values[k] = v
    if set(values) != set(FIELDS):
        raise ValueError(f'field contract mismatch: {name}: {set(values) ^ set(FIELDS)}')
    matrix_path = (scalar_dir if family == 'scalar_p1_diffusion' else elastic_dir) / matrix_name
    if Path(values['path']) != matrix_path:
        raise ValueError(f'matrix identity mismatch: {name}')
    if (int(values['rows']), int(values['nnz']), int(values['offdiag'])) != (
        expected_rows, expected_nnz, expected_offdiag
    ):
        raise ValueError(f'CSR dimensions/nnz mismatch: {name}')
    nondom = read_opt_int(values['nondominant_rows'], 'nondominant_rows')
    if nondom > expected_rows:
        raise ValueError(f'nondominant count exceeds dimension: {name}')
    positive = read_fraction(values['positive_fraction'], 'positive_fraction')
    strong = read_opt_float(values['strong_fraction'], 'strong_fraction')
    if strong > 1:
        raise ValueError('strong fraction out of range')
    rhs_nnz = read_fraction(values['rhs_nnz_fraction'], 'rhs_nnz_fraction')
    max_ratio = read_opt_float(values['max_offdiag_ratio'], 'max_offdiag_ratio')
    diag_spread = read_opt_float(values['diagonal_spread'], 'diagonal_spread')
    audit_ms = float(values['audit_ms'])
    if not math.isfinite(audit_ms) or audit_ms < 0:
        raise ValueError('invalid wall time')
    if (values['invalid_ratio'], values['shadow_reason'], values['research_rank'],
        values['production_recommended']) != (
        'false', 'InsufficientIndependentEvidence', 'None', 'None'
    ):
        raise ValueError(f'missing fail-closed shadow contract: {name}')
    return dict(
        case=name, family=family, matrix_sha256=matrix_sha, log_sha256=digest,
        rows=expected_rows, nnz=expected_nnz, offdiag=expected_offdiag,
        nondominant_rows=nondom,
        nondominant_fraction=f'{nondom / expected_rows:.12f}',
        positive_fraction=values['positive_fraction'],
        diagonal_log10_spread=f'{diag_spread:.12g}',
        max_offdiag_ratio=f'{max_ratio:.12g}',
        strong_fraction=f'{strong:.12g}',
        rhs_nonzero_fraction=values['rhs_nnz_fraction'],
        invalid_ratio='false', audit_ms=values['audit_ms'],
        shadow_reason=values['shadow_reason'],
        research_rank='None', production_recommended='None',
    )


def verify_evidence(evidence: Path, scalar_dir: Path, elastic_dir: Path, *, check_matrices: bool = True) -> tuple[list[dict], list[dict]]:
    if not evidence.is_dir() or evidence.is_symlink():
        raise ValueError('missing evidence directory')
    evidence = evidence.resolve(strict=True)
    scalar_dir, elastic_dir = scalar_dir.resolve(strict=True), elastic_dir.resolve(strict=True)
    actual = {p.name for p in evidence.iterdir()}
    if actual != EXPECTED_RAW:
        raise ValueError(f'raw evidence file list mismatch: {sorted(actual ^ EXPECTED_RAW)}')
    files: dict[str, bytes] = {}
    for name in sorted(EXPECTED_RAW):
        p = evidence / name
        if not p.is_file() or p.is_symlink():
            raise ValueError(f'missing, directory or symlink raw file: {name}')
        files[name] = p.read_bytes()
    if files['HEAD.txt'] != (BASE_HEAD + '\n').encode():
        raise ValueError('owner run HEAD mismatches fc32f4a')
    index = {}
    for line in files['SHA256_LOGS.txt'].decode('ascii').splitlines():
        parts = line.split('  ', 1)
        if len(parts) != 2:
            raise ValueError('malformed original SHA256_LOGS line')
        value, filename = parts
        if not re.fullmatch('[0-9a-f]{64}', value) or filename in index:
            raise ValueError('malformed or duplicate original SHA256_LOGS entry')
        index[filename] = value
    expected_logs = {f'{name}.log' for name, *_ in CASES}
    if set(index) != expected_logs:
        raise ValueError('original SHA256_LOGS lists different files')
    if any(index[name] != sha(files[name]) for name in expected_logs):
        raise ValueError('original SHA256_LOGS mismatch')
    if check_matrices:
        for name, family, matrix_name, mat_sha, *_ in CASES:
            matrix_path = (scalar_dir if family == 'scalar_p1_diffusion' else elastic_dir) / matrix_name
            if not matrix_path.is_file() or matrix_path.is_symlink():
                raise ValueError(f'original matrix absent: {name}')
            h = hashlib.sha256()
            with matrix_path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != mat_sha:
                raise ValueError(f'original matrix SHA mismatch: {name}')
    observations = [parse_log(files[f'{case[0]}.log'], case, scalar_dir, elastic_dir) for case in CASES]
    manifest = [dict(filename=n, bytes=len(files[n]), sha256=sha(files[n])) for n in sorted(files)]
    return observations, manifest


def render_doc(rows: list[dict], raw: list[dict]) -> bytes:
    content = [
        '# C1-G5F — six-matrix runtime feature evidence (owner checkpoint, 2026-11-14)',
        '',
        '## Provenance and status',
        '',
        '- Source run HEAD: `fc32f4aeb4ee3e76f85d7db7b05db1a8e2dac509` (pre-C1-G5F checkpoint).',
        '- Owner WSL2: FIX1 gate PASS; 74 feature-on + 58 feature-off auto tests, 1 facade API test; fmt and Clippy PASS.',
        '- Two distinct **constructed** FEM generator families, three matrices each; not an independent held-out trained-model evaluation.',
        '- All six observables derive from the Rust `hybit-auto` API using the probe RHS `b=A*ones` and expected RHS reuse=1.',
        '- 6 owner raw log files plus `HEAD.txt` and `SHA256_LOGS.txt` are retained outside Git; their SHA-256/byte sizes are indexed.',
        '- No model training, cost-model certification, or change to production/FFI/default routing. The embedded standard still abstains.',
        '',
        '## Observations',
        '',
        '| Case | DOF | Nondominant rows | Nondominant % | Positive offdiag % | Strong links % | Diagonal log10 spread |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        content.append(f"| `{r['case']}` | {r['rows']} | {r['nondominant_rows']} | {100*float(r['nondominant_fraction']):.2f} | {100*float(r['positive_fraction']):.2f} | {100*float(r['strong_fraction']):.2f} | {float(r['diagonal_log10_spread']):.3f} |")
    content += [
        '',
        '## Interpretation limits',
        '',
        '- The strong-link statistic uses the C1-G5F library definition. It is **not** a universal measure of AMG difficulty.',
        '- Diagonal non-dominance alone does not indicate non-SPD or a need for native Multilevel. SPD requires separate numerical/mathematical checks.',
        '- The probe RHS is manufactured and not representative of physical external loads or the four RHS used in C1-G5E.',
        '- Audit elapsed times are one-shot diagnostics on differently sized matrices. They are **not** comparable four-arm solution costs.',
        '- Six matrix observations within two families must not be treated as six independent held-out families.',
        '- The 48-way elasticity outcomes remain a separate, prior frozen research dataset. No labels are joined into runtime features.',
        '- Future work: physical FEM families, matched costs across RHS/reuse budgets, family-heldout classification and conservative abstention.',
        '',
        '## Evidence files',
        '',
        '- `benchmarks/evidence/C1G5F_20261114_features.csv` — exact validated observation values, one row per matrix.',
        '- `benchmarks/evidence/C1G5F_20261114_summary.json` — qualification flags (production blocked).',
        '- `benchmarks/evidence/C1G5F_20261114_raw_sha256.csv` — ' + str(len(raw)) + ' original owner evidence files.',
        '- `benchmarks/scripts/C1G5F_feature_evidence.py` and `test_C1G5F_feature_evidence.py` — fail-closed reproducible audit.',
        '',
    ]
    return ('\n'.join(content)).encode('utf-8')


def build_outputs(rows: list[dict], raw: list[dict]) -> dict[str, bytes]:
    summary = dict(
        schema='C1G5F_FEATURE_EVIDENCE_V1', origin_head=BASE_HEAD,
        generator_families=2, generator_matrices=6,
        measured_observations=6, original_evidence_files=len(raw),
        raw_logs_preserved_outside_git=True,
        rhs='manufactured b=A*ones; expected_rhs_reuse=1; not C1-G5E four-RHS benchmark',
        training_performed=False, independent_heldout_passed=False,
        embedded_standard_promoted=False, production_promotion_allowed=False,
        observation_cases=[row['case'] for row in rows],
    )
    return {
        'benchmarks/evidence/C1G5F_20261114_features.csv': csv_bytes(rows, CSV_FIELDS),
        'benchmarks/evidence/C1G5F_20261114_summary.json': json_bytes(summary),
        'benchmarks/evidence/C1G5F_20261114_raw_sha256.csv': csv_bytes(raw, ('filename', 'bytes', 'sha256')),
        'docs/C1G5F_SIX_MATRIX_FEATURE_EVIDENCE_20261114.md': render_doc(rows, raw),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--scalar-dir', type=Path, required=True)
    parser.add_argument('--elastic-dir', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    rows, raw = verify_evidence(args.evidence, args.scalar_dir, args.elastic_dir)
    outputs = build_outputs(rows, raw)
    print(json.dumps({'observations': len(rows), 'original_files': len(raw),
                      'evidence_digest': sha(b''.join(outputs[k] for k in sorted(outputs))),
                      'production_promotion_allowed': False}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, UnicodeError) as error:
        raise SystemExit(f'C1-G5F EVIDENCE ERROR: {error}')
