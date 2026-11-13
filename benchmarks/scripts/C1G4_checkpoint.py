#!/usr/bin/env python3
"""C1-G4: reproduce and freeze certified A9 rebaseline; never alter solver behavior.

The full 72 raw logs are parsed again using the previously accepted A9 gates.
The two persisted C1-G4 outputs are checked against the frozen raw records.
No runtime-speed or rescue claims are inferred solely from a PASS.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

try:
    from benchmarks.scripts.C1G4_current_head_verify import (
        FIELDS, EXPECTED_N, expected_specs, stem_for, normalized_record, audit)
except ModuleNotFoundError:
    from C1G4_current_head_verify import (
        FIELDS, EXPECTED_N, expected_specs, stem_for, normalized_record, audit)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rederive(repo: Path, evidence: Path) -> tuple[dict, list[dict[str, str]]]:
    # First independently enforce provenance, CSR residual and 72-log gates.
    checked = audit(repo, evidence, check_only=True)
    frozen_summary = evidence / 'c1g4_current_head_summary.json'
    frozen_csv = evidence / 'c1g4_current_head_comparison.csv'
    if not frozen_csv.is_file() or not frozen_summary.is_file():
        raise ValueError('missing accepted FIX1 CSV or JSON')
    if json.loads(frozen_summary.read_text(encoding='utf-8')) != checked:
        raise ValueError('frozen summary differs from audited original logs')
    sys.path.insert(0, str(repo))
    from benchmarks.scripts.C1H12_A9_rhs_gate import parse_a9
    from benchmarks.scripts.C1H12_A4_numerical_gate import parse_time
    with frozen_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('frozen CSV field ordering/schema mismatch')
        stored = list(reader)
    if len(stored) != EXPECTED_N:
        raise ValueError(f'frozen CSV expected {EXPECTED_N} records, got {len(stored)}')
    index = {}
    for row in stored:
        key = tuple(row[k] for k in ('case', 'arm', 'rhs_mode', 'policy'))
        if key in index:
            raise ValueError('duplicate frozen CSV identity')
        index[key] = row
    derived = []
    for spec in expected_specs():
        stem = stem_for(spec)
        parsed = parse_a9((evidence / (stem + '.log')).read_text(encoding='utf-8'), *spec)
        time = parse_time((evidence / (stem + '.time')).read_text(encoding='utf-8'))
        canonical = normalized_record(spec, parsed, time)
        recorded = index.get(spec)
        if recorded is None:
            raise ValueError('missing frozen CSV identity: ' + stem)
        for key in FIELDS:
            if recorded[key] != str(canonical[key]):
                raise ValueError(f'frozen CSV altered at {stem}.{key}')
        derived.append(recorded)
    if len(index) != len(derived):
        raise ValueError('unexpected frozen CSV rows')
    return checked, derived


def stats_for(rows: list[dict[str, str]], mode: str) -> dict:
    selected = [r for r in rows if mode == 'all' or
                (mode == 'non_ones' and r['rhs_mode'] != 'ones') or r['rhs_mode'] == mode]
    base = [r for r in selected if r['policy'] == 'two_level']
    multi = [r for r in selected if r['policy'] == 'multilevel']
    pairs = []
    lookup = {(r['case'], r['arm'], r['rhs_mode'], r['policy']): r for r in selected}
    for b in base:
        key = (b['case'], b['arm'], b['rhs_mode'], 'multilevel')
        m = lookup.get(key)
        if m is None:
            raise ValueError('missing paired multilevel outcome')
        bt, mt = float(b['solve_ms']), float(m['solve_ms'])
        if not all(math.isfinite(v) and v > 0 for v in (bt, mt)):
            raise ValueError('invalid solve-time measurement')
        pairs.append((bt / mt, float(b['setup_ms']), float(m['setup_ms'])))
    if len(base) != len(multi) or len(pairs) != len(base):
        raise ValueError('unpaired comparison')
    return {
        'pairs': len(pairs),
        'two_level_converged': sum(r['status'] == 'Converged' for r in base),
        'multilevel_converged': sum(r['status'] == 'Converged' for r in multi),
        'two_level_iterations_sum': sum(int(r['iterations']) for r in base),
        'multilevel_iterations_sum': sum(int(r['iterations']) for r in multi),
        'two_level_solve_ms_sum': sum(float(r['solve_ms']) for r in base),
        'multilevel_solve_ms_sum': sum(float(r['solve_ms']) for r in multi),
        'two_level_setup_ms_sum': sum(float(r['setup_ms']) for r in base),
        'multilevel_setup_ms_sum': sum(float(r['setup_ms']) for r in multi),
        'paired_solve_speedup_median': statistics.median(x[0] for x in pairs),
        'two_level_peak_rss_kib_mean': statistics.mean(int(r['peak_rss_kib']) for r in base),
        'multilevel_peak_rss_kib_mean': statistics.mean(int(r['peak_rss_kib']) for r in multi),
        'two_level_max_true_relative': max(float(r['verified_true_relative']) for r in base),
        'multilevel_max_true_relative': max(float(r['verified_true_relative']) for r in multi),
        'two_level_max_forward_error': max((float(r['relative_x_error']) for r in base
                                             if r['relative_x_error'] != 'n/a'), default=None),
        'multilevel_max_forward_error': max((float(r['relative_x_error']) for r in multi
                                            if r['relative_x_error'] != 'n/a'), default=None),
    }


def compute(repo: Path, evidence: Path) -> tuple[dict, str, list[dict[str, str]]]:
    validated, rows = rederive(repo, evidence)
    if len(rows) != 72 or validated['status'] != 'PASS_ORIGINAL_CSR_AND_GRID':
        raise ValueError('numerical gate missing or wrong count')
    if validated['outcomes_total'] != {'BOTH_CONVERGED': 36}:
        raise ValueError('this checkpoint only accepts the observed 36 both-converged pairs')
    groups = {name: stats_for(rows, name) for name in
              ('all', 'ones', 'non_ones', 'index_smooth', 'seeded_random', 'point_load')}
    # Original A9 strict gate is the numerical authority; these are descriptive economics.
    result = {
        'checkpoint': 'C1G4_AMG_REBASELINE_20261113',
        'source_head': validated['head'],
        'matrix_sha256': validated['matrix_sha256'],
        'numerical_policy': validated['numerical_policy'],
        'numerical_gate': validated['status'],
        'runs': 72, 'pairs': 36, 'outcomes': validated['outcomes_total'],
        'group_statistics': groups,
        'frozen_comparison_sha256': sha256(evidence / 'c1g4_current_head_comparison.csv'),
        'frozen_summary_sha256': sha256(evidence / 'c1g4_current_head_summary.json'),
        'qualification': 'Constructed FEM / manufactured or algebraic RHS only; '
                         'one run per config; no held-out external FEM, no physical RHS, '
                         'no statistically qualified time advantage and no rescue.',
    }
    lines = []
    for spec in expected_specs():
        stem = stem_for(spec)
        for ext in ('log', 'time'):
            name = stem + '.' + ext
            lines.append({'file': name, 'sha256': sha256(evidence / name)})
    return result, csv_of(lines, ('file', 'sha256')), rows


def csv_of(rows: list[dict], fields: tuple[str, ...]) -> str:
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def md_report(result: dict, evidence: Path) -> str:
    g = result['group_statistics']
    def row(label, name):
        r = g[name]
        return (f'| {label} | {r["pairs"]} | {r["two_level_converged"]}/{r["pairs"]} | '
                f'{r["multilevel_converged"]}/{r["pairs"]} | '
                f'{r["two_level_iterations_sum"]} | {r["multilevel_iterations_sum"]} | '
                f'{r["two_level_solve_ms_sum"]:.3f} | {r["multilevel_solve_ms_sum"]:.3f} | '
                f'{r["paired_solve_speedup_median"]:.3f} |')
    return f'''# C1-G4 — current-HEAD multilevel AMG rebaseline (2026-11-13)

## Acceptance

- Numerical gate: **{result['numerical_gate']}** on frozen `develop/0.10.0` HEAD `{result['source_head']}`.
- 3 constructed n=20,736 FEM matrices, 3 aggregation/basis arms, 4 RHS modes, 2 preconditioners = **72 processes, 36 matched pairs**.
- **36 BOTH_CONVERGED, 0 RESCUE, 0 REGRESSION, 0 BOTH_NOT_CONVERGED.** Original CSR true residual at `1e-10`, 600 iteration cap, one thread, Rust 1.99.1.
- RHS: `ones`, index-smooth manufactured, seeded-random manufactured, and algebraic point-load (no true reference). All runs used the frozen matrix hashes below.
- The A9 prototype remains opt-in; there is **no production AMG**, no host-physical load, no held-out external FEM and no new default or API change.

## Descriptive comparison (one OS process per configuration)

| RHS cohort | Pairs | Two-Level converged | Multilevel converged | Two-Level iterations total | Multilevel iterations total | Two-Level solve ms total | Multilevel solve ms total | Median paired solve ratio (Two-Level/Multilevel) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{row('All', 'all')}
{row('Ones only', 'ones')}
{row('Non-ones', 'non_ones')}
{row('Index smooth', 'index_smooth')}
{row('Seeded random', 'seeded_random')}
{row('Point load', 'point_load')}

The sum of solver-only milliseconds omits hierarchy setup, file loading, external process startup and diagnostics. The median paired ratio is descriptive, **not** a statistically established speedup. Setup sums and peak RSS are in the accompanying JSON. Disparate matrices, RHS and coarse policies must not be conflated with an isolated multilevel gain.

## Provenance

- Original, retained owner WSL logs (outside repository): `{evidence}`.
- Frozen comparison SHA-256: `{result['frozen_comparison_sha256']}`.
- Frozen summary SHA-256: `{result['frozen_summary_sha256']}`.
- All **144 raw log/time file SHA-256 values** are in `benchmarks/evidence/C1G4_AMG_20261113_raw_sha256.csv`; raw log files are not committed.
- Matrix SHA-256: island_shell `{result['matrix_sha256']['island_shell']}`, multi_patch `{result['matrix_sha256']['multi_patch']}`, rotated_anisotropy `{result['matrix_sha256']['rotated_anisotropy']}`.
- Rust example: `crates/hybit/examples/c1h12_a9_rhs_diversity_probe.rs` (unchanged).
- Validation: `benchmarks/scripts/C1G4_current_head_verify.py` and `test_C1G4_current_head_verify.py`; frozen input is checked using existing A9/A8 exact CSR gates.
- Checkpoint command uses `benchmarks/scripts/C1G4_checkpoint.py` to rederive results from raw logs and refuse altered frozen CSV/JSON.

## Follow-up

Do not promote multilevel to production based on this baseline. Next validate the present hierarchy with a real FEM SPD/operator justification and independently matched physical load, and compare with Jacobi-PCG / bounded local-direct Hybrid, plus native AMG external baselines. For difficult ill-conditioned cases use stricter accuracy checks and memory budgets; preserve both convergence and regression cohorts.
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--evidence', required=True, type=Path)
    ap.add_argument('--dest', type=Path, help='new directory for frozen evidence and markdown')
    args = ap.parse_args()
    repo, evidence = args.repo.resolve(), args.evidence.resolve()
    try:
        result, rawhash, rows = compute(repo, evidence)
        print(json.dumps({k: v for k,v in result.items() if k != 'matrix_sha256'}, indent=2))
        if args.dest is not None:
            dest = args.dest.resolve()
            if dest.exists():
                raise ValueError('destination must not exist')
            dest.mkdir(parents=True)
            (dest / 'C1G4_AMG_20261113_summary.json').write_text(
                json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            import shutil
            shutil.copyfile(evidence / 'c1g4_current_head_comparison.csv',
                            dest / 'C1G4_AMG_20261113_comparison.csv')
            (dest / 'C1G4_AMG_20261113_raw_sha256.csv').write_text(rawhash, encoding='utf-8')
            (dest / 'C1G4_AMG_REBASELINE_20261113.md').write_text(md_report(result, evidence), encoding='utf-8')
        print('C1-G4 CHECKPOINT: original CSR + frozen output + 144 raw-file hash verification PASS')
        return 0
    except (ValueError, OSError, KeyError, ImportError, UnicodeError, TypeError) as exc:
        print(f'C1-G4 CHECKPOINT FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
