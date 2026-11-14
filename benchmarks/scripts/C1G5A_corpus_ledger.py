#!/usr/bin/env python3
"""C1-G5A: strict, read-only offline ledger from frozen C1-G4 experiment data.

This is a research data contract, NOT a trained or deployed autonomous selector.
Only pre-solve facts may be provided to the deliberately constant shadow control.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path

CASES = ('island_shell', 'multi_patch', 'rotated_anisotropy')
CONFIGS = ('graph_piecewise', 'strong_graph_piecewise', 'strong_graph_smoothed')
RHS = ('ones', 'index_smooth', 'seeded_random', 'point_load')
POLICIES = ('two_level', 'multilevel')
BASELINE = 'two_level:graph_piecewise'
SOURCE_HEAD = '37bc52dfb265123267081afc58bd2e6e7b0b6c97'
SOURCE_CSV = 'benchmarks/evidence/C1G4_AMG_20261113_comparison.csv'
SOURCE_SUMMARY = 'benchmarks/evidence/C1G4_AMG_20261113_summary.json'
CASE_NNZ = {'island_shell': 103104, 'multi_patch': 103104, 'rotated_anisotropy': 144002}
CASE_HASH = {
    'island_shell': '2d3f2834e3737610357525a3fdae9abaebb9ee21969d1800b0bd9c15414be45f',
    'multi_patch': '06caafe4c2d82a849250e882f7dcaf66eaf9f51632e699308ce0e01a662e490d',
    'rotated_anisotropy': '03abf3e7d98f840dbb4a2ed09282d7abfbf28c928cff18837f0087af4fb51c42',
}
EXTERNAL = {
    # Local owner-audited copies, NOT independent SPD proof or paired physical RHS.
    'boneS01': ('Oberwolfach_bone', 127224, 5516602, '158c2a260c1d326f4ce66258ba48f218be6d09761bb4012045488246d5874461'),
    'cant': ('Williams_cant', 62451, 4007383, '67556f4af0534ba28c283d788530fada209f3994d448dbc189ef40b91e87486d'),
    'thermal1': ('Schmid_thermal', 82654, 574458, '260f4a65c7dd22ce6e87b1035169c14d53fbe44639f0005e46207e057fce70ee'),
}
INPUT_COLUMNS = (
    'case', 'arm', 'rhs_mode', 'policy', 'rhs_hash', 'status', 'iterations',
    'verified_true_relative', 'relative_x_error', 'levels', 'coarse_factor_bytes',
    'scratch_bytes', 'setup_ms', 'solve_ms', 'total_wall_ms', 'elapsed_s', 'peak_rss_kib',
)
INVENTORY_COLUMNS = ('matrix_id', 'family_id', 'origin', 'matrix_contract', 'spd_evidence',
                     'nrows', 'nnz', 'matrix_sha256', 'rhs_availability', 'measurement_coverage', 'provenance')
LEDGER_COLUMNS = (
    'problem_id', 'matrix_id', 'family_id', 'matrix_sha256', 'matrix_contract',
    'nrows', 'nnz', 'rhs_mode_offline_label', 'rhs_hash_provenance', 'rhs_type',
    'requested_tol', 'max_iterations', 'threads', 'reuse_count', 'candidate_id',
    'solver_class', 'preconditioner', 'preconditioner_configuration', 'implementation_scope',
    'measurement_status', 'solve_status', 'iterations', 'verified_true_relative',
    'relative_x_error_offline_only', 'load_ms', 'setup_ms', 'solve_ms',
    'inprocess_total_wall_ms', 'peak_rss_kib', 'coarse_factor_bytes',
    'scratch_bytes', 'source_head', 'source_csv_sha256', 'comparability_group',
)
DECISION_COLUMNS = (
    'problem_id', 'matrix_id', 'family_id', 'rhs_offline_label', 'observed_arm_count',
    'missing_arm_count', 'shadow_arm', 'shadow_reason', 'shadow_cost_ms',
    'observed_best_arm', 'observed_best_cost_ms', 'shadow_regret_ms',
    'shadow_regret_ratio', 'oracle_qualification', 'heldout_eligibility',
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline='')
    out = csv.DictWriter(stream, columns, lineterminator='\n', extrasaction='raise')
    out.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise ValueError('internal ledger schema mismatch')
        out.writerow(row)
    return stream.getvalue().encode('utf-8')


def checked_float(value: str, field: str, minimum: float = 0.0) -> float:
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f'non-finite or out-of-range {field}: {value!r}')
    return result


def checked_int(value: str, field: str, minimum: int = 0) -> int:
    result = int(value)
    if str(result) != str(value) or result < minimum:
        raise ValueError(f'invalid {field}: {value!r}')
    return result


def check_input(data: bytes, frozen: dict) -> list[dict[str, str]]:
    if frozen.get('source_head') != SOURCE_HEAD or frozen.get('numerical_gate') != 'PASS_ORIGINAL_CSR_AND_GRID':
        raise ValueError('unrecognized C1-G4 source provenance/numerical gate')
    if frozen.get('matrix_sha256') != CASE_HASH:
        raise ValueError('source matrix hashes changed')
    policy = frozen.get('numerical_policy', {})
    if policy != {'tol': 1e-10, 'max_iterations': 600, 'threads': 1, 'rhs': list(RHS)}:
        raise ValueError('input numerical policy changed')
    if frozen.get('runs') != 72 or frozen.get('pairs') != 36 or frozen.get('outcomes') != {'BOTH_CONVERGED': 36}:
        raise ValueError('frozen C1-G4 acceptance count changed')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as ex:
        raise ValueError('source CSV is not UTF-8') from ex
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != list(INPUT_COLUMNS):
        raise ValueError('frozen C1-G4 schema changed')
    indexed = {}
    rhs_fingerprints = {}
    for row in reader:
        if any(v is None for v in row.values()) or None in row or set(row) != set(INPUT_COLUMNS):
            raise ValueError('incomplete source row')
        key = tuple(row[k] for k in ('case', 'arm', 'rhs_mode', 'policy'))
        case, arm, rhs, policy_name = key
        if case not in CASES or arm not in CONFIGS or rhs not in RHS or policy_name not in POLICIES:
            raise ValueError('unexpected case/configuration/RHS/preconditioner')
        if key in indexed:
            raise ValueError('duplicate source numerical identity')
        if not (len(row['rhs_hash']) == 16 and all(c in '0123456789abcdef' for c in row['rhs_hash'])):
            raise ValueError('invalid RHS fingerprint')
        if (case, rhs) in rhs_fingerprints and rhs_fingerprints[case, rhs] != row['rhs_hash']:
            raise ValueError('RHS fingerprint changed between candidates')
        rhs_fingerprints[case, rhs] = row['rhs_hash']
        if row['status'] != 'Converged':
            raise ValueError('accepted frozen record does not converge')
        checked_int(row['iterations'], 'iterations', 0)
        if int(row['iterations']) > 600:
            raise ValueError('iteration budget exceeded')
        checked_int(row['levels'], 'levels', 2)
        checked_int(row['peak_rss_kib'], 'peak_rss_kib', 1)
        checked_int(row['coarse_factor_bytes'], 'coarse_factor_bytes', 1)
        checked_int(row['scratch_bytes'], 'scratch_bytes')
        residual = checked_float(row['verified_true_relative'], 'verified_true_relative')
        if residual > 1e-10:
            raise ValueError('false convergence in ledger source')
        for field in ('setup_ms', 'solve_ms', 'total_wall_ms', 'elapsed_s'):
            checked_float(row[field], field, 1e-15 if field != 'elapsed_s' else 0.0)
        if float(row['total_wall_ms']) + 1e-5 < float(row['setup_ms']) + float(row['solve_ms']):
            raise ValueError('total wall understated')
        if rhs == 'point_load':
            if row['relative_x_error'] != 'n/a':
                raise ValueError('point_load cannot claim exact manufactured x')
        else:
            checked_float(row['relative_x_error'], 'relative_x_error')
        if policy_name == 'two_level' and int(row['levels']) != 2:
            raise ValueError('wrong two-level hierarchy')
        if policy_name == 'multilevel' and int(row['levels']) < 3:
            raise ValueError('wrong multilevel hierarchy')
        indexed[key] = row
    expected = {(a, b, c, d) for a in CASES for b in CONFIGS for c in RHS for d in POLICIES}
    if indexed.keys() != expected:
        raise ValueError(f'incomplete source grid: missing={len(expected - indexed.keys())} extra={len(indexed.keys() - expected)}')
    for case in CASES:
        fingerprints = [rhs_fingerprints[case, r] for r in RHS]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError('distinct RHS modes share fingerprint')
    if sum(int(r['iterations']) for r in indexed.values() if r['policy'] == 'two_level') != frozen['group_statistics']['all']['two_level_iterations_sum']:
        raise ValueError('two-level iteration total differs from frozen summary')
    if sum(int(r['iterations']) for r in indexed.values() if r['policy'] == 'multilevel') != frozen['group_statistics']['all']['multilevel_iterations_sum']:
        raise ValueError('multilevel iteration total differs from frozen summary')
    for field, arm in [('two_level_solve_ms_sum', 'two_level'), ('multilevel_solve_ms_sum', 'multilevel')]:
        actual = sum(float(r['solve_ms']) for r in indexed.values() if r['policy'] == arm)
        if not math.isclose(actual, frozen['group_statistics']['all'][field], rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError('solve time differs from frozen summary')
    return [indexed[key] for key in sorted(indexed)]


def matrix_inventory() -> list[dict]:
    result = []
    for case in CASES:
        result.append(dict(matrix_id=f'c1g_n144/{case}', family_id='constructed_c1g_n144',
                           origin='constructed_fem', matrix_contract='SPD_CANDIDATE',
                           spd_evidence='source_generator_contract_not_independent_global_spd_proof',
                           nrows=20736, nnz=CASE_NNZ[case], matrix_sha256=CASE_HASH[case],
                           rhs_availability='4_research_rhs_modes', measurement_coverage='TWO_LEVEL_AND_MULTILEVEL',
                           provenance='C1G4_20261113'))
    for case, (family, nrows, nnz, digest) in EXTERNAL.items():
        result.append(dict(matrix_id=f'suitesparse/{case}', family_id=f'suitesparse_{family}',
                           origin='external_fem_matrix', matrix_contract='SPD_UNVERIFIED',
                           spd_evidence='owner_audit_exact_symmetric_not_spd_certified',
                           nrows=nrows, nnz=nnz, matrix_sha256=digest,
                           rhs_availability='NO_PAIRED_RHS', measurement_coverage='NOT_RUN',
                           provenance='owner_matrix_audit_and_sha256_20261113'))
    return result


def presolve_features(matrix: dict, tol: float, max_iterations: int) -> dict:
    """Only public pre-solve properties. No labels, solver outcomes, hashes, or RHS mode."""
    return {'matrix_contract': matrix['matrix_contract'], 'nrows': matrix['nrows'],
            'nnz': matrix['nnz'], 'requested_tol': tol, 'max_iterations': max_iterations}


def shadow_constant_control(features: dict) -> tuple[str, str]:
    """Frozen negative control, not a learned or deployed selector."""
    if features['matrix_contract'] != 'SPD_CANDIDATE':
        return ('NO_RECOMMENDATION', 'UNVERIFIED_OR_UNSUPPORTED_MATRIX_CLASS')
    return (BASELINE, 'FIXED_UNTRAINED_TWO_LEVEL_CONTROL_NO_PERFORMANCE_PREDICTION')


def make_outputs(source_csv: bytes, source_summary: bytes) -> dict[str, bytes]:
    frozen = json.loads(source_summary)
    src_sha = sha256(source_csv)
    rows = check_input(source_csv, frozen)
    inventory = matrix_inventory()
    inventory_by_case = {case: inventory[i] for i, case in enumerate(CASES)}
    lookup = {(r['case'], r['arm'], r['rhs_mode'], r['policy']): r for r in rows}
    ledger = []
    decisions = []
    oracle_counter = Counter()
    shadow_regrets = []
    for case in CASES:
        mat = inventory_by_case[case]
        for rhs in RHS:
            problem_id = f'c1g4/{case}/{rhs}'
            selected = []
            fingerprint = lookup[(case, CONFIGS[0], rhs, 'two_level')]['rhs_hash']
            for arm in CONFIGS:
                for policy in POLICIES:
                    src = lookup[(case, arm, rhs, policy)]
                    candidate_id = f'{policy}:{arm}'
                    selected.append((candidate_id, float(src['total_wall_ms'])))
                    ledger.append(dict(
                        problem_id=problem_id, matrix_id=mat['matrix_id'], family_id=mat['family_id'],
                        matrix_sha256=mat['matrix_sha256'], matrix_contract=mat['matrix_contract'],
                        nrows=mat['nrows'], nnz=mat['nnz'], rhs_mode_offline_label=rhs,
                        rhs_hash_provenance=fingerprint,
                        rhs_type='ALGEBRAIC_POINT_LOAD' if rhs == 'point_load' else 'MANUFACTURED',
                        requested_tol='1e-10', max_iterations=600, threads=1, reuse_count=1,
                        candidate_id=candidate_id, solver_class='PCG', preconditioner=policy,
                        preconditioner_configuration=arm,
                        implementation_scope='RESEARCH_A9_PROBE', measurement_status='MEASURED',
                        solve_status=src['status'], iterations=src['iterations'],
                        verified_true_relative=src['verified_true_relative'],
                        relative_x_error_offline_only='' if rhs == 'point_load' else src['relative_x_error'],
                        load_ms='', # not exported by C1G4 frozen CSV
                        setup_ms=src['setup_ms'], solve_ms=src['solve_ms'],
                        inprocess_total_wall_ms=src['total_wall_ms'], peak_rss_kib=src['peak_rss_kib'],
                        coarse_factor_bytes=src['coarse_factor_bytes'], scratch_bytes=src['scratch_bytes'],
                        source_head=SOURCE_HEAD, source_csv_sha256=src_sha,
                        comparability_group=problem_id,
                    ))
            for pending in ('jacobi_pcg', 'hybrid_local_cholesky'):
                ledger.append(dict(
                    problem_id=problem_id, matrix_id=mat['matrix_id'], family_id=mat['family_id'],
                    matrix_sha256=mat['matrix_sha256'], matrix_contract=mat['matrix_contract'],
                    nrows=mat['nrows'], nnz=mat['nnz'], rhs_mode_offline_label=rhs,
                    rhs_hash_provenance=fingerprint,
                    rhs_type='ALGEBRAIC_POINT_LOAD' if rhs == 'point_load' else 'MANUFACTURED',
                    requested_tol='1e-10', max_iterations=600, threads=1, reuse_count=1,
                    candidate_id=pending, solver_class='PCG', preconditioner=pending,
                    preconditioner_configuration='NOT_APPLICABLE', implementation_scope='NOT_RUN',
                    measurement_status='NOT_RUN', solve_status='NOT_RUN', iterations='',
                    verified_true_relative='', relative_x_error_offline_only='',
                    load_ms='', setup_ms='', solve_ms='', inprocess_total_wall_ms='',
                    peak_rss_kib='', coarse_factor_bytes='', scratch_bytes='',
                    source_head='', source_csv_sha256='', comparability_group=problem_id,
                ))
            # Observed-only hindsight label: cannot be treated as safe production oracle.
            observed_best = min(selected, key=lambda x: (x[1], x[0]))
            features = presolve_features(mat, 1e-10, 600)
            shadow, reason = shadow_constant_control(features)
            shadow_cost = dict(selected)[shadow]
            observed_regret = shadow_cost - observed_best[1]
            if observed_regret < -1e-9:
                raise ValueError('invalid observed regret')
            oracle_counter[observed_best[0]] += 1
            shadow_regrets.append(shadow_cost / observed_best[1])
            decisions.append(dict(
                problem_id=problem_id, matrix_id=mat['matrix_id'], family_id=mat['family_id'],
                rhs_offline_label=rhs, observed_arm_count=6, missing_arm_count=2,
                shadow_arm=shadow, shadow_reason=reason, shadow_cost_ms=f'{shadow_cost:.6f}',
                observed_best_arm=observed_best[0], observed_best_cost_ms=f'{observed_best[1]:.6f}',
                shadow_regret_ms=f'{max(0.0, observed_regret):.6f}',
                shadow_regret_ratio=f'{shadow_cost / observed_best[1]:.9f}',
                oracle_qualification='OBSERVED_ONLY_INPROCESS_NO_MEMORY_BUDGET_NO_REUSE',
                heldout_eligibility='NO_ONE_MEASURED_MATRIX_FAMILY_ONLY',
            ))
    counts = Counter(row['measurement_status'] for row in ledger)
    assert counts == {'MEASURED': 72, 'NOT_RUN': 24}, counts
    assert len(decisions) == 12
    report = {
        'checkpoint': 'C1G5A_UNIFIED_CORPUS_SHADOW_20261114',
        'source_head': SOURCE_HEAD,
        'source_comparison_sha256': src_sha,
        'source_original_csr_gate': frozen['numerical_gate'],
        'scope': 'SPD research only; GeneralSquare is separate; no production routing changes',
        'matrix_inventory': {'constructed_measured': 3, 'external_matrix_only': 3},
        'distinct_measured_families': 1,
        'observations': {'MEASURED': 72, 'NOT_RUN': 24, 'groups': 12, 'external_unpaired_matrices': 3},
        'measured_candidates_per_group': 6,
        'planned_missing_candidates_per_group': 2,
        'shadow_policy': BASELINE,
        'shadow_mode': 'CONSTANT_NEGATIVE_CONTROL_NO_LEARNING_NO_PRODUCTION_RECOMMENDATION',
        'shadow_regret_ratio_median_descriptive': statistics.median(shadow_regrets),
        'observed_best_counts_descriptive': dict(sorted(oracle_counter.items())),
        'oracle_scope': 'INCOMPLETE_OBSERVED_ONLY; one OS process/arm; in-process total_wall_ms; no true memory budget or amortized RHS',
        'held_out_family_gate': 'BLOCKED_ONLY_ONE_MEASURED_FAMILY',
        'feature_allowlist': ['matrix_contract', 'nrows', 'nnz', 'requested_tol', 'max_iterations'],
        'forbidden_inference_features': ['rhs_mode_offline_label', 'rhs_hash_provenance',
            'source_head', 'source_csv_sha256', 'iterations', 'verified_true_relative',
            'relative_x_error_offline_only', 'solve_ms', 'setup_ms', 'inprocess_total_wall_ms',
            'peak_rss_kib', 'observed_best_arm'],
        'limitations': [
            'No Jacobi/Hybrid matched measurements; NOT_RUN is not a failure',
            'All 3 measured matrices are one constructed-family group; no held-out validation',
            'No external matrix SPD proof, paired physical RHS, or proven forward-error bound',
            'No controlled repeated performance, hardware/CPU affinity record or RHS reuse benchmark',
            'No real selection prediction: shadow is a constant sanity/control policy',
        ],
    }
    prefix = 'benchmarks/evidence/C1G5A_20261114_'
    return {
        prefix + 'matrix_inventory.csv': csv_bytes(inventory, INVENTORY_COLUMNS),
        prefix + 'ledger.csv': csv_bytes(ledger, LEDGER_COLUMNS),
        prefix + 'shadow_audit.csv': csv_bytes(decisions, DECISION_COLUMNS),
        prefix + 'summary.json': (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'),
    }


def from_repo(repo: Path) -> dict[str, bytes]:
    return make_outputs((repo / SOURCE_CSV).read_bytes(), (repo / SOURCE_SUMMARY).read_bytes())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--check', action='store_true', help='compare generated data with installed artifacts, no writes')
    p.add_argument('--write-to', type=Path, help='create outputs in a new destination directory')
    args = p.parse_args()
    if args.check == (args.write_to is not None):
        p.error('choose exactly one of --check or --write-to')
    try:
        generated = from_repo(args.repo)
        if args.check:
            for relative, content in generated.items():
                if (args.repo / relative).read_bytes() != content:
                    raise ValueError(f'ledger file mismatch: {relative}')
            print('C1-G5A LEDGER VERIFY PASS: 72 measured + 24 NOT_RUN, 12 constant-shadow audits')
        else:
            dest = args.write_to
            if dest.exists():
                raise ValueError('output directory already exists')
            dest.mkdir(parents=True)
            for name, content in generated.items():
                (dest / Path(name).name).write_bytes(content)
            print(f'C1-G5A generated read-only corpus at {dest}')
        return 0
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as ex:
        print(f'C1-G5A LEDGER FAIL: {ex}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
