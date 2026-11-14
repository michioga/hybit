"""Fail-closed owner source evidence regression tests for C1-G5F."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmarks.scripts import C1G5F_feature_evidence as evidence


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upload_root = Path(__file__).resolve().parents[3] / '_no_uploads_in_git'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.logs = self.root / 'raw'
        self.logs.mkdir()
        self.g4 = self.root / 'scalar'
        self.g4.mkdir()
        self.g5e = self.root / 'elastic'
        self.g5e.mkdir()

    def source(self, case):
        name, family, matrix, sha, nr, nnz, offdiag, source_hash = case
        # Build fixture directly, no private owner paths or training labels.
        return (
            f'C1G5F|path={(self.g4 if family == "scalar_p1_diffusion" else self.g5e) / matrix}'
            f'|rows={nr}|nnz={nnz}|offdiag={offdiag}|positive_fraction=0.200000000000e0'
            f'|nondominant_rows=Some({nr//2})|max_offdiag_ratio=Some(1.2)'
            f'|strong_fraction=Some(0.20)|invalid_ratio=false|rhs_nnz_fraction=0.5'
            f'|diagonal_spread=Some(1.0)|shadow_reason=InsufficientIndependentEvidence'
            f'|research_rank=None|production_recommended=None|audit_ms=0.5\n'
        ).encode()

    def test_case_count_and_origins(self):
        self.assertEqual(len(evidence.CASES), 6)
        self.assertEqual(len({r[0] for r in evidence.CASES}), 6)
        self.assertEqual({r[1] for r in evidence.CASES},
                         {'scalar_p1_diffusion', 'tri_p1_plane_strain_elasticity'})

    def test_all_real_owner_log_hashes_are_pinned(self):
        self.assertEqual(len({r[-1] for r in evidence.CASES}), 6)
        for item in evidence.CASES:
            self.assertEqual(len(item[-1]), 64)
            self.assertEqual(len(item[3]), 64)

    def test_log_records_reject_hash_modified(self):
        c = evidence.CASES[0]
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            evidence.parse_log(self.source(c), c, self.g4, self.g5e)

    def test_log_contract_with_matching_fingerprint(self):
        c = evidence.CASES[0]
        raw = self.source(c)
        altered = (*c[:7], evidence.sha(raw))
        result = evidence.parse_log(raw, altered, self.g4, self.g5e)
        self.assertEqual(result['nondominant_rows'], 10368)
        self.assertEqual(result['shadow_reason'], 'InsufficientIndependentEvidence')
        self.assertEqual(result['nondominant_fraction'], '0.500000000000')

    def test_mismatched_matrix_path_is_rejected_even_with_rehashed_log(self):
        c = evidence.CASES[0]
        bad = self.source(c).replace(b'c1g_p1_island_shell_n144.mtx', b'wrong.mtx')
        altered = (*c[:7], evidence.sha(bad))
        with self.assertRaisesRegex(ValueError, 'matrix identity'):
            evidence.parse_log(bad, altered, self.g4, self.g5e)

    def test_duplicate_field_rejected_with_rehashed_log(self):
        c = evidence.CASES[0]
        bad = self.source(c).replace(b'|nnz=', b'|rows=999|nnz=')
        altered = (*c[:7], evidence.sha(bad))
        with self.assertRaisesRegex(ValueError, 'duplicate field'):
            evidence.parse_log(bad, altered, self.g4, self.g5e)

    def test_false_promotion_or_research_rank_rejected(self):
        c = evidence.CASES[0]
        for wrong in (b'production_recommended=Some(JacobiPcg)', b'research_rank=Some(JacobiPcg)'):
            if wrong.startswith(b'production'):
                raw = self.source(c).replace(b'production_recommended=None', wrong)
            else:
                raw = self.source(c).replace(b'research_rank=None', wrong)
            altered = (*c[:7], evidence.sha(raw))
            with self.assertRaisesRegex(ValueError, 'fail-closed'):
                evidence.parse_log(raw, altered, self.g4, self.g5e)

    def test_nondominant_count_exceed_dimension_rejected(self):
        c = evidence.CASES[0]
        raw = self.source(c).replace(b'nondominant_rows=Some(10368)', b'nondominant_rows=Some(100000)')
        altered = (*c[:7], evidence.sha(raw))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            evidence.parse_log(raw, altered, self.g4, self.g5e)

    def test_nan_or_inf_rejected(self):
        c = evidence.CASES[0]
        for raw in (self.source(c).replace(b'audit_ms=0.5', b'audit_ms=nan'),
                    self.source(c).replace(b'strong_fraction=Some(0.20)', b'strong_fraction=Some(inf)')):
            altered = (*c[:7], evidence.sha(raw))
            with self.assertRaises(ValueError):
                evidence.parse_log(raw, altered, self.g4, self.g5e)

    def test_fraction_boundaries(self):
        self.assertEqual(evidence.read_fraction('0', 'p'), 0)
        self.assertEqual(evidence.read_fraction('1', 'p'), 1)
        with self.assertRaises(ValueError):
            evidence.read_fraction('1.1', 'p')
        with self.assertRaises(ValueError):
            evidence.read_fraction('-0.1', 'p')

    def test_summary_fails_closed(self):
        rows = []
        for c in evidence.CASES:
            raw = self.source(c)
            rows.append(evidence.parse_log(raw, (*c[:7], evidence.sha(raw)), self.g4, self.g5e))
        raw = [{'filename': c[0] + '.log', 'bytes': 1, 'sha256': c[-1]}
               for c in evidence.CASES]
        outputs = evidence.build_outputs(rows, raw)
        self.assertEqual(len(outputs), 4)
        import json
        summary = json.loads(outputs['benchmarks/evidence/C1G5F_20261114_summary.json'])
        self.assertFalse(summary['production_promotion_allowed'])
        self.assertFalse(summary['independent_heldout_passed'])
        self.assertFalse(summary['training_performed'])
        self.assertEqual(summary['generator_families'], 2)

    def test_original_sha_index_is_required(self):
        self.assertEqual(len(evidence.EXPECTED_RAW), 8)
        self.assertIn('HEAD.txt', evidence.EXPECTED_RAW)
        self.assertIn('SHA256_LOGS.txt', evidence.EXPECTED_RAW)

    def test_output_schema_is_unique(self):
        self.assertEqual(len(evidence.CSV_FIELDS), len(set(evidence.CSV_FIELDS)))
        self.assertEqual(len(evidence.FIELDS), len(set(evidence.FIELDS)))


if __name__ == '__main__':
    unittest.main()
