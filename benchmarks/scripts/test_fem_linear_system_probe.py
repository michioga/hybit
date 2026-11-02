"""Offline unit tests for the FEM-agnostic input adapter (stdlib only)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fem_linear_system_probe as probe


class FemProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mm = self.root / 'sample.mtx'
        self.rhs = self.root / 'sample.rhs'
        self.mm.write_text('%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 2.0\n2 1 -1.0\n2 2 2.0\n')
        self.rhs.write_text('1\n2\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_generic_symmetric_matrix_and_vector(self):
        mm = probe.audit_matrix(self.mm)
        b = probe.audit_rhs(self.rhs, 2)
        self.assertEqual((mm['rows'], mm['stored_entries'], mm['symmetry_declared']), (2, 3, 'symmetric'))
        self.assertEqual(b['length'], 2)

    def test_rejects_bad_counts_and_nonfinite(self):
        self.rhs.write_text('1\nNaN\n')
        with self.assertRaises(ValueError):
            probe.audit_rhs(self.rhs, 2)
        self.mm.write_text('%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 2.0\n2 2 3.0\n')
        with self.assertRaises(ValueError):
            probe.audit_matrix(self.mm)

    def test_frontistr_dump_pairing_and_rank_rejection(self):
        (self.root / 'dump_matrix_5_0.mm').write_bytes(self.mm.read_bytes())
        (self.root / 'dump_matrix_5_0.rhs').write_bytes(self.rhs.read_bytes())
        a, b, call = probe.discover(self.root, None)
        self.assertEqual((a.name, b.name, call), ('dump_matrix_5_0.mm', 'dump_matrix_5_0.rhs', 5))
        (self.root / 'dump_matrix_5_1.mm').write_bytes(self.mm.read_bytes())
        with self.assertRaisesRegex(ValueError, 'distributed rank'):
            probe.discover(self.root, None)

    def test_status_not_merely_exit_code(self):
        raw = 'status : MaxIterations\nverified residual : 1e-2\n'
        self.assertFalse(probe.collect(raw, 1e-8, 0)['numerical_gate'])
        raw = 'status : Converged\nverified residual : 9e-9\n'
        self.assertTrue(probe.collect(raw, 1e-8, 0)['numerical_gate'])

    def test_generic_cli_audit(self):
        argv = ['fem_linear_system_probe.py', '--matrix', str(self.mm), '--rhs', str(self.rhs)]
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()) as output:
            status = probe.main()
        self.assertEqual(status, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['host_adapter'], 'generic-fem-matrix-market')
        self.assertEqual(result['spd_status'], 'NOT_CERTIFIED')


if __name__ == '__main__':
    unittest.main()
