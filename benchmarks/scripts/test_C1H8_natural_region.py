"""C1-H8 generator and classifier tests, stdlib only."""
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks.scripts.C1H8_natural_region import (
    CAPACITIES, LOCAL_SIZES, RHS, classify, grid, make_system, write_case,
)


class NaturalRegionTests(unittest.TestCase):
    def test_sweep_size_and_controls(self):
        self.assertEqual(len(grid('full')), 36)
        self.assertEqual(len(grid('smoke')), 5)
        self.assertIn((96, 'smooth', 128), grid('smoke'))
        self.assertEqual(CAPACITIES, LOCAL_SIZES)
        self.assertEqual(len(RHS), 3)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            make_system(65, 'smooth')
        with self.assertRaises(ValueError):
            make_system(96, 'unknown')
        with self.assertRaises(ValueError):
            grid('unknown')

    def test_local_symmetric_spd_energy(self):
        for local in LOCAL_SIZES:
            rows, b, xref = make_system(local, 'smooth')
            self.assertEqual(len(rows), 160)
            self.assertEqual(len(xref), 160)
            self.assertEqual(len(b), 160)
            self.assertEqual(sum(len(row) for row in rows), 2 * local + 160)
            self.assertEqual(rows[local - 1][local], -1e-4)
            self.assertTrue(all(rows[i][j] == rows[j][i] for i in range(160)
                                for j in rows[i]))
            for i in range(160):
                self.assertEqual(b[i], sum(v * xref[j] for j, v in rows[i].items()))
            # Strict Dirichlet positive chain plus PSD weak-edge perturbation.
            for k in (1, 2, 3, 7):
                x = [math.sin(k * 0.31 * i) + 0.11 for i in range(160)]
                energy = sum(x[i] * v * x[j] for i, row in enumerate(rows)
                             for j, v in row.items())
                self.assertGreater(energy, 0.0)

    def test_global_structure(self):
        rows, _, _ = make_system(None, 'flat')
        self.assertEqual(sum(map(len, rows)), 478)
        self.assertTrue(all(rows[i][i] == 2.0 for i in range(160)))

    def test_emitted_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            base = write_case(Path(temp), 64, 'oscillatory')
            lines = (Path(str(base) + '.mtx')).read_text().splitlines()
            self.assertEqual(lines[0], '%%MatrixMarket matrix coordinate real general')
            self.assertEqual(lines[2], '160 160 288')
            self.assertEqual(len((Path(str(base) + '.rhs.txt')).read_text().splitlines()), 160)

    def test_classifier(self):
        rescued = {'rescue': True, 'unnecessary_promotion': False, 'escalations': 1}
        retained = {'rescue': False, 'unnecessary_promotion': False, 'escalations': 0}
        self.assertEqual(classify(rescued, 96, 128), 'rescue')
        self.assertEqual(classify(retained, 96, 64), 'capped_keep')
        self.assertEqual(classify(retained, None, 128), 'global_keep')
        self.assertEqual(classify(rescued, 96, 64), 'truncated_promotion')
        self.assertEqual(classify(rescued, None, 128), 'global_promotion')


if __name__ == '__main__':
    unittest.main()
