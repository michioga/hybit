#!/usr/bin/env python3
"""Unit checks for the C1-H7 evidence-only generator/parser."""
import unittest
import math
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C1H7_sensitivity import system, emit_case, parse_output, N, LOCAL


class C1H7Tests(unittest.TestCase):
    def test_spd_local_energy_and_symmetry(self):
        for bridge in (1e-6, 1e-4, 0.5):
            rows, _, _ = system('local', bridge, 'smooth')
            self.assertEqual(len(rows), N)
            for i, row in enumerate(rows):
                for j, value in row.items():
                    self.assertEqual(value, rows[j][i])
            # Dirichlet first-block SPD plus strictly positive diagonal outside,
            # plus positive semidefinite weak-edge Laplacian; Gershgorin is not
            # sufficient at interior nodes but the energy construction is.
            for shape in ([1.0]*N, [math.sin(0.19*i) for i in range(N)],
                          [float(i%13==0) for i in range(N)]):
                energy = sum(x*sum(v*shape[j] for j,v in row.items())
                             for x, row in zip(shape, rows))
                self.assertGreater(energy, 0.0)
            self.assertEqual(len(rows[LOCAL]), 2)

    def test_known_nnzs_and_rhs(self):
        for kind, nnz in [('local',352),('global',478)]:
            rows,b,x = system(kind,1e-4,'smooth')
            self.assertEqual(sum(map(len,rows)),nnz)
            self.assertTrue(any(v!=0 for v in b))
            self.assertEqual(len(x),N)
        with tempfile.TemporaryDirectory() as d:
            base=emit_case(Path(d),'local',1e-4,'smooth')
            self.assertTrue(base.with_suffix('.mtx').is_file())
            self.assertEqual(len((Path(d)/(base.name+'.rhs.txt')).read_text().splitlines()),N)

    def test_generator_rejects_invalid(self):
        for x in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                system('local',x,'smooth')

    @staticmethod
    def fixture(plain='MaxIterations', auto='Converged', esc=1):
        return f'''[1/2] Plain Jacobi-PCG
status             : {plain}
iterations         : 30
verified residual  : 2.557405e-7
relative x error   : 5.485638e-6
[2/2] HyBIT Auto
status             : {auto}
iterations         : {14 if esc else 30}
verified residual  : {"2.079629e-14" if esc else "2.557405e-7"}
relative x error   : {"1.415770e-13" if esc else "5.485638e-6"}
escalations        : {esc}
factor memory      : {"0.039" if esc else "0.000"} MiB
total wall         : 0.205 ms
'''

    def test_parser_golden(self):
        v=parse_output(self.fixture())
        self.assertTrue(v['rescue'])
        self.assertFalse(v['unnecessary_promotion'])

    def test_parser_no_promotion(self):
        v=parse_output(self.fixture('MaxIterations','MaxIterations',0))
        self.assertFalse(v['rescue'])
        self.assertEqual(v['escalations'],0)

    def test_parser_rejects_invariant_and_nonfinite(self):
        with self.assertRaises(ValueError):
            parse_output(self.fixture('MaxIterations','Converged',0))
        with self.assertRaises(ValueError):
            parse_output(self.fixture().replace('2.079629e-14','nan'))


if __name__=='__main__':
    unittest.main(verbosity=2)
