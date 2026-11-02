"""C1-H11 unit tests: parse and classify, never claim physical provenance."""
import tempfile
import unittest
from pathlib import Path
from benchmarks.scripts.C1H11_real_fem_evidence import load_cases, parsed_output


def log(plain='MaxIterations', auto='Converged', escalations=1,
        plain_res='1.0e-2', auto_res='1.0e-12', xref=True):
    extra = 'relative x error   : 1.0e-10\n' if xref else ''
    return ('[1/2] Plain Jacobi-PCG\n'
            f'status             : {plain}\niterations         : 30\n'
            f'verified residual  : {plain_res}\n'
            'setup / solve      : 0.010 / 0.020 ms\n' + extra +
            '[2/2] HyBIT Auto\n'
            f'status             : {auto}\niterations         : 14\n'
            f'verified residual  : {auto_res}\n'
            f'escalations        : {escalations}\n'
            'factor memory      : 0.039 MiB\n'
            'total wall         : 0.400 ms\n' + extra)

class EvidenceTests(unittest.TestCase):
    def test_rescue(self):
        r = parsed_output(log(), 1e-10, 100, True)
        self.assertEqual(r['class'], 'RESCUE')
        self.assertEqual(r['escalations'], 1)

    def test_false_convergence_rejected(self):
        r = parsed_output(log(auto_res='1e-5'), 1e-10, 100, True)
        self.assertEqual(r['class'], 'FALSE_REPORTED_CONVERGENCE')

    def test_keep_pcg_status_and_iteration_invariant(self):
        with self.assertRaisesRegex(ValueError, 'KEEP_PCG'):
            parsed_output(log(escalations=0), 1e-10, 100, True)

    def test_invalid_or_duplicated_arms(self):
        with self.assertRaises(ValueError):
            parsed_output(log() + log(), 1e-10, 100, True)

    def test_manifest_refuses_missing_evidence_and_no_real_input(self):
        with tempfile.TemporaryDirectory() as d:
            import json
            p = Path(d) / 'manifest.json'
            p.write_text(json.dumps({'cases': [{'id':'fake', 'origin':'external_fem_serial',
                        'matrix':'missing.mtx', 'rhs':'missing.rhs',
                        'provenance':'missing.txt', 'spd_evidence':'missing.txt'}]}))
            with self.assertRaises((FileNotFoundError, ValueError)):
                load_cases(p)

    def test_no_xref_is_not_invented(self):
        r = parsed_output(log(xref=False), 1e-10, 100, False)
        self.assertIsNone(r['auto_forward_error'])

if __name__ == '__main__':
    unittest.main()
